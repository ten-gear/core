import { getBackend } from "./backend/index.js";
import { Parameter } from "./module.js";
import { Scalar } from "./scalar.js";
import { Tensor } from "./tensor.js";
import { TensorBuffer } from "./tensor_data.js";

export abstract class Optimizer{
    parameters: Parameter[];
    constructor(parameters: Parameter[]){
        this.parameters = parameters;
    }

    abstract zeroGrad(): void;
    abstract step(): void;
}

export class SGD extends Optimizer{
    lr: number; // learning rate
    constructor(parameters: Parameter[], lr: number = 1.0){
        super(parameters);
        this.lr = lr;
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad instanceof Tensor)
                    p.value.grad.fill_(0);
                else if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step() {
        for (let p of this.parameters) {
            if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            }
            else if (p.value instanceof Tensor) {
                if (p.value.grad !== undefined) {
                    getBackend().sgd(
                        p.value.data._storage, p.value.data._shape, p.value.data._stride,
                        p.value.grad.data._storage, p.value.grad.data._shape, p.value.grad.data._stride,
                        this.lr,
                    );
                }
            }
        }
    }
}

export class SGDMomentum extends Optimizer{
    lr: number; // learning rate
    beta: number
    prev: Map<Parameter, number | TensorBuffer>;
    constructor(parameters: Parameter[], lr: number = 1.0, beta: number = 0.9){
        super(parameters);
        this.lr = lr;
        this.beta = beta;
        this.prev = new Map<Parameter, number | TensorBuffer>();
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad instanceof Tensor)
                    p.value.grad.fill_(0);
                else if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step(): void{
        for (const p of this.parameters){
            if (p.value instanceof Scalar){
                if (p.value.derivative !== undefined){
                    let oldVelocity = this.prev.get(p);

                    if (typeof oldVelocity !== "number")
                        oldVelocity = 0;

                    let velocity = this.beta * oldVelocity + (1 - this.beta) * p.value.derivative;
                    p.value.data -= this.lr * velocity;

                    this.prev.set(p, velocity);
                }
            }
            else if (p.value instanceof Tensor) {
                if (p.value.grad !== undefined) {
                    let velocity = this.prev.get(p);
                    if (velocity === undefined || typeof velocity === "number") {
                        velocity = getBackend().alloc(p.value.data.size, 0);
                        this.prev.set(p, velocity);
                    }

                    const shape = p.value.data._shape;
                    const stride = p.value.data._stride;
                    getBackend().sgdMomentum(
                        p.value.data._storage, shape, stride,
                        p.value.grad.data._storage, p.value.grad.data._shape, p.value.grad.data._stride,
                        velocity, shape, stride,
                        this.lr, this.beta,
                    );
                }
            }
        }
    }
}

export class AdamW extends Optimizer{
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    m: Map<Parameter, number | TensorBuffer>;
    v: Map<Parameter, number | TensorBuffer>;
    steps: Map<Parameter, number>;

    constructor(parameters: Parameter[], lr: number = 1e-3, beta1: number = 0.9,
        beta2: number = 0.999, eps: number = 1e-8, weightDecay: number = 0.01){
        super(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new Map<Parameter, number | TensorBuffer>();
        this.v = new Map<Parameter, number | TensorBuffer>();
        this.steps = new Map<Parameter, number>();
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad instanceof Tensor)
                    p.value.grad.fill_(0);
                else if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step(): void{
        for (const p of this.parameters){
            if (p.value instanceof Scalar){
                if (p.value.derivative !== undefined){
                    const t = (this.steps.get(p) ?? 0) + 1;
                    const oldM = this.m.get(p);
                    const oldV = this.v.get(p);
                    const m = this.beta1 * (typeof oldM === "number" ? oldM : 0) + (1 - this.beta1) * p.value.derivative;
                    const v = this.beta2 * (typeof oldV === "number" ? oldV : 0) + (1 - this.beta2) * p.value.derivative * p.value.derivative;
                    const mHat = m / (1 - Math.pow(this.beta1, t));
                    const vHat = v / (1 - Math.pow(this.beta2, t));

                    p.value.data = p.value.data * (1 - this.lr * this.weightDecay) - this.lr * mHat / (Math.sqrt(vHat) + this.eps);
                    this.m.set(p, m);
                    this.v.set(p, v);
                    this.steps.set(p, t);
                }
            }
            else if (p.value instanceof Tensor){
                if (p.value.grad !== undefined){
                    let m = this.m.get(p);
                    if (m === undefined || typeof m === "number"){
                        m = getBackend().alloc(p.value.data.size, 0);
                        this.m.set(p, m);
                    }
                    let v = this.v.get(p);
                    if (v === undefined || typeof v === "number"){
                        v = getBackend().alloc(p.value.data.size, 0);
                        this.v.set(p, v);
                    }

                    const t = (this.steps.get(p) ?? 0) + 1;
                    const shape = p.value.data._shape;
                    const stride = p.value.data._stride;
                    getBackend().adamw(
                        p.value.data._storage, shape, stride,
                        p.value.grad.data._storage, p.value.grad.data._shape, p.value.grad.data._stride,
                        m, shape, stride,
                        v, shape, stride,
                        this.lr, this.beta1, this.beta2, this.eps, this.weightDecay, t,
                    );
                    this.steps.set(p, t);
                }
            }
        }
    }
}
