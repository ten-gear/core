import { randomFloat } from "./helperfunctions.js";
import { Parameter } from "./module.js";
import { Layer } from "./model.js";
import { Tensor } from "./tensor.js";
import { flashAttention, logsoftmax, maxpool2d, rope, rparam, dropout } from "./tensor_functions.js";
import { Shape, TensorData, writeStorage } from "./tensor_data.js";

export type WeightInit = (fanIn: number, shape: Shape) => Tensor;

function heParameter(fanIn: number, shape: Shape): Tensor {
    const bound = Math.sqrt(6 / fanIn);
    const data = TensorData.fill(shape);
    const values = new Float32Array(data.size);
    for (let i = 0; i < values.length; i++)
        values[i] = randomFloat(-bound, bound);
    writeStorage(data._storage, values);

    const tensor = new Tensor(data);
    tensor._requiresGrad(true);
    return tensor;
}

export function normalParameter(std: number): WeightInit {
    if (!Number.isFinite(std) || std < 0)
        throw new Error("normalParameter expects a finite non-negative standard deviation");
    return (_fanIn: number, shape: Shape): Tensor => {
        const data = TensorData.fill(shape);
        const values = new Float32Array(data.size);
        for (let i = 0; i < values.length; i += 2) {
            const u1 = Math.max(Math.random(), Number.MIN_VALUE);
            const u2 = Math.random();
            const radius = std * Math.sqrt(-2 * Math.log(u1));
            const angle = 2 * Math.PI * u2;
            values[i] = radius * Math.cos(angle);
            if (i + 1 < values.length)
                values[i + 1] = radius * Math.sin(angle);
        }
        writeStorage(data._storage, values);
        const tensor = new Tensor(data);
        tensor._requiresGrad(true);
        return tensor;
    };
}

function zeroParameter(shape: Shape): Tensor {
    const tensor = new Tensor(TensorData.fill(shape, 0));
    tensor._requiresGrad(true);
    return tensor;
}

function onesParameter(shape: Shape): Tensor {
    const tensor = new Tensor(TensorData.fill(shape, 1));
    tensor._requiresGrad(true);
    return tensor;
}
export function causalMask(seq: number): Tensor {
    const values = new Float32Array(seq * seq);
    for (let t = 0; t < seq; t++)
        for (let s = 0; s < seq; s++)
            if (s > t) values[t * seq + s] = -1e9;
    const mask = new Tensor(TensorData.empty([1, 1, seq, seq]));
    writeStorage(mask.data._storage, values);
    return mask;
}

export function ropeFrequencies(seq: number, headDim: number, theta = 10000): { cos: Tensor; sin: Tensor } {
    if (!Number.isInteger(seq) || seq < 1)
        throw new Error("ropeFrequencies expects a positive integer sequence length");
    if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0)
        throw new Error("ropeFrequencies expects a positive even head dimension");
    if (!Number.isFinite(theta) || theta <= 0)
        throw new Error("ropeFrequencies expects a positive finite theta");

    const half = headDim / 2;
    const cosValues = new Float32Array(seq * half);
    const sinValues = new Float32Array(seq * half);
    for (let i = 0; i < half; i++) {
        const freq = theta ** (-2 * i / headDim);
        for (let t = 0; t < seq; t++) {
            const angle = t * freq;
            cosValues[t * half + i] = Math.cos(angle);
            sinValues[t * half + i] = Math.sin(angle);
        }
    }
    const cos = new Tensor(TensorData.empty([seq, half]));
    const sin = new Tensor(TensorData.empty([seq, half]));
    writeStorage(cos.data._storage, cosValues);
    writeStorage(sin.data._storage, sinValues);
    return { cos, sin };
}

export class RoPE extends Layer {
    readonly seq: number;
    readonly headDim: number;
    readonly theta: number;
    readonly cos: Tensor;
    readonly sin: Tensor;

    constructor(seq: number, headDim: number, theta: number = 10000) {
        super();
        this.seq = seq;
        this.headDim = headDim;
        this.theta = theta;
        const frequencies = ropeFrequencies(seq, headDim, theta);
        this.cos = frequencies.cos;
        this.sin = frequencies.sin;
    }

    forward(input: Tensor, offset: number = 0): Tensor {
        return rope(input, this.sin, this.cos, offset);
    }
}

export class FlashAttention extends Layer {
    readonly causal: boolean;
    readonly scale: number | undefined;

    constructor(causal: boolean = false, scale?: number) {
        super();
        this.causal = causal;
        this.scale = scale;
    }

    forward(query: Tensor, key: Tensor = query, value: Tensor = key): Tensor {
        return flashAttention(query, key, value, this.causal, this.scale);
    }
}

export class Linear extends Layer {
    readonly inSize: number;
    readonly outSize: number;
    weights: Parameter;
    bias: Parameter;

    constructor(inSize: number, outSize: number, init: WeightInit = heParameter) {
        super();
        this.inSize = inSize;
        this.outSize = outSize;
        this.weights = this.addParameter("weights", init(inSize, [inSize, outSize]));
        this.bias = this.addParameter("bias", zeroParameter([outSize]));
    }

    forward(input: Tensor): Tensor {
        const shape = input.shape();
        const rows = input.size() / this.inSize;
        const flat = input.view(rows, this.inSize);
        const out = flat.matmul(this.weights.value as Tensor).add(this.bias.value as Tensor);
        return out.view(...shape.slice(0, -1), this.outSize);
    }
}

export class Conv2d extends Layer {
    readonly inChannels: number;
    readonly outChannels: number;
    readonly kernelHeight: number;
    readonly kernelWidth: number;
    weights: Parameter;
    bias: Parameter;

    constructor(inChannels: number, outChannels: number, kernelHeight: number, kernelWidth: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.weights = this.addParameter(
            "weights",
            heParameter(
                inChannels * kernelHeight * kernelWidth,
                [outChannels, inChannels, kernelHeight, kernelWidth],
            ),
        );
        this.bias = this.addParameter("bias", zeroParameter([outChannels, 1, 1]));
    }

    forward(input: Tensor): Tensor {
        return input
            .conv2d(this.weights.value as Tensor)
            .add(this.bias.value as Tensor);
    }
}

export class ReLU extends Layer {
    forward(input: Tensor): Tensor {
        return input.relu();
    }
}

export class MaxPool2d extends Layer {
    readonly kernel: [number, number];

    constructor(kernel: [number, number]) {
        super();
        this.kernel = [...kernel];
    }

    forward(input: Tensor): Tensor {
        return maxpool2d(input, this.kernel);
    }
}

export class Flatten extends Layer {
    forward(input: Tensor): Tensor {
        if (input.dims() < 2)
            throw new Error("Flatten expects a batch dimension");

        const batch = input.shape()[0];
        return input.view(batch, input.size() / batch);
    }
}

export class LogSoftmax extends Layer {
    readonly dim: number;

    constructor(dim: number) {
        super();
        this.dim = dim;
    }

    forward(input: Tensor): Tensor {
        return logsoftmax(input, this.dim);
    }
}

export class Embedding extends Layer {
    readonly table: Parameter;
    constructor(vocab: number, nEmbd: number, init?: WeightInit) {
        super();
        const table = init === undefined
            ? rparam([vocab, nEmbd], -0.02, 0.02)
            : init(nEmbd, [vocab, nEmbd]);
        this.table = this.addParameter("table", table);
        (this.table.value as Tensor)._requiresGrad(true);
    }
    forward(ids: Tensor): Tensor {
        return (this.table.value as Tensor).embedding(ids);
    }
}

export class LayerNorm extends Layer {
    readonly gamma: Parameter;
    readonly beta: Parameter;
    readonly epsilon: number;
    constructor(nEmbd: number, epsilon = 1e-5) {
        super();
        this.epsilon = epsilon;
        this.gamma = this.addParameter("gamma", onesParameter([nEmbd]));
        this.beta = this.addParameter("beta", zeroParameter([nEmbd]));
    }
    forward(input: Tensor): Tensor {
        return input.layernorm(this.gamma.value as Tensor, this.beta.value as Tensor, this.epsilon);
    }
}

export class MHAFlashAttention extends Layer {
    readonly nHead: number;
    readonly headDim: number;
    readonly q: Linear;
    readonly k: Linear;
    readonly v: Linear;
    readonly out: Linear;
    readonly rope: RoPE;
    readonly dropoutRate: number;
    constructor(
        nEmbd: number,
        nHead: number,
        maxSeq: number,
        dropout = 0,
        init: WeightInit = heParameter,
        outInit: WeightInit = init,
    ) {
        super();
        if (nEmbd % nHead !== 0) throw new Error("nEmbd must divide by nHead");
        this.nHead = nHead;
        this.headDim = nEmbd / nHead;
        this.dropoutRate = dropout;
        this.q = new Linear(nEmbd, nEmbd, init);
        this.k = new Linear(nEmbd, nEmbd, init);
        this.v = new Linear(nEmbd, nEmbd, init);
        this.out = new Linear(nEmbd, nEmbd, outInit);
        this.rope = new RoPE(maxSeq, this.headDim);
    }
    forward(input: Tensor, offset = 0): Tensor {
        const [batch, time, nEmbd] = input.shape();
        const heads = (x: Tensor) =>  x.view(batch, time, this.nHead, this.headDim).permute(0, 2, 1, 3);
        let q = this.rope.forward(heads(this.q.forward(input)), offset);
        let k = this.rope.forward(heads(this.k.forward(input)), offset);
        const v = heads(this.v.forward(input));
        const context = flashAttention(q, k, v, true);
        const mixed = context.permute(0, 2, 1, 3).tocontiguous().view(batch, time, nEmbd);
        return dropout(this.out.forward(mixed), this.dropoutRate, !this.training);
    }
}

export class MLP extends Layer {
    readonly fc: Linear;
    readonly proj: Linear;
    constructor(nEmbd: number, factor = 4, init: WeightInit = heParameter, projInit: WeightInit = init) {
        super();
        this.fc = new Linear(nEmbd, factor * nEmbd, init);
        this.proj = new Linear(factor * nEmbd, nEmbd, projInit);
    }
    forward(input: Tensor): Tensor {
        return this.proj.forward(this.fc.forward(input).gelu());
    }
}
