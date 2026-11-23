import { randomFloat } from "./helperfunctions.js";
import { Parameter } from "./module.js";
import { Layer } from "./model.js";
import { Tensor } from "./tensor.js";
import { flashAttention, logsoftmax, maxpool2d, rope } from "./tensor_functions.js";
import { Shape, TensorData, writeStorage } from "./tensor_data.js";

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

function zeroParameter(shape: Shape): Tensor {
    const tensor = new Tensor(TensorData.fill(shape, 0));
    tensor._requiresGrad(true);
    return tensor;
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

    constructor(inSize: number, outSize: number) {
        super();
        this.inSize = inSize;
        this.outSize = outSize;
        this.weights = this.addParameter("weights", heParameter(inSize, [inSize, outSize]));
        this.bias = this.addParameter("bias", zeroParameter([outSize]));
    }

    forward(input: Tensor): Tensor {
        return input
            .matmul(this.weights.value as Tensor)
            .add(this.bias.value as Tensor);
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
