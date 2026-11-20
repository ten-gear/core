import { Tensor } from "./tensor.js";
import { Shape, shapeBroadcast, TensorData, writeStorage } from "./tensor_data.js";
import { Context } from "./autodiff.js";
import * as operators from "./operators.js";
// swap out if wanted, tensor_ops has non-multithreaded, the fastTensor stuff is multithreaded
//import { tensorMap, tensorZip, tensorReduce } from "./tensor_ops.js";
//import {/*fastTensorMap as tensorMap, fastTensorZip as tensorZip, fastTensorReduce as tensorReduce, fastMatMul as matMul,*/ fastConv1d as tensorConv1d, fastConv2d as tensorConv2d} from "./fast_ops.js";
import {tensorMap, tensorZip, tensorReduce, matMul, tensorConv1d, tensorConv2d, tensorGather, tensorScatter, tensorEmbedding, tensorEmbeddingBackward, tensorLayerNorm, tensorLayerNormBackward, tensorSoftmax, tensorSoftmaxBackward, tensorLogSoftmax, tensorLogSoftmaxBackward } from "./native_ops.js"
import { randomFloat } from "./helperfunctions.js";

export function neg(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.neg)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function inv(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.inv)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function sigmoid(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.sigmoid)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function relu(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.relu)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function gelu(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.gelu)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function silu(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.silu)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function log(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.log)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function exp(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.exp)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function sqrt(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.sqrt)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function id(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.id)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function max(input: TensorData, dim: number): TensorData{
    const outShape = [...input._shape];
    outShape[dim] = 1;
    const out = TensorData.empty(outShape)
    tensorReduce(operators.max, -Infinity)(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, dim);
    return out;
}

export function add(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.add)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function mul(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.mul)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function lt(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.lt)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function eq(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.eq)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function sub(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.sub)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function div(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.div)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function isclose(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.isClose)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function geluBackward(grad: TensorData, input: TensorData): TensorData{
    const outShape = shapeBroadcast(grad._shape, input._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.geluBackward)(
        out._storage, out._shape, out._stride,
        grad._storage, grad._shape, grad._stride,
        input._storage, input._shape, input._stride,
    );
    return out;
}

export function siluBackward(grad: TensorData, input: TensorData): TensorData{
    const outShape = shapeBroadcast(grad._shape, input._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.siluBackward)(
        out._storage, out._shape, out._stride,
        grad._storage, grad._shape, grad._stride,
        input._storage, input._shape, input._stride,
    );
    return out;
}

function shapesCanBeMul(aShape: Shape, bShape: Shape): boolean{
    if (aShape.length != bShape.length || aShape.length < 2) return false;

    for (let i = 0; i < aShape.length - 2; i++)
        if (aShape[i] != bShape[i]) return false;

    return aShape[aShape.length - 1] == bShape[bShape.length - 2];
}

export function matmul(a: TensorData, b: TensorData): TensorData{
    if (!shapesCanBeMul(a._shape, b._shape))
        throw "Incompatible shapes for matrix mul, a: " + a._shape + ", b: " + b._shape;

    // [..., M, K] @ [..., K, N] -> [..., M, N]. Leading dimensions are the batch.
    const outShape = [...a._shape];
    outShape[outShape.length - 1] = b._shape[b._shape.length - 1]!;
    let out = TensorData.empty(outShape);
    matMul(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function conv1d(input: TensorData, weight: TensorData, reverse: boolean = false, outWidth?: number): TensorData{
    /*
     """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output
    */
    
    const batch = input._shape[0], inChannels = input._shape[1], w = input._shape[2];
    const outChannel = weight._shape[0], inChannels2 = weight._shape[1], kw = weight._shape[2];

    if (inChannels != inChannels2) throw "in channels must be same between both input shape and weight shape";

    let out = TensorData.empty([batch, outChannel, outWidth ? outWidth : w]);
    tensorConv1d(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, weight._storage, weight._shape, weight._stride, reverse);
    return out;
}

//                                                                                      new h then w
export function conv2d(input: TensorData, weight: TensorData, reverse: boolean = false, ...outDims : number[]): TensorData{
    const batch = input._shape[0], inChannels = input._shape[1], h = outDims[0] ?? input._shape[2], w = outDims[1] ?? input._shape[3];
    const outChannel = weight._shape[0], inChannels2 = weight._shape[1], kh = weight._shape[2], kw = weight._shape[3];

    if (inChannels != inChannels2) throw "in channels must be same between both input shape and weight shape";

    let out = TensorData.empty([batch, outChannel, h, w]);
    tensorConv2d(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, weight._storage, weight._shape, weight._stride, reverse);
    return out;
}

// for reducing
export function sum(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.empty(outShape);
    tensorReduce(operators.add, 0)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function prod(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.empty(outShape);
    tensorReduce(operators.mul, 1)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function view(a: TensorData, shape: Shape): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => val === a._stride[idx]) === false)
        throw "view received non-contiguous Tensor, received: " + a;

    if (operators.prod(shape) !== operators.prod(a._shape))
        throw "view received different sizes, received: " + a + " and shape: " + shape;

    return TensorData.share(a._storage, shape);
}

export function tocontiguous(a: TensorData): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => val === a._stride[idx]) === true)
        return a; // already contiguous
    return id(a);
}

export function gather(indicies: TensorData, values: TensorData, dim: number): TensorData{
    const outShape = [...indicies.shape];
    const out = TensorData.empty(outShape);
    tensorGather(out._storage, out.shape, out._stride, indicies._storage, indicies._shape, indicies._stride, values._storage, values._shape, values._stride, dim);
    return out;
}

// Adds grad into out. The destination coordinate is the grad coordinate with dim replaced by indicies.
export function scatter(out: TensorData, indicies: TensorData, grad: TensorData, dim: number): TensorData{
    tensorScatter(out._storage, out._shape, out._stride, indicies._storage, indicies._shape, indicies._stride, grad._storage, grad._shape, grad._stride, dim);
    return out;
}

// out[*indicies.shape, table.shape[1]] = table[indicies, :]. table is [vocabulary, features].
export function embedding(indicies: TensorData, table: TensorData): TensorData{
    if (table._shape.length !== 2)
        throw "embedding received table with dimensions not equal to 2, received: " + table._shape.length;
    const features = table._shape[1]!;
    const out = TensorData.empty([...indicies.shape, features]);
    tensorEmbedding(out._storage, out._shape, out._stride, indicies._storage, indicies._shape, indicies._stride, table._storage, table._shape, table._stride, features);
    return out;
}

// table[indicies, :] += grad. grad is [*indicies.shape, features]. Adds into the values already in table.
export function embeddingBackward(table: TensorData, indicies: TensorData, grad: TensorData): TensorData{
    if (table._shape.length !== 2)
        throw "embeddingBackward received table with dimensions not equal to 2, received: " + table._shape.length;
    const features = table._shape[1]!;
    tensorEmbeddingBackward(table._storage, table._shape, table._stride, indicies._storage, indicies._shape, indicies._stride, grad._storage, grad._shape, grad._stride, features);
    return table;
}

export type LayerNormForwardResult = {
    output: TensorData;
    mean: TensorData;
    invStd: TensorData;
};

// Normalize the final input dimension. Statistics are returned for the fused backward pass.
export function layernorm(input: TensorData, gamma: TensorData, beta: TensorData, epsilon: number = 1e-5): LayerNormForwardResult {
    if (input._shape.length < 1) throw new Error('layernorm expects an input with rank at least 1');
    const features = input._shape[input._shape.length - 1]!;
    if (features <= 0) throw new Error('layernorm requires a non-empty normalized dimension');
    if (gamma._shape.length !== 1 || gamma._shape[0] !== features ||
        beta._shape.length !== 1 || beta._shape[0] !== features)
        throw new Error('layernorm gamma and beta must be rank-1 tensors matching the final input dimension');
    if (!Number.isFinite(epsilon) || epsilon < 0)
        throw new Error('layernorm expects a finite non-negative epsilon');

    const rows = input.size / features;
    const output = TensorData.empty(input.shape);
    const mean = TensorData.empty([rows]);
    const invStd = TensorData.empty([rows]);
    tensorLayerNorm(
        output._storage, output._shape, output._stride,
        input._storage, input._shape, input._stride,
        gamma._storage, gamma._shape, gamma._stride,
        beta._storage, beta._shape, beta._stride,
        mean._storage, mean._shape, mean._stride,
        invStd._storage, invStd._shape, invStd._stride,
        epsilon,
    );
    return { output, mean, invStd };
}

export function layernormBackward(gradOutput: TensorData, input: TensorData, gamma: TensorData, mean: TensorData, invStd: TensorData): [TensorData, TensorData, TensorData] {
    const gradInput = TensorData.empty(input.shape);
    const gradGamma = TensorData.empty(gamma.shape);
    const gradBeta = TensorData.empty(gamma.shape);
    tensorLayerNormBackward(
        gradInput._storage, gradInput._shape, gradInput._stride,
        gradGamma._storage, gradGamma._shape, gradGamma._stride,
        gradBeta._storage, gradBeta._shape, gradBeta._stride,
        gradOutput._storage, gradOutput._shape, gradOutput._stride,
        input._storage, input._shape, input._stride,
        gamma._storage, gamma._shape, gamma._stride,
        mean._storage, mean._shape, mean._stride,
        invStd._storage, invStd._shape, invStd._stride,
    );
    return [gradInput, gradGamma, gradBeta];
}

function normalizeDim(dim: number, rank: number, operation: string): number {
    const normalized = dim < 0 ? dim + rank : dim;
    if (normalized < 0 || normalized >= rank)
        throw new Error(`${operation} received dim outside the input rank`);
    return normalized;
}

export function softmaxData(input: TensorData, dim: number): TensorData {
    const normalizedDim = normalizeDim(dim, input.dims, 'softmax');
    const out = TensorData.empty(input.shape);
    tensorSoftmax(
        out._storage, out._shape, out._stride,
        input._storage, input._shape, input._stride,
        normalizedDim,
    );
    return out;
}

export function softmaxBackward(gradOutput: TensorData, out: TensorData, dim: number): TensorData {
    const normalizedDim = normalizeDim(dim, out.dims, 'softmaxBackward');
    const gradInput = TensorData.empty(out.shape);
    tensorSoftmaxBackward(
        gradInput._storage, gradInput._shape, gradInput._stride,
        gradOutput._storage, gradOutput._shape, gradOutput._stride,
        out._storage, out._shape, out._stride,
        normalizedDim,
    );
    return gradInput;
}

export function logsoftmaxData(input: TensorData, dim: number): TensorData {
    const normalizedDim = normalizeDim(dim, input.dims, 'logsoftmax');
    const out = TensorData.empty(input.shape);
    tensorLogSoftmax(
        out._storage, out._shape, out._stride,
        input._storage, input._shape, input._stride,
        normalizedDim,
    );
    return out;
}

export function logsoftmaxBackward(gradOutput: TensorData, out: TensorData, dim: number): TensorData {
    const normalizedDim = normalizeDim(dim, out.dims, 'logsoftmaxBackward');
    const gradInput = TensorData.empty(out.shape);
    tensorLogSoftmaxBackward(
        gradInput._storage, gradInput._shape, gradInput._stride,
        gradOutput._storage, gradOutput._shape, gradOutput._stride,
        out._storage, out._shape, out._stride,
        normalizedDim,
    );
    return gradInput;
}

export abstract class TensorFunction{
    static forward(ctx: Context, ...t: Tensor[]): Tensor{
        throw new Error("Subclass must create forward");
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        throw new Error("Subclass must create backward");
    }
}

export function unbroadcast(a: Tensor, original: Shape): Tensor{
    let res = a;
    while (res.dims() > original.length){ // first make shape length match (same num of dimensions)
        res = res.sum(0); // shrink from the left, e.g., (shapes) og = [2, 1] res = [3, 2, 1] should cut off to res = [1, 2, 1]
        res = res.view(...res.shape().slice(1)); // then res = [2, 1]
    }

    for (let i = 0; i < original.length; i++){ // second reduce across each dimension as needed
        if (original[i] === 1 && res.shape()[i] > 1)
            res = res.sum(i);
    }

    return res;
}

export function tile2d(input: Tensor, kernel : [number, number]): [Tensor, number, number]{
    // reshapes a Tensor of shape [batch, channel, height, width] to [batch, channel, newHeight, newWidth, kernelHeight * kernelWidth] returning the new tensor as well as the new height and width
    // assumes that the input size is divisible by the kernel size
    // also assumes stride to be equal to kernel size, so no overlapping stuff

    const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
    const kh = kernel[0], kw = kernel[1];
    if (height % kh !== 0 || width % kw !== 0) throw "Expected input size in tile2d to be divisible by kernel size";

    const newHeight = height / kh;
    const newWidth = width / kw;

    return [input.view(batch, channel, newHeight, kh, newWidth, kw).permute(0, 1, 2, 4, 3, 5).tocontiguous().view(batch, channel, newHeight, newWidth, kh * kw), newHeight, newWidth];
}

export function avgpool2d(input: Tensor, kernel: [number, number]): Tensor{
    const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
    const kh = kernel[0], kw = kernel[1];
    const [tiled, newHeight, newWidth] = tile2d(input, kernel);

    //[batch, channel, newHeight, newWidth, kernelHeight * kernelWidth]
    // .sum(4) means sum along the tiled dimension --^
    return tiled.sum(4).div(kh * kw).view(batch, channel, newHeight, newWidth);
}

export function maxpool2d(input: Tensor, kernel: [number, number]): Tensor{
    const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
    const kh = kernel[0], kw = kernel[1];
    const [tiled, newHeight, newWidth] = tile2d(input, kernel);

    return tiled.max(4).view(batch, channel, newHeight, newWidth);
}

export function rparam(shape: Shape, lowerBound: number = -1, upperBound: number = 1): Tensor{
    // Random tensor where each cell is -1 to 1 with shape as the shape
    const data = TensorData.fill(shape);
    const values = new Float32Array(data.size);
    for (let i = 0; i < values.length; i++)
        values[i] = randomFloat(lowerBound, upperBound);
    writeStorage(data._storage, values);
    const tensor = new Tensor(data);
    return tensor;
}

export function dropout(input: Tensor, rate: number, ignore: boolean = false): Tensor{
    if (rate < 0.0 || rate > 1.0) throw "rate must be in [0, 1)";

    if (ignore || rate == 0.0) return input;

    const threshold = TensorData.fill(input.shape(), 1 - rate);
    const random = rparam(input.shape(), 0, 1);

    const mask = new Tensor(lt(random.data, threshold));
    return input.mul(mask).div(1.0 - rate);
}

export function argmax(input: Tensor, dim: number): Tensor{
    return new Tensor(max(input.data, dim)).eq(input);
}

export function softmax(input: Tensor, dim: number): Tensor{
    return Tensor.apply(Softmax(dim), input);
}

export function logsoftmax(input: Tensor, dim: number): Tensor{
    return Tensor.apply(LogSoftmax(dim), input);
}

export class Neg extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        return new Tensor(neg(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.neg()];
    }
}

// I don't like it, and it's kinda messy using Tensor operations inside the Operation classes but it works...
export class Inv extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(inv(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.neg().div(a.mul(a))];
    }
}

export class Add extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(add(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput, a.shape()), unbroadcast(gradOutput, b.shape())];
    }
}

export class Sub extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(sub(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput, a.shape()), unbroadcast(gradOutput.neg(), b.shape())];
    }
}

export class Div extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(div(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        const gradA = gradOutput.div(b);
        const gradB = gradOutput.neg().mul(a).div(b.mul(b));
        return [unbroadcast(gradA, a.shape()), unbroadcast(gradB, b.shape())];
    }
}

export class Mul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(mul(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput.mul(b), a.shape()), unbroadcast(gradOutput.mul(a), b.shape())];
    }
}

export class Sigmoid extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(sigmoid(a.data));
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [res] = ctx.savedValues;
        return [gradOutput.mul(res).mul(res.neg().add(1))];
    }
}

export class ReLU extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(relu(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        const [a] = ctx.savedValues;
        return [gradOutput.mul(a.gt(0))];
    }
}

export class GELU extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        const res = new Tensor(gelu(a.data));
        ctx.saveForBackward(a);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        const [a] = ctx.savedValues;
        return [new Tensor(geluBackward(gradOutput.data, a.data))];
    }
}

export class SiLU extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        const res = new Tensor(silu(a.data));
        ctx.saveForBackward(a);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        const [a] = ctx.savedValues;
        return [new Tensor(siluBackward(gradOutput.data, a.data))];
    }
}

export class Log extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(log(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.div(a)];
    }
}

export class Exp extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(exp(a.data));
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [res] = ctx.savedValues;
        return [gradOutput.mul(res)];
    }
}

export class Sqrt extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        const res = new Tensor(sqrt(a.data));
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        const [res] = ctx.savedValues;
        return [gradOutput.div(res.mul(2))];
    }
}

export class LT extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(lt(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class EQ extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(eq(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class IsClose extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(isclose(a.data, b.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

function transposeMatrixDims(a: Tensor): Tensor{
    const rank = a.shape().length;
    const order = new Array(rank);
    for (let i = 0; i < rank; i++)
        order[i] = i;

    order[rank - 1] = rank - 2;
    order[rank - 2] = rank - 1;
    return a.permute(...order);
}

export class MatMul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(matmul(a.data, b.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [gradOutput.matmul(transposeMatrixDims(b)), transposeMatrixDims(a).matmul(gradOutput)];
    }
}

export class Conv1d extends TensorFunction{
    static forward(ctx: Context, input: Tensor, weight: Tensor): Tensor{
        ctx.saveForBackward(input, weight);
        return new Tensor(conv1d(input.data, weight.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [input, weight] = ctx.savedValues;

        // derivative of input, then weight
        return [new Tensor(conv1d(gradOutput.data, weight.permute(1, 0, 2).data, true)), new Tensor(conv1d(input.permute(1, 0, 2).data, gradOutput.permute(1, 0, 2).data, false, weight.shape()[2])).permute(1, 0, 2).tocontiguous()];
    }
}

export class Conv2d extends TensorFunction{
    static forward(ctx: Context, input: Tensor, weight: Tensor): Tensor{
        ctx.saveForBackward(input, weight);
        return new Tensor(conv2d(input.data, weight.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [input, weight] = ctx.savedValues;

        // derivative of input, then weight
        return [new Tensor(conv2d(gradOutput.data, weight.permute(1, 0, 2, 3).data, true)), new Tensor(conv2d(input.permute(1, 0, 2, 3).data, gradOutput.permute(1, 0, 2, 3).data, false, weight.shape()[2], weight.shape()[3])).permute(1, 0, 2, 3).tocontiguous()];
    }
}

export class Embedding extends TensorFunction{
    static forward(ctx: Context, indicies: Tensor, table: Tensor): Tensor{
        ctx.saveForBackward(indicies, table);
        return new Tensor(embedding(indicies.data, table.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        const [indicies, table] = ctx.savedValues;
        const gradTable = new Tensor(TensorData.fill(table.shape(), 0));
        return [new Tensor(TensorData.fill(indicies.shape(), 0)) , new Tensor(embeddingBackward(gradTable.data, indicies.data, gradOutput.data))];
    }
}

export function LayerNorm(epsilon: number = 1e-5): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor, gamma: Tensor, beta: Tensor): Tensor {
            const result = layernorm(input.data, gamma.data, beta.data, epsilon);
            if (ctx.noGrad) {
                result.mean.release();
                result.invStd.release();
            } else {
                ctx.saveForBackward(
                    input, gamma, new Tensor(result.mean), new Tensor(result.invStd)
                );
            }
            return new Tensor(result.output);
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [input, gamma, mean, invStd] = ctx.savedValues as Tensor[];
            const [gradInput, gradGamma, gradBeta] = layernormBackward(gradOutput.data, input.data, gamma.data, mean.data, invStd.data);
            return [new Tensor(gradInput), new Tensor(gradGamma), new Tensor(gradBeta)];
        }
    };
}

export function Softmax(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor): Tensor {
            const result = new Tensor(softmaxData(input.data, dim));
            ctx.saveForBackward(result);
            return result;
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [result] = ctx.savedValues as Tensor[];
            return [new Tensor(softmaxBackward(gradOutput.data, result.data, dim))];
        }
    };
}

export function LogSoftmax(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor): Tensor {
            const result = new Tensor(logsoftmaxData(input.data, dim));
            ctx.saveForBackward(result);
            return result;
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [result] = ctx.savedValues as Tensor[];
            return [new Tensor(logsoftmaxBackward(gradOutput.data, result.data, dim))];
        }
    };
}

export function Gather(dim: number): typeof TensorFunction {
    return class extends TensorFunction{
        static forward(ctx: Context, indicies: Tensor, values: Tensor): Tensor{
            ctx.saveForBackward(indicies, values, dim);
            return new Tensor(gather(indicies.data, values.data, dim));
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
            const [indicies, values, dim] = ctx.savedValues;
            const gradValues = new Tensor(TensorData.fill(values.shape(), 0));
            return [new Tensor(TensorData.fill(indicies.shape(), 0)) , new Tensor(scatter(gradValues.data, indicies.data, gradOutput.data, dim))];
        }
    }
}

export function Max(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor): Tensor {
            ctx.saveForBackward(input);
            return new Tensor(max(input.data, dim));
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [input] = ctx.savedValues;
            return [argmax(input, dim).mul(gradOutput)];
        }
    };
}

export class ToContiguous extends TensorFunction {
    static forward(ctx: Context, a: Tensor): Tensor {
        return new Tensor(tocontiguous(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
        return [gradOutput];
    }
}

export function Sum(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(sum(a.data, dim));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.mul(Tensor.ones(a.shape()))];
        }
    };
}

export function Permute(order: number[]): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            return new Tensor(a.data.permute(...order));
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const inverseOrder = new Array(order.length);

            for (let i = 0; i < order.length; i++)
                inverseOrder[order[i]!] = i;

            return [new Tensor(gradOutput.data.permute(...inverseOrder))];
        }
    };
}

export function View(newShape: Shape): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(view(a.data, newShape));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.tocontiguous().view(...a.shape())];
        }
    };
}