import * as operators from '../operators.js';
import { tensorMap, tensorReduce, tensorZip } from '../tensor_ops.js';
import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from './device_backend.js';

const unary = {
    neg: operators.neg, inv: operators.inv, sigmoid: operators.sigmoid,
    relu: operators.relu, gelu: operators.gelu, silu: operators.silu, log: operators.log,
    exp: operators.exp, sqrt: operators.sqrt, id: operators.id,
} satisfies Record<UnaryOp, (x: number) => number>;

const binary = {
    add: operators.add, sub: operators.sub, mul: operators.mul, div: operators.div,
    lt: operators.lt, eq: operators.eq, isClose: operators.isClose,
    geluBackward: operators.geluBackward, siluBackward: operators.siluBackward,
} satisfies Record<BinaryOp, (a: number, b: number) => number>;

const reductions = {
    max: [operators.max, -Infinity],
    sum: [operators.add, 0],
    prod: [operators.mul, 1],
} satisfies Record<ReduceOp, [(a: number, b: number) => number, number]>;

function host(storage: BackendStorage): Float32Array {
    if (!(storage instanceof Float32Array)) throw new Error('CPU backend requires Float32Array storage');
    return storage;
}

function positionToIndex(ordinal: number, shape: Shape, out: number[]): void {
    for (let i = shape.length - 1; i >= 0; i--) {
        out[i] = ordinal % shape[i]!;
        ordinal = Math.trunc(ordinal / shape[i]!);
    }
}

function indexToPosition(index: number[], stride: Stride): number {
    let position = 0;
    for (let i = 0; i < stride.length; i++) position += index[i]! * stride[i]!;
    return position;
}

function contiguous(shape: Shape, stride: Stride): boolean {
    let expected = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i]! > 1 && stride[i] !== expected) return false;
        expected *= shape[i]!;
    }
    return true;
}

function leadingOffset(index: number[], stride: Stride, count: number): number {
    let position = 0;
    for (let i = 0; i < count; i++) position += index[i]! * stride[i]!;
    return position;
}

function rowOffset(row: number, shape: Shape, stride: Stride): number {
    let position = 0;
    for (let dim = shape.length - 2; dim >= 0; dim--) {
        const index = row % shape[dim]!;
        row = Math.trunc(row / shape[dim]!);
        position += index * stride[dim]!;
    }
    return position;
}

function sliceOffset(row: number, shape: Shape, stride: Stride, dim: number): number {
    let position = 0;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
        if (axis === dim) continue;
        const index = row % shape[axis]!;
        row = Math.trunc(row / shape[axis]!);
        position += index * stride[axis]!;
    }
    return position;
}

function sameShape(a: Shape, b: Shape): boolean {
    return a.length === b.length && a.every((value, index) => value === b[index]);
}

function layernormDimensions(
    inputShape: Shape, gammaShape: Shape, meanShape: Shape, invStdShape: Shape,
): [number, number] {
    if (inputShape.length < 1) throw new Error('layernorm expects an input with rank at least 1');
    if (gammaShape.length !== 1) throw new Error('layernorm expects gamma to have rank 1');
    if (meanShape.length !== 1 || invStdShape.length !== 1)
        throw new Error('layernorm expects mean and invStd to have rank 1');
    const features = inputShape[inputShape.length - 1]!;
    if (features <= 0) throw new Error('layernorm requires a non-empty normalized dimension');
    if (gammaShape[0] !== features) throw new Error('layernorm gamma shape mismatch');
    const rows = inputShape.reduce((size, dim) => size * dim, 1) / features;
    if (meanShape[0] !== rows || invStdShape[0] !== rows)
        throw new Error('layernorm saved-stat shape mismatch');
    return [rows, features];
}

function softmaxDimensions(shape: Shape, dim: number, operation: string): [number, number] {
    if (shape.length < 1) throw new Error(`${operation} expects an input with rank at least 1`);
    if (dim < 0 || dim >= shape.length) throw new Error(`${operation} received dim outside the input rank`);
    const dimSize = shape[dim]!;
    if (dimSize <= 0) throw new Error(`${operation} requires a non-empty normalized dimension`);
    return [shape.reduce((size, value) => size * value, 1) / dimSize, dimSize];
}

export class CpuBackend implements DeviceBackend {
    readonly kind = 'cpu' as const;
    readonly supportsSynchronousRead = true;

    isStorage(value: unknown): boolean { return value instanceof Float32Array; }
    alloc(numel: number, fill = 0): Float32Array {
        const out = new Float32Array(numel);
        if (fill !== 0) out.fill(fill);
        return out;
    }
    empty(numel: number): Float32Array { return new Float32Array(numel); }
    retain(_storage: BackendStorage): void {}
    release(_storage: BackendStorage): void {}
    fill(storage: BackendStorage, value: number): void { host(storage).fill(value); }
    copyFromHost(storage: BackendStorage, values: Float32Array): void {
        const target = host(storage);
        if (target.length !== values.length) throw new Error('copyFromHost length does not match storage');
        target.set(values);
    }
    copyToHost(storage: BackendStorage): Float32Array { return host(storage); }
    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> { return host(storage); }
    async synchronize(): Promise<void> {}

    map(op: UnaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride): void {
        tensorMap(unary[op])(host(out), outShape, outStride, host(input), inShape, inStride);
    }

    zip(op: BinaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        tensorZip(binary[op])(host(out), outShape, outStride, host(a), aShape, aStride, host(b), bShape, bStride);
    }

    reduce(op: ReduceOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const [fn, base] = reductions[op];
        tensorReduce(fn, base)(host(out), outShape, outStride, host(input), inShape, inStride, dim);
    }

    matmul(out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        if (aShape.length !== bShape.length || aShape.length < 2)
            throw new Error('CPU matmul requires matching ranks of at least 2');
        const rank = aShape.length;
        for (let i = 0; i < rank - 2; i++)
            if (aShape[i] !== bShape[i]) throw new Error('CPU matmul batch dimensions must match');
        const m = aShape[rank - 2]!, k = aShape[rank - 1]!, n = bShape[rank - 1]!;
        if (k !== bShape[rank - 2]) throw new Error('CPU matmul inner dimensions must match');
        if (outShape.length !== rank) throw new Error('CPU matmul output rank must match the inputs');

        const output = host(out), av = host(a), bv = host(b);
        const batchRank = rank - 2;
        const batchShape = aShape.slice(0, batchRank);
        let batch = 1;
        for (const dim of batchShape) batch *= dim;
        if (batch === 0 || m === 0 || n === 0) return;

        const batchIndex = new Array(batchRank).fill(0);
        const matrix = m * n;
        for (let ordinal = 0; ordinal < batch * matrix; ordinal++) {
            const within = ordinal % matrix;
            const row = Math.trunc(within / n);
            const col = within % n;
            let aBase = 0, bBase = 0, outBase = 0;
            if (batchRank > 0) {
                positionToIndex(Math.trunc(ordinal / matrix), batchShape, batchIndex);
                aBase = leadingOffset(batchIndex, aStride, batchRank);
                bBase = leadingOffset(batchIndex, bStride, batchRank);
                outBase = leadingOffset(batchIndex, outStride, batchRank);
            }

            let total = 0;
            let aPos = aBase + row * aStride[rank - 2]!;
            let bPos = bBase + col * bStride[rank - 1]!;
            for (let inner = 0; inner < k; inner++) {
                total += av[aPos]! * bv[bPos]!;
                aPos += aStride[rank - 1]!;
                bPos += bStride[rank - 2]!;
            }
            output[outBase + row * outStride[rank - 2]! + col * outStride[rank - 1]!] = total;
        }
    }

    conv1d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const output = host(out), x = host(input), w = host(weight);
        for (let batch = 0; batch < outShape[0]!; batch++) {
            for (let oc = 0; oc < outShape[1]!; oc++) {
                for (let ow = 0; ow < outShape[2]!; ow++) {
                    let total = 0;
                    for (let ic = 0; ic < inShape[1]!; ic++) {
                        for (let kernel = 0; kernel < weightShape[2]!; kernel++) {
                            const at = reverse ? ow - kernel : ow + kernel;
                            if (at >= 0 && at < inShape[2]!)
                                total += x[batch*inStride[0]! + ic*inStride[1]! + at*inStride[2]!]!
                                    * w[oc*weightStride[0]! + ic*weightStride[1]! + kernel*weightStride[2]!]!;
                        }
                    }
                    output[batch*outStride[0]! + oc*outStride[1]! + ow*outStride[2]!] = total;
                }
            }
        }
    }

    conv2d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const output = host(out), x = host(input), w = host(weight);
        for (let batch = 0; batch < outShape[0]!; batch++) {
            for (let oc = 0; oc < outShape[1]!; oc++) {
                for (let oh = 0; oh < outShape[2]!; oh++) {
                    for (let ow = 0; ow < outShape[3]!; ow++) {
                        let total = 0;
                        for (let ic = 0; ic < inShape[1]!; ic++) {
                            for (let kh = 0; kh < weightShape[2]!; kh++) {
                                const ih = reverse ? oh - kh : oh + kh;
                                if (ih < 0 || ih >= inShape[2]!) continue;
                                for (let kw = 0; kw < weightShape[3]!; kw++) {
                                    const iw = reverse ? ow - kw : ow + kw;
                                    if (iw >= 0 && iw < inShape[3]!)
                                        total += x[batch*inStride[0]! + ic*inStride[1]! + ih*inStride[2]! + iw*inStride[3]!]!
                                            * w[oc*weightStride[0]! + ic*weightStride[1]! + kh*weightStride[2]! + kw*weightStride[3]!]!;
                                }
                            }
                        }
                        output[batch*outStride[0]! + oc*outStride[1]! + oh*outStride[2]! + ow*outStride[3]!] = total;
                    }
                }
            }
        }
    }

    sgd(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, _gradShape: Shape, gradStride: Stride, lr: number): void {
        const p = host(param), g = host(grad), index = new Array(shape.length).fill(0);
        const size = shape.reduce((a, b) => a * b, 1);
        for (let i = 0; i < size; i++) {
            positionToIndex(i, shape, index);
            p[indexToPosition(index, stride)]! -= lr * g[indexToPosition(index, gradStride)]!;
        }
    }

    sgdMomentum(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, _gradShape: Shape, gradStride: Stride,
        velocity: BackendStorage, _velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        const p = host(param), g = host(grad), v = host(velocity);
        const index = new Array(shape.length).fill(0);
        const size = shape.reduce((a, b) => a * b, 1);
        for (let i = 0; i < size; i++) {
            positionToIndex(i, shape, index);
            const pp = indexToPosition(index, stride);
            const gp = indexToPosition(index, gradStride);
            const vp = indexToPosition(index, velocityStride);
            const next = beta * v[vp]! + (1 - beta) * g[gp]!;
            v[vp] = next;
            p[pp]! -= lr * next;
        }
    }

    gather(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        values: BackendStorage, valuesShape: Shape, valuesStride: Stride,
        dim: number): void{
            const o = host(out), i = host(indicies), v = host(values);
            if (dim < 0 || dim >= outShape.length){
                throw new Error("gather received dim outside dimensions of out");
            }
        
            if (indiciesShape.length != valuesShape.length)
                throw new Error("indicies, and values have differing ranks inside gather");

            for (let x = 0; x < indiciesShape.length; x++){
                if (x != dim){
                    if (indiciesShape[x] < 0 || indiciesShape[x] > valuesShape[x]){
                        throw new Error("gather received incompatible shapes for indicies and values");
                    }
                }
            }

            const outIndex = new Array(outShape.length);
            const size = outShape.reduce((a, b) => a * b, 1);
            for (let ordinal = 0; ordinal < size; ordinal++){
                positionToIndex(ordinal, outShape, outIndex);
                const outPos = indexToPosition(outIndex, outStride);
                outIndex[dim] = i[indexToPosition(outIndex, indiciesStride)];
                o[outPos] = v[indexToPosition(outIndex, valuesStride)];
            }
        }

    scatter(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        dim: number): void {
        const destination = host(out), index = host(indicies), gradient = host(grad);
        if (dim < 0 || dim >= gradShape.length)
            throw new Error('scatter received dim outside dimensions of grad');
        if (indiciesShape.length !== outShape.length || indiciesShape.length !== gradShape.length)
            throw new Error('indicies, out, and grad have differing ranks inside scatter');
        for (let i = 0; i < indiciesShape.length; i++) {
            if (i !== dim && (indiciesShape[i]! < 0 || indiciesShape[i]! > outShape[i]!))
                throw new Error('scatter received incompatible shapes for indicies and out');
        }

        const gradIndex = new Array<number>(gradShape.length);
        const size = gradShape.reduce((total, dimSize) => total * dimSize, 1);
        for (let ordinal = 0; ordinal < size; ordinal++) {
            positionToIndex(ordinal, gradShape, gradIndex);
            const gradPos = indexToPosition(gradIndex, gradStride);
            gradIndex[dim] = Math.trunc(index[indexToPosition(gradIndex, indiciesStride)]!);
            destination[indexToPosition(gradIndex, outStride)]! += gradient[gradPos]!;
        }
    }

    embedding(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        table: BackendStorage, tableShape: Shape, tableStride: Stride,
        features: number): void {
        const destination = host(out), index = host(indicies), rows = host(table);
        if (tableShape.length !== 2)
            throw new Error('embedding received table with dimensions not equal to 2, received: ' + tableShape.length);
        if (indiciesShape.length + 1 !== outShape.length)
            throw new Error('embedding received indicies/out with bad dimensions, received indicies.rank: ' + indiciesShape.length + ', out.rank: ' + outShape.length);
        for (let i = 0; i < indiciesShape.length; i++) {
            if (indiciesShape[i] !== outShape[i])
                throw new Error('embedding received indicies/out with bad shapes');
        }
        if (tableShape[1] !== features || outShape[outShape.length - 1] !== features)
            throw new Error('embedding received out with bad shape, last element of out.shape must equal features');

        const size = outShape.reduce((total, dimSize) => total * dimSize, 1);
        if (size === 0) return;
        if (contiguous(outShape, outStride) && contiguous(indiciesShape, indiciesStride) && contiguous(tableShape, tableStride)) {
            for (let ordinal = 0; ordinal < size; ordinal++) {
                const row = Math.trunc(ordinal / features);
                const col = ordinal % features;
                destination[ordinal] = rows[Math.trunc(index[row]!) * features + col];
            }
            return;
        }

        const outIndex = new Array<number>(outShape.length);
        for (let ordinal = 0; ordinal < size; ordinal++) {
            positionToIndex(ordinal, outShape, outIndex);
            const token = Math.trunc(index[leadingOffset(outIndex, indiciesStride, indiciesShape.length)]!);
            const feature = outIndex[outShape.length - 1]!;
            destination[indexToPosition(outIndex, outStride)] = rows[token * tableStride[0]! + feature * tableStride[1]!];
        }
    }

    embeddingBackward(table: BackendStorage, tableShape: Shape, tableStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        features: number): void {
        const rows = host(table), index = host(indicies), gradient = host(grad);
        if (tableShape.length !== 2)
            throw new Error('embeddingBackward received table with dimensions not equal to 2, received: ' + tableShape.length);
        if (indiciesShape.length + 1 !== gradShape.length)
            throw new Error('embeddingBackward received indicies/grad with bad dimensions, received indicies.rank: ' + indiciesShape.length + ', grad.rank: ' + gradShape.length);
        for (let i = 0; i < indiciesShape.length; i++) {
            if (indiciesShape[i] !== gradShape[i])
                throw new Error('embeddingBackward received indicies/grad with bad shapes');
        }
        if (tableShape[1] !== features || gradShape[gradShape.length - 1] !== features)
            throw new Error('embeddingBackward received grad with bad shape, last element of grad.shape must equal features');

        const size = gradShape.reduce((total, dimSize) => total * dimSize, 1);
        if (size === 0) return;
        if (contiguous(gradShape, gradStride) && contiguous(indiciesShape, indiciesStride) && contiguous(tableShape, tableStride)) {
            for (let ordinal = 0; ordinal < size; ordinal++) {
                const row = Math.trunc(ordinal / features);
                const col = ordinal % features;
                rows[Math.trunc(index[row]!) * features + col]! += gradient[ordinal]!;
            }
            return;
        }

        const gradIndex = new Array<number>(gradShape.length);
        for (let ordinal = 0; ordinal < size; ordinal++) {
            positionToIndex(ordinal, gradShape, gradIndex);
            const gradPos = indexToPosition(gradIndex, gradStride);
            const token = Math.trunc(index[leadingOffset(gradIndex, indiciesStride, indiciesShape.length)]!);
            const feature = gradIndex[gradShape.length - 1]!;
            rows[token * tableStride[0]! + feature * tableStride[1]!]! += gradient[gradPos]!;
        }
    }

    layernorm(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        gamma: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        beta: BackendStorage, betaShape: Shape, betaStride: Stride,
        mean: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStd: BackendStorage, invStdShape: Shape, invStdStride: Stride,
        epsilon: number): void {
        if (!sameShape(outShape, inShape)) throw new Error('layernorm input/output shape mismatch');
        const [rows, features] = layernormDimensions(inShape, gammaShape, meanShape, invStdShape);
        if (betaShape.length !== 1 || betaShape[0] !== features)
            throw new Error('layernorm beta shape mismatch');
        if (!Number.isFinite(epsilon) || epsilon < 0)
            throw new Error('layernorm expects a finite non-negative epsilon');

        const output = host(out), x = host(input), scale = host(gamma), bias = host(beta);
        const means = host(mean), inverseStd = host(invStd);
        const inFeatureStride = inStride[inShape.length - 1]!;
        const outFeatureStride = outStride[outShape.length - 1]!;
        for (let row = 0; row < rows; row++) {
            const inBase = rowOffset(row, inShape, inStride);
            const outBase = rowOffset(row, outShape, outStride);
            let rowMean = 0;
            let m2 = 0;
            for (let col = 0; col < features; col++) {
                const value = x[inBase + col * inFeatureStride]!;
                const count = col + 1;
                const delta = value - rowMean;
                rowMean += delta / count;
                m2 += delta * (value - rowMean);
            }
            const rowInvStd = 1 / Math.sqrt(m2 / features + epsilon);
            for (let col = 0; col < features; col++) {
                const xHat = (x[inBase + col * inFeatureStride]! - rowMean) * rowInvStd;
                output[outBase + col * outFeatureStride] =
                    scale[col * gammaStride[0]!]! * xHat + bias[col * betaStride[0]!]!;
            }
            means[row * meanStride[0]!] = rowMean;
            inverseStd[row * invStdStride[0]!] = rowInvStd;
        }
    }

    layernormBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradGamma: BackendStorage, gradGammaShape: Shape, gradGammaStride: Stride,
        gradBeta: BackendStorage, gradBetaShape: Shape, gradBetaStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        gamma: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        mean: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStd: BackendStorage, invStdShape: Shape, invStdStride: Stride,
    ): void {
        if (!sameShape(gradInputShape, inShape) || !sameShape(gradOutputShape, inShape))
            throw new Error('layernormBackward input-gradient shape mismatch');
        const [rows, features] = layernormDimensions(inShape, gammaShape, meanShape, invStdShape);
        if (gradGammaShape.length !== 1 || gradGammaShape[0] !== features ||
            gradBetaShape.length !== 1 || gradBetaShape[0] !== features)
            throw new Error('layernormBackward parameter-gradient shape mismatch');

        const dx = host(gradInput), dGamma = host(gradGamma), dBeta = host(gradBeta);
        const dy = host(gradOutput), x = host(input), scale = host(gamma);
        const means = host(mean), inverseStd = host(invStd);
        const dxFeatureStride = gradInputStride[gradInputShape.length - 1]!;
        const dyFeatureStride = gradOutputStride[gradOutputShape.length - 1]!;
        const inFeatureStride = inStride[inShape.length - 1]!;

        for (let row = 0; row < rows; row++) {
            const dxBase = rowOffset(row, gradInputShape, gradInputStride);
            const dyBase = rowOffset(row, gradOutputShape, gradOutputStride);
            const inBase = rowOffset(row, inShape, inStride);
            const rowMean = means[row * meanStride[0]!]!;
            const rowInvStd = inverseStd[row * invStdStride[0]!]!;
            let sumQ = 0;
            let sumQXHat = 0;
            for (let col = 0; col < features; col++) {
                const xHat = (x[inBase + col * inFeatureStride]! - rowMean) * rowInvStd;
                const q = dy[dyBase + col * dyFeatureStride]! * scale[col * gammaStride[0]!]!;
                sumQ += q;
                sumQXHat += q * xHat;
            }
            const meanQ = sumQ / features;
            const meanQXHat = sumQXHat / features;
            for (let col = 0; col < features; col++) {
                const xHat = (x[inBase + col * inFeatureStride]! - rowMean) * rowInvStd;
                const q = dy[dyBase + col * dyFeatureStride]! * scale[col * gammaStride[0]!]!;
                dx[dxBase + col * dxFeatureStride] = rowInvStd * (q - meanQ - xHat * meanQXHat);
            }
        }

        for (let col = 0; col < features; col++) {
            let gammaSum = 0;
            let betaSum = 0;
            for (let row = 0; row < rows; row++) {
                const dyBase = rowOffset(row, gradOutputShape, gradOutputStride);
                const inBase = rowOffset(row, inShape, inStride);
                const gradient = dy[dyBase + col * dyFeatureStride]!;
                const xHat = (x[inBase + col * inFeatureStride]!
                    - means[row * meanStride[0]!]!) * inverseStd[row * invStdStride[0]!]!;
                gammaSum += gradient * xHat;
                betaSum += gradient;
            }
            dGamma[col * gradGammaStride[0]!] = gammaSum;
            dBeta[col * gradBetaStride[0]!] = betaSum;
        }
    }

    private softmaxForward(logarithmic: boolean,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        dim: number, operation: string): void {
        if (!sameShape(outShape, inShape)) throw new Error(`${operation} input/output shape mismatch`);
        const [rows, dimSize] = softmaxDimensions(inShape, dim, operation);
        const output = host(out), x = host(input);
        for (let row = 0; row < rows; row++) {
            const outBase = sliceOffset(row, outShape, outStride, dim);
            const inputBase = sliceOffset(row, inShape, inStride, dim);
            let maximum = -Infinity;
            for (let col = 0; col < dimSize; col++)
                maximum = Math.max(maximum, x[inputBase + col * inStride[dim]!]!);
            let sumExp = 0;
            for (let col = 0; col < dimSize; col++)
                sumExp += Math.exp(x[inputBase + col * inStride[dim]!]! - maximum);

            if (logarithmic) {
                const logSumExp = maximum + Math.log(sumExp);
                for (let col = 0; col < dimSize; col++)
                    output[outBase + col * outStride[dim]!] =
                        x[inputBase + col * inStride[dim]!]! - logSumExp;
            } else {
                for (let col = 0; col < dimSize; col++)
                    output[outBase + col * outStride[dim]!] =
                        Math.exp(x[inputBase + col * inStride[dim]!]! - maximum) / sumExp;
            }
        }
    }

    softmax(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        this.softmaxForward(false, out, outShape, outStride, input, inShape, inStride, dim, 'softmax');
    }

    logSoftmax(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        this.softmaxForward(true, out, outShape, outStride, input, inShape, inStride, dim, 'logSoftmax');
    }

    private softmaxBackwardImpl(logarithmic: boolean,
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        dim: number, operation: string): void {
        if (!sameShape(gradInputShape, outShape) || !sameShape(gradOutputShape, outShape))
            throw new Error(`${operation} gradient shape mismatch`);
        const [rows, dimSize] = softmaxDimensions(outShape, dim, operation);
        const dx = host(gradInput), dy = host(gradOutput), y = host(out);
        for (let row = 0; row < rows; row++) {
            const dxBase = sliceOffset(row, gradInputShape, gradInputStride, dim);
            const dyBase = sliceOffset(row, gradOutputShape, gradOutputStride, dim);
            const outBase = sliceOffset(row, outShape, outStride, dim);
            let reduced = 0;
            for (let col = 0; col < dimSize; col++) {
                const gradient = dy[dyBase + col * gradOutputStride[dim]!]!;
                reduced += logarithmic
                    ? gradient
                    : gradient * y[outBase + col * outStride[dim]!]!;
            }
            for (let col = 0; col < dimSize; col++) {
                const gradient = dy[dyBase + col * gradOutputStride[dim]!]!;
                const value = y[outBase + col * outStride[dim]!]!;
                dx[dxBase + col * gradInputStride[dim]!] = logarithmic
                    ? gradient - Math.exp(value) * reduced
                    : value * (gradient - reduced);
            }
        }
    }

    softmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride, dim: number,
    ): void {
        this.softmaxBackwardImpl(
            false, gradInput, gradInputShape, gradInputStride,
            gradOutput, gradOutputShape, gradOutputStride,
            out, outShape, outStride, dim, 'softmaxBackward',
        );
    }

    logSoftmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride, dim: number,
    ): void {
        this.softmaxBackwardImpl(
            true, gradInput, gradInputShape, gradInputStride,
            gradOutput, gradOutputShape, gradOutputStride,
            out, outShape, outStride, dim, 'logSoftmaxBackward',
        );
    }
}
