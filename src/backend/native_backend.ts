import { native, type NativeAddon, type NativeStorage } from '../native.js';
import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from './device_backend.js';

function nativeStorage(storage: BackendStorage): NativeStorage {
    if (storage instanceof Float32Array) throw new Error('Native backend requires NativeStorage');
    return storage as NativeStorage;
}

export class NativeBackend implements DeviceBackend {
    readonly kind: 'cuda' | 'native-cpu';
    readonly supportsSynchronousRead = true;

    constructor(readonly addon: NativeAddon) {
        this.kind = addon.backendInfo?.().kind === 'cuda' ? 'cuda' : 'native-cpu';
    }

    isStorage(value: unknown): boolean {
        return typeof value === 'object' && value !== null && !(value instanceof Float32Array);
    }
    alloc(numel: number, fill = 0): NativeStorage { return this.addon.alloc(numel, fill); }
    empty(numel: number): NativeStorage { return this.addon.empty(numel); }
    retain(storage: BackendStorage): void { this.addon.retain(nativeStorage(storage)); }
    release(storage: BackendStorage): void { this.addon.release(nativeStorage(storage)); }
    fill(storage: BackendStorage, value: number): void { this.addon.fill(nativeStorage(storage), value); }
    copyFromHost(storage: BackendStorage, values: Float32Array): void { this.addon.copyFromHost(nativeStorage(storage), values); }
    copyToHost(storage: BackendStorage): Float32Array { return this.addon.copyToHost(nativeStorage(storage)); }
    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> { return this.copyToHost(storage); }
    async synchronize(): Promise<void> { this.addon.synchronize(); }

    map(op: UnaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride): void {
        const fn = {
            neg: this.addon.mapNeg, inv: this.addon.mapInv, sigmoid: this.addon.mapSigmoid,
            relu: this.addon.mapRelu, gelu: this.addon.mapGelu, silu: this.addon.mapSilu, log: this.addon.mapLog,
            exp: this.addon.mapExp, sqrt: this.addon.mapSqrt, id: this.addon.mapId,
        }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride);
    }

    zip(op: BinaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        const fn = {
            add: this.addon.zipAdd, sub: this.addon.zipSub, mul: this.addon.zipMul,
            div: this.addon.zipDiv, lt: this.addon.zipLt, eq: this.addon.zipEq,
            isClose: this.addon.zipIsClose, geluBackward: this.addon.zipGeluBackward,
            siluBackward: this.addon.zipSiluBackward,
        }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(a), aShape, aStride, nativeStorage(b), bShape, bStride);
    }

    reduce(op: ReduceOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const fn = { max: this.addon.reduceMax, sum: this.addon.reduceSum, prod: this.addon.reduceProd }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, dim);
    }

    matmul(out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        this.addon.matmul(nativeStorage(out), outShape, outStride, nativeStorage(a), aShape, aStride, nativeStorage(b), bShape, bStride);
    }
    conv1d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        this.addon.conv1d(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, nativeStorage(weight), weightShape, weightStride, reverse);
    }
    conv2d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        this.addon.conv2d(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, nativeStorage(weight), weightShape, weightStride, reverse);
    }
    sgd(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride, lr: number): void {
        this.addon.sgd(nativeStorage(param), shape, stride, nativeStorage(grad), gradShape, gradStride, lr);
    }
    sgdMomentum(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        velocity: BackendStorage, velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        this.addon.sgdMomentum(
            nativeStorage(param), shape, stride,
            nativeStorage(grad), gradShape, gradStride,
            nativeStorage(velocity), velocityShape, velocityStride,
            lr, beta,
        );
    }
    adamw(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        m: BackendStorage, mShape: Shape, mStride: Stride,
        v: BackendStorage, vShape: Shape, vStride: Stride,
        lr: number, beta1: number, beta2: number, eps: number, weightDecay: number, t: number): void {
        this.addon.adamw(
            nativeStorage(param), shape, stride,
            nativeStorage(grad), gradShape, gradStride,
            nativeStorage(m), mShape, mStride,
            nativeStorage(v), vShape, vStride,
            lr, beta1, beta2, eps, weightDecay, t,
        );
    }

    gather(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        values: BackendStorage, valuesShape: Shape, valuesStride: Stride,
        dim: number): void {
        this.addon.gather(
            nativeStorage(out), outShape, outStride,
            nativeStorage(indicies), indiciesShape, indiciesStride,
            nativeStorage(values), valuesShape, valuesStride,
            dim
        );
    }

    scatter(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        dim: number): void {
        this.addon.scatter(
            nativeStorage(out), outShape, outStride,
            nativeStorage(indicies), indiciesShape, indiciesStride,
            nativeStorage(grad), gradShape, gradStride,
            dim,
        );
    }

    embedding(out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        table: BackendStorage, tableShape: Shape, tableStride: Stride,
        features: number): void {
        this.addon.embedding(
            nativeStorage(out), outShape, outStride,
            nativeStorage(indicies), indiciesShape, indiciesStride,
            nativeStorage(table), tableShape, tableStride,
            features,
        );
    }

    embeddingBackward(table: BackendStorage, tableShape: Shape, tableStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        features: number): void {
        this.addon.embeddingBackward(
            nativeStorage(table), tableShape, tableStride,
            nativeStorage(indicies), indiciesShape, indiciesStride,
            nativeStorage(grad), gradShape, gradStride,
            features,
        );
    }

    layernorm(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        gamma: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        beta: BackendStorage, betaShape: Shape, betaStride: Stride,
        mean: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStd: BackendStorage, invStdShape: Shape, invStdStride: Stride,
        epsilon: number): void {
        this.addon.layernorm(
            nativeStorage(out), outShape, outStride,
            nativeStorage(input), inShape, inStride,
            nativeStorage(gamma), gammaShape, gammaStride,
            nativeStorage(beta), betaShape, betaStride,
            nativeStorage(mean), meanShape, meanStride,
            nativeStorage(invStd), invStdShape, invStdStride,
            epsilon,
        );
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
        this.addon.layernormBackward(
            nativeStorage(gradInput), gradInputShape, gradInputStride,
            nativeStorage(gradGamma), gradGammaShape, gradGammaStride,
            nativeStorage(gradBeta), gradBetaShape, gradBetaStride,
            nativeStorage(gradOutput), gradOutputShape, gradOutputStride,
            nativeStorage(input), inShape, inStride,
            nativeStorage(gamma), gammaShape, gammaStride,
            nativeStorage(mean), meanShape, meanStride,
            nativeStorage(invStd), invStdShape, invStdStride,
        );
    }

    softmax(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        this.addon.softmax(
            nativeStorage(out), outShape, outStride,
            nativeStorage(input), inShape, inStride, dim,
        );
    }

    softmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride, dim: number,
    ): void {
        this.addon.softmaxBackward(
            nativeStorage(gradInput), gradInputShape, gradInputStride,
            nativeStorage(gradOutput), gradOutputShape, gradOutputStride,
            nativeStorage(out), outShape, outStride, dim,
        );
    }

    logSoftmax(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        this.addon.logSoftmax(
            nativeStorage(out), outShape, outStride,
            nativeStorage(input), inShape, inStride, dim,
        );
    }

    logSoftmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride, dim: number,
    ): void {
        this.addon.logSoftmaxBackward(
            nativeStorage(gradInput), gradInputShape, gradInputStride,
            nativeStorage(gradOutput), gradOutputShape, gradOutputStride,
            nativeStorage(out), outShape, outStride, dim,
        );
    }
}
