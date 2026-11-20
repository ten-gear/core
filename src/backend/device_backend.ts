export type Shape = number[];
export type Stride = number[];
export type BackendStorage = Float32Array | object;
export type BackendKind = 'cpu' | 'native-cpu' | 'cuda' | 'webgpu';
export type UnaryOp = 'neg' | 'inv' | 'sigmoid' | 'relu' | 'gelu' | 'silu' | 'log' | 'exp' | 'sqrt' | 'id';
export type BinaryOp = 'add' | 'sub' | 'mul' | 'div' | 'lt' | 'eq' | 'isClose' | 'geluBackward' | 'siluBackward';
export type ReduceOp = 'max' | 'sum' | 'prod';

export interface DeviceBackend {
    readonly kind: BackendKind;
    readonly supportsSynchronousRead: boolean;

    isStorage(value: unknown): boolean;
    alloc(numel: number, fill?: number): BackendStorage;
    empty(numel: number): BackendStorage;
    retain(storage: BackendStorage): void;
    release(storage: BackendStorage): void;
    fill(storage: BackendStorage, value: number): void;
    copyFromHost(storage: BackendStorage, values: Float32Array): void;
    copyToHost(storage: BackendStorage): Float32Array;
    copyToHostAsync(storage: BackendStorage): Promise<Float32Array>;
    synchronize(): Promise<void>;

    map(
        op: UnaryOp,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
    ): void;
    zip(
        op: BinaryOp,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride,
    ): void;
    reduce(
        op: ReduceOp,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        dim: number,
    ): void;
    matmul(
        out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride,
    ): void;
    conv1d(
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride,
        reverse: boolean,
    ): void;
    conv2d(
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride,
        reverse: boolean,
    ): void;
    sgd(
        param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        lr: number,
    ): void;
    sgdMomentum(
        param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        velocity: BackendStorage, velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number,
    ): void;
    gather: (
        out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        values: BackendStorage, valuesShape: Shape, valuesStride: Stride,
        dim: number
    ) => void;
    scatter: (
        out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        dim: number,
    ) => void;
    embedding: (
        out: BackendStorage, outShape: Shape, outStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        table: BackendStorage, tableShape: Shape, tableStride: Stride,
        features: number,
    ) => void;
    embeddingBackward: (
        table: BackendStorage, tableShape: Shape, tableStride: Stride,
        indicies: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        features: number,
    ) => void;
    layernorm: (
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        gamma: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        beta: BackendStorage, betaShape: Shape, betaStride: Stride,
        mean: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStd: BackendStorage, invStdShape: Shape, invStdStride: Stride,
        epsilon: number,
    ) => void;
    layernormBackward: (
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradGamma: BackendStorage, gradGammaShape: Shape, gradGammaStride: Stride,
        gradBeta: BackendStorage, gradBetaShape: Shape, gradBetaStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        gamma: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        mean: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStd: BackendStorage, invStdShape: Shape, invStdStride: Stride,
    ) => void;
    softmax(
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        dim: number,
    ): void;
    softmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        dim: number,
    ): void;
    logSoftmax(
        out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        dim: number,
    ): void;
    logSoftmaxBackward(
        gradInput: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradOutput: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        out: BackendStorage, outShape: Shape, outStride: Stride,
        dim: number,
    ): void;
}
