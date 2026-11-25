import type { Shape, Stride } from './tensor_data.js';

export interface NativeStorage {
    readonly __nativeStorage: unique symbol;
}

export type MapNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
) => void;

export type ZipNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    a: NativeStorage, aShape: Shape, aStride: Stride,
    b: NativeStorage, bShape: Shape, bStride: Stride,
) => void;

export type ReduceNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    dim: number,
) => void;

export type ConvNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    weight: NativeStorage, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
) => void;

export type LayerNormNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    gamma: NativeStorage, gammaShape: Shape, gammaStride: Stride,
    beta: NativeStorage, betaShape: Shape, betaStride: Stride,
    mean: NativeStorage, meanShape: Shape, meanStride: Stride,
    invStd: NativeStorage, invStdShape: Shape, invStdStride: Stride,
    epsilon: number,
) => void;

export type LayerNormBackwardNative = (
    gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride,
    gradGamma: NativeStorage, gradGammaShape: Shape, gradGammaStride: Stride,
    gradBeta: NativeStorage, gradBetaShape: Shape, gradBetaStride: Stride,
    gradOutput: NativeStorage, gradOutputShape: Shape, gradOutputStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    gamma: NativeStorage, gammaShape: Shape, gammaStride: Stride,
    mean: NativeStorage, meanShape: Shape, meanStride: Stride,
    invStd: NativeStorage, invStdShape: Shape, invStdStride: Stride,
) => void;

export type SoftmaxBackwardNative = (
    gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride,
    gradOutput: NativeStorage, gradOutputShape: Shape, gradOutputStride: Stride,
    out: NativeStorage, outShape: Shape, outStride: Stride,
    dim: number,
) => void;

export type RopeNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    sin: NativeStorage, sinShape: Shape, sinStride: Stride,
    cos: NativeStorage, cosShape: Shape, cosStride: Stride,
    offset: number,
) => void;

export type AttentionNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    lse: NativeStorage, lseShape: Shape, lseStride: Stride,
    query: NativeStorage, queryShape: Shape, queryStride: Stride,
    key: NativeStorage, keyShape: Shape, keyStride: Stride,
    value: NativeStorage, valueShape: Shape, valueStride: Stride,
    scale: number, causal: boolean,
) => void;

export type AttentionBackwardNative = (
    gradQuery: NativeStorage, gradQueryShape: Shape, gradQueryStride: Stride,
    gradKey: NativeStorage, gradKeyShape: Shape, gradKeyStride: Stride,
    gradValue: NativeStorage, gradValueShape: Shape, gradValueStride: Stride,
    gradOut: NativeStorage, gradOutShape: Shape, gradOutStride: Stride,
    query: NativeStorage, queryShape: Shape, queryStride: Stride,
    key: NativeStorage, keyShape: Shape, keyStride: Stride,
    value: NativeStorage, valueShape: Shape, valueStride: Stride,
    out: NativeStorage, outShape: Shape, outStride: Stride,
    lse: NativeStorage, lseShape: Shape, lseStride: Stride,
    scale: number, causal: boolean,
) => void;

// Names match the strings addon.cc puts on `exports`.
export type NativeAddon = {
    mapNeg: MapNative;
    mapInv: MapNative;
    mapSigmoid: MapNative;
    mapRelu: MapNative;
    mapGelu: MapNative;
    mapSilu: MapNative;
    mapLog: MapNative;
    mapExp: MapNative;
    mapSqrt: MapNative;
    mapId: MapNative;
    zipAdd: ZipNative;
    zipSub: ZipNative;
    zipMul: ZipNative;
    zipDiv: ZipNative;
    zipLt: ZipNative;
    zipEq: ZipNative;
    zipIsClose: ZipNative;
    zipGeluBackward: ZipNative;
    zipSiluBackward: ZipNative;
    reduceMax: ReduceNative;
    reduceSum: ReduceNative;
    reduceProd: ReduceNative;
    matmul: ZipNative;
    conv1d: ConvNative;
    conv2d: ConvNative;
    alloc: (numel: number, fill?: number) => NativeStorage;
    empty: (numel: number) => NativeStorage;
    retain: (storage: NativeStorage) => void;
    release: (storage: NativeStorage) => void;
    fill: (storage: NativeStorage, value: number) => void;
    copyToHost: (storage: NativeStorage) => Float32Array;
    copyFromHost: (storage: NativeStorage, values: Float32Array) => void;
    synchronize: () => void;
    backendInfo?: () => { kind: 'cuda' | 'cpu'; available?: boolean };
    sgd: (
        param: NativeStorage, shape: Shape, stride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        lr: number,
    ) => void;
    sgdMomentum: (
        param: NativeStorage, shape: Shape, stride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        velocity: NativeStorage, velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number,
    ) => void;
    adamw: (
        param: NativeStorage, shape: Shape, stride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        m: NativeStorage, mShape: Shape, mStride: Stride,
        v: NativeStorage, vShape: Shape, vStride: Stride,
        lr: number, beta1: number, beta2: number, eps: number, weightDecay: number, t: number,
    ) => void;
    gather: (
        out: NativeStorage, outShape: Shape, outStride: Stride,
        indicies: NativeStorage, indiciesShape: Shape, indiciesStride: Stride,
        values: NativeStorage, valuesShape: Shape, valuesStride: Stride,
        dim: number
    ) => void;
    scatter: (
        out: NativeStorage, outShape: Shape, outStride: Stride,
        indicies: NativeStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        dim: number,
    ) => void;
    embedding: (
        out: NativeStorage, outShape: Shape, outStride: Stride,
        indicies: NativeStorage, indiciesShape: Shape, indiciesStride: Stride,
        table: NativeStorage, tableShape: Shape, tableStride: Stride,
        features: number,
    ) => void;
    embeddingBackward: (
        table: NativeStorage, tableShape: Shape, tableStride: Stride,
        indicies: NativeStorage, indiciesShape: Shape, indiciesStride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        features: number,
    ) => void;
    layernorm: LayerNormNative;
    layernormBackward: LayerNormBackwardNative;
    softmax: ReduceNative;
    softmaxBackward: SoftmaxBackwardNative;
    logSoftmax: ReduceNative;
    logSoftmaxBackward: SoftmaxBackwardNative;
    rope: RopeNative;
    ropeBackward: RopeNative;
    attention: AttentionNative;
    attentionBackward: AttentionBackwardNative;
};

// Set by initBackend(). Keeping this named export preserves the low-level
// escape hatch used by native-specific tests without importing Node built-ins
// into browser bundles at module evaluation time.
export let native: NativeAddon | null = null;

function nativePackageSuffix(platform: string, arch: string): string | null {
    switch (`${platform}-${arch}`) {
        case 'win32-x64': return 'win32-x64';
        case 'linux-x64': return 'linux-x64-gnu';
        case 'linux-arm64': return 'linux-arm64-gnu';
        case 'darwin-arm64': return 'darwin-arm64';
        case 'darwin-x64': return 'darwin-x64';
        default: return null;
    }
}

function warnNative(error: unknown): void {
    if (process.env?.TSGRAD_DEBUG_BACKEND) console.warn('Unable to load native tensor backend', error);
}

function cudaReady(addon: NativeAddon): boolean {
    try {
        const info = addon.backendInfo?.();
        return info?.kind === 'cuda' && info.available === true;
    } catch (error) {
        warnNative(error);
        return false;
    }
}

export async function loadNativeAddon(): Promise<NativeAddon | null> {
    if (native) return native;
    if (typeof process === 'undefined' || !process.versions?.node) return null;

    const moduleName = 'node:module';
    const fsName = 'node:fs';
    const urlName = 'node:url';
    const [{ createRequire }, { existsSync }, { fileURLToPath }] = await Promise.all([
        import(/* @vite-ignore */ moduleName) as Promise<typeof import('node:module')>,
        import(/* @vite-ignore */ fsName) as Promise<typeof import('node:fs')>,
        import(/* @vite-ignore */ urlName) as Promise<typeof import('node:url')>,
    ]);
    const require = createRequire(import.meta.url);
    const addonRelativePath = ['..', 'backend', 'build', 'Release', 'tsgrad_native.node'].join('/');
    const addonPath = fileURLToPath(new URL(addonRelativePath, import.meta.url));
    if (existsSync(addonPath)) {
        try {
            native = require(addonPath) as NativeAddon;
            return native;
        } catch (error) {
            warnNative(error);
        }
    }

    const suffix = nativePackageSuffix(process.platform, process.arch);
    if (!suffix) return null;

    const cudaName = `tengear-${suffix}-cuda12`;
    try {
        const cuda = require(cudaName) as NativeAddon;
        if (cudaReady(cuda)) {
            native = cuda;
            return native;
        }
    } catch (error) {
        warnNative(error);
    }

    try {
        native = require(`tengear-${suffix}`) as NativeAddon;
        return native;
    } catch (error) {
        warnNative(error);
        return null;
    }
}
