import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from '../device_backend.js';
import {
    alignedMapShader, alignedZipShader, CONV1D_SHADER, CONV2D_SHADER, FILL_SHADER,
    EMBEDDING_BACKWARD_CONTIGUOUS_SHADER, EMBEDDING_BACKWARD_SHADER, EMBEDDING_CONTIGUOUS_SHADER, EMBEDDING_SHADER, GATHER_SHADER, MATMUL_SHADER, MAX_DIMS, reduceShader, SCATTER_SHADER, SGD_MOMENTUM_SHADER, SGD_SHADER,
    LAYERNORM_BACKWARD_INPUT_SHADER, LAYERNORM_BACKWARD_PARAMS_SHADER, LAYERNORM_SHADER,
    stridedMapShader, stridedZipShader, WORKGROUP_SIZE,
} from './webgpu_kernels.js';
import { WebGPUStorage } from './webgpu_storage.js';

// WebGPU flag values are fixed by the specification. TypeScript's current DOM
// declarations expose the interfaces but not these runtime constant objects.
const BUFFER_USAGE = {
    MAP_READ: 0x0001,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
} as const;
const MAP_MODE_READ = 0x0001;

function gpuStorage(storage: BackendStorage): WebGPUStorage {
    if (!(storage instanceof WebGPUStorage) || storage.released)
        throw new Error('WebGPU backend requires live WebGPUStorage');
    return storage;
}

function product(shape: Shape): number {
    return shape.reduce((a, b) => a * b, 1);
}

function positionToIndex(ordinal: number, shape: Shape, out: number[]): void {
    for (let i = shape.length - 1; i >= 0; i--) {
        out[i] = ordinal % shape[i]!;
        ordinal = Math.trunc(ordinal / shape[i]!);
    }
}

function leadingOffset(index: number[], stride: Stride, count: number): number {
    let position = 0;
    for (let i = 0; i < count; i++) position += index[i]! * stride[i]!;
    return position;
}

function same(a: Shape, b: Shape): boolean {
    return a.length === b.length && a.every((v, i) => v === b[i]);
}

function contiguous(shape: Shape, stride: Stride): boolean {
    let expected = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i]! > 1 && stride[i] !== expected) return false;
        expected *= shape[i]!;
    }
    return true;
}

function floatBits(value: number): number {
    const buffer = new ArrayBuffer(4);
    new Float32Array(buffer)[0] = value;
    return new Uint32Array(buffer)[0]!;
}

export class WebGpuBackend implements DeviceBackend {
    readonly kind = 'webgpu' as const;
    readonly supportsSynchronousRead = false;
    private readonly pipelines = new Map<string, GPUComputePipeline>();
    private readonly free = new Map<number, GPUBuffer[]>();
    private pendingDestroy: GPUBuffer[] = [];

    constructor(readonly device: GPUDevice) {}

    isStorage(value: unknown): boolean { return value instanceof WebGPUStorage && !value.released; }

    private takeBuffer(byteSize: number): GPUBuffer {
        const list = this.free.get(byteSize);
        const reused = list?.pop();
        if (reused) return reused;
        return this.device.createBuffer({
            size: byteSize,
            usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST,
        });
    }

    alloc(numel: number, fill = 0): WebGPUStorage {
        const byteSize = Math.max(4, numel * Float32Array.BYTES_PER_ELEMENT);
        const storage = new WebGPUStorage(this.takeBuffer(byteSize), numel, byteSize);
        this.fill(storage, fill);
        return storage;
    }

    empty(numel: number): WebGPUStorage {
        const byteSize = Math.max(4, numel * Float32Array.BYTES_PER_ELEMENT);
        return new WebGPUStorage(this.takeBuffer(byteSize), numel, byteSize);
    }

    retain(storage: BackendStorage): void { gpuStorage(storage).owners++; }

    release(storage: BackendStorage): void {
        const value = gpuStorage(storage);
        value.owners--;
        if (value.owners < 0) throw new Error('WebGPU storage released too many times');
        if (value.owners === 0) {
            value.released = true;
            // Every command uses this device's single ordered queue. Reusing
            // the same GPUBuffer is safe: any later write is submitted after
            // all earlier reads that referenced it. Never destroy pooled
            // data buffers until backend shutdown.
            this.free.set(value.byteSize, [...(this.free.get(value.byteSize) ?? []), value.buffer]);
        }
    }

    private pipeline(key: string, shader: string): GPUComputePipeline {
        let pipeline = this.pipelines.get(key);
        if (!pipeline) {
            pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: this.device.createShaderModule({ code: shader }), entryPoint: 'main' },
            });
            this.pipelines.set(key, pipeline);
        }
        return pipeline;
    }

    private params(values: number[], uniform = false): GPUBuffer {
        const data = Uint32Array.from(values);
        const byteSize = Math.max(uniform ? 16 : 4, Math.ceil(data.byteLength / 4) * 4);
        const buffer = this.device.createBuffer({
            size: byteSize,
            usage: (uniform ? BUFFER_USAGE.UNIFORM : BUFFER_USAGE.STORAGE) | BUFFER_USAGE.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    private pad(values: number[]): number[] {
        if (values.length > MAX_DIMS) throw new Error(`WebGPU tensors support at most ${MAX_DIMS} dimensions`);
        if (values.some(v => v < 0 || !Number.isSafeInteger(v))) throw new Error('WebGPU shapes and strides must be non-negative integers');
        return [...values, ...new Array(MAX_DIMS - values.length).fill(0)];
    }

    private submit(
        key: string,
        shader: string,
        entries: GPUBindGroupEntry[],
        workgroups: [number, number?, number?],
        temporary: GPUBuffer[],
    ): void {
        const pipeline = this.pipeline(key, shader);
        const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        this.deferDestroy(...temporary);
    }

    private deferDestroy(...buffers: GPUBuffer[]): void {
        this.pendingDestroy.push(...buffers);
        if (this.pendingDestroy.length >= 128) {
            void this.flushPendingDestroy();
        }
    }

    private async flushPendingDestroy(): Promise<void> {
        // Capture only buffers used by work submitted before this fence.
        // Later submissions append to a fresh array and are cleaned by a
        // later fence, so concurrent readback cannot destroy their params.
        const batch = this.pendingDestroy.splice(0);
        await this.device.queue.onSubmittedWorkDone();
        batch.forEach(buffer => buffer.destroy());
    }

    fill(storage: BackendStorage, value: number): void {
        const target = gpuStorage(storage);
        if (target.numel === 0) return;
        if (value === 0) {
            const encoder = this.device.createCommandEncoder();
            encoder.clearBuffer(target.buffer);
            this.device.queue.submit([encoder.finish()]);
            return;
        }
        const params = this.params([target.numel, floatBits(value)], true);
        this.submit('fill', FILL_SHADER, [
            { binding: 0, resource: { buffer: target.buffer } },
            { binding: 1, resource: { buffer: params } },
        ], [Math.ceil(target.numel / WORKGROUP_SIZE)], [params]);
    }

    copyFromHost(storage: BackendStorage, values: Float32Array): void {
        const target = gpuStorage(storage);
        if (values.length !== target.numel) throw new Error('copyFromHost length does not match storage');
        if (values.length > 0) this.device.queue.writeBuffer(target.buffer, 0, values);
    }

    copyToHost(_storage: BackendStorage): Float32Array {
        throw new Error('WebGPU readback is asynchronous; use readStorageAsync() or Tensor.toFloat32()');
    }

    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> {
        const source = gpuStorage(storage);
        if (source.numel === 0) return new Float32Array();
        const staging = this.device.createBuffer({
            size: source.byteSize,
            usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
        });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source.buffer, 0, staging, 0, source.byteSize);
        this.device.queue.submit([encoder.finish()]);
        const cleanup = this.flushPendingDestroy();
        await staging.mapAsync(MAP_MODE_READ);
        const result = new Float32Array(staging.getMappedRange()).slice(0, source.numel);
        staging.unmap();
        staging.destroy();
        await cleanup;
        return result;
    }

    async synchronize(): Promise<void> {
        await this.flushPendingDestroy();
    }

    private copyStorage(source: WebGPUStorage, target: WebGPUStorage): void {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source.buffer, 0, target.buffer, 0, Math.min(source.byteSize, target.byteSize));
        this.device.queue.submit([encoder.finish()]);
    }

    map(op: UnaryOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage);
        if (out.buffer === input.buffer) {
            const temp = this.empty(out.numel);
            this.map(op, temp, outShape, outStride, input, inShape, inStride);
            this.copyStorage(temp, out);
            this.release(temp);
            return;
        }
        const size = product(outShape);
        if (size === 0) return;
        const aligned = same(outShape, inShape) && contiguous(outShape, outStride) && contiguous(inShape, inStride);
        const params = aligned
            ? this.params([size], true)
            : this.params([size, outShape.length, inShape.length, 0,
                ...this.pad(outShape), ...this.pad(outStride), ...this.pad(inShape), ...this.pad(inStride)]);
        this.submit(`map:${op}:${aligned}`, aligned ? alignedMapShader(op) : stridedMapShader(op), [
            { binding: 0, resource: { buffer: input.buffer } },
            { binding: 1, resource: { buffer: out.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    zip(op: BinaryOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        aStorage: BackendStorage, aShape: Shape, aStride: Stride,
        bStorage: BackendStorage, bShape: Shape, bStride: Stride): void {
        const out = gpuStorage(outStorage), a = gpuStorage(aStorage), b = gpuStorage(bStorage);
        if (out.buffer === a.buffer || out.buffer === b.buffer) {
            const temp = this.empty(out.numel);
            this.zip(op, temp, outShape, outStride, a, aShape, aStride, b, bShape, bStride);
            this.copyStorage(temp, out);
            this.release(temp);
            return;
        }
        const size = product(outShape);
        if (size === 0) return;
        const aligned = same(outShape, aShape) && same(outShape, bShape)
            && contiguous(outShape, outStride) && contiguous(aShape, aStride) && contiguous(bShape, bStride);
        const params = aligned
            ? this.params([size], true)
            : this.params([size, outShape.length, aShape.length, bShape.length,
                ...this.pad(outShape), ...this.pad(outStride),
                ...this.pad(aShape), ...this.pad(aStride),
                ...this.pad(bShape), ...this.pad(bStride)]);
        this.submit(`zip:${op}:${aligned}`, aligned ? alignedZipShader(op) : stridedZipShader(op), [
            { binding: 0, resource: { buffer: a.buffer } },
            { binding: 1, resource: { buffer: b.buffer } },
            { binding: 2, resource: { buffer: out.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    reduce(op: ReduceOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage);
        const outSize = product(outShape);
        if (outSize === 0) return;
        const params = this.params([outSize, inShape.length, dim, inShape[dim]!,
            ...this.pad(outShape), ...this.pad(outStride), ...this.pad(inShape), ...this.pad(inStride)]);
        this.submit(`reduce:${op}`, reduceShader(op), [
            { binding: 0, resource: { buffer: input.buffer } },
            { binding: 1, resource: { buffer: out.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [outSize], [params]);
    }

    matmul(outStorage: BackendStorage, _outShape: Shape, outStride: Stride,
        aStorage: BackendStorage, aShape: Shape, aStride: Stride,
        bStorage: BackendStorage, bShape: Shape, bStride: Stride): void {
        if (aShape.length !== bShape.length || aShape.length < 2)
            throw new Error('WebGPU matmul requires matching ranks of at least 2');
        const rank = aShape.length;
        for (let i = 0; i < rank - 2; i++)
            if (aShape[i] !== bShape[i]) throw new Error('WebGPU matmul batch dimensions must match');
        const m = aShape[rank - 2]!, k = aShape[rank - 1]!, n = bShape[rank - 1]!;
        if (k !== bShape[rank - 2]) throw new Error('WebGPU matmul inner dimensions must match');

        const batchRank = rank - 2;
        const batchShape = aShape.slice(0, batchRank);
        let batch = 1;
        for (const dim of batchShape) batch *= dim;
        if (batch === 0 || m === 0 || n === 0) return;

        const out = gpuStorage(outStorage), a = gpuStorage(aStorage), b = gpuStorage(bStorage);
        const batchIndex = new Array(batchRank).fill(0);
        for (let ordinal = 0; ordinal < batch; ordinal++) {
            let aBase = 0, bBase = 0, outBase = 0;
            if (batchRank > 0) {
                positionToIndex(ordinal, batchShape, batchIndex);
                aBase = leadingOffset(batchIndex, aStride, batchRank);
                bBase = leadingOffset(batchIndex, bStride, batchRank);
                outBase = leadingOffset(batchIndex, outStride, batchRank);
            }
            const params = this.params([
                m, n, k,
                outStride[rank - 2]!, outStride[rank - 1]!,
                aStride[rank - 2]!, aStride[rank - 1]!,
                bStride[rank - 2]!, bStride[rank - 1]!,
                aBase, bBase, outBase,
            ]);
            this.submit('matmul:tiled16', MATMUL_SHADER, [
                { binding: 0, resource: { buffer: a.buffer } }, { binding: 1, resource: { buffer: b.buffer } },
                { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
            ], [Math.ceil(n / 16), Math.ceil(m / 16)], [params]);
        }
    }

    conv1d(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        weightStorage: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage), weight = gpuStorage(weightStorage);
        const values = [outShape[0]!, outShape[1]!, outShape[2]!, inShape[1]!, inShape[2]!, weightShape[2]!, Number(reverse),
            ...outStride, ...inStride, ...weightStride];
        const params = this.params(values);
        this.submit('conv1d', CONV1D_SHADER, [
            { binding: 0, resource: { buffer: input.buffer } }, { binding: 1, resource: { buffer: weight.buffer } },
            { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(product(outShape) / WORKGROUP_SIZE)], [params]);
    }

    conv2d(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        weightStorage: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage), weight = gpuStorage(weightStorage);
        const values = [outShape[0]!, outShape[1]!, outShape[2]!, outShape[3]!, inShape[1]!, inShape[2]!, inShape[3]!,
            weightShape[2]!, weightShape[3]!, Number(reverse), ...outStride, ...inStride, ...weightStride];
        const params = this.params(values);
        this.submit('conv2d', CONV2D_SHADER, [
            { binding: 0, resource: { buffer: input.buffer } }, { binding: 1, resource: { buffer: weight.buffer } },
            { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(product(outShape) / WORKGROUP_SIZE)], [params]);
    }

    sgd(paramStorage: BackendStorage, shape: Shape, stride: Stride,
        gradStorage: BackendStorage, _gradShape: Shape, gradStride: Stride, lr: number): void {
        if (!contiguous(shape, stride) || !contiguous(shape, gradStride)) throw new Error('WebGPU SGD requires contiguous tensors');
        const param = gpuStorage(paramStorage), grad = gpuStorage(gradStorage), size = product(shape);
        const params = this.params([size, floatBits(lr)]);
        this.submit('sgd', SGD_SHADER, [
            { binding: 0, resource: { buffer: param.buffer } }, { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    sgdMomentum(paramStorage: BackendStorage, shape: Shape, stride: Stride,
        gradStorage: BackendStorage, _gradShape: Shape, gradStride: Stride,
        velocityStorage: BackendStorage, _velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        if (!contiguous(shape, stride) || !contiguous(shape, gradStride) || !contiguous(shape, velocityStride))
            throw new Error('WebGPU SGD momentum requires contiguous tensors');
        const param = gpuStorage(paramStorage), grad = gpuStorage(gradStorage), velocity = gpuStorage(velocityStorage);
        const size = product(shape), params = this.params([size, floatBits(lr), floatBits(beta)]);
        this.submit('sgdMomentum', SGD_MOMENTUM_SHADER, [
            { binding: 0, resource: { buffer: param.buffer } }, { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: velocity.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    gather(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        indiciesStorage: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        valuesStorage: BackendStorage, valuesShape: Shape, valuesStride: Stride,
        dim: number): void {
        const out = gpuStorage(outStorage), indicies = gpuStorage(indiciesStorage), values = gpuStorage(valuesStorage);
        if (out.buffer === indicies.buffer || out.buffer === values.buffer) {
            const temp = this.empty(out.numel);
            this.gather(temp, outShape, outStride, indicies, indiciesShape, indiciesStride, values, valuesShape, valuesStride, dim);
            this.copyStorage(temp, out);
            this.release(temp);
            return;
        }
        if (dim < 0 || dim >= outShape.length)
            throw new Error('gather received dim outside dimensions of out');
        if (indiciesShape.length !== valuesShape.length)
            throw new Error('indicies, and values have differing ranks inside gather');
        for (let i = 0; i < indiciesShape.length; i++) {
            if (i !== dim && (indiciesShape[i]! < 0 || indiciesShape[i]! > valuesShape[i]!))
                throw new Error('gather received incompatible shapes for indicies and values');
        }

        const size = product(outShape);
        if (size === 0) return;
        const params = this.params([
            size, outShape.length, indiciesShape.length, dim,
            ...this.pad(outShape), ...this.pad(outStride),
            ...this.pad(indiciesStride), ...this.pad(valuesStride),
        ]);
        this.submit('gather', GATHER_SHADER, [
            { binding: 0, resource: { buffer: indicies.buffer } },
            { binding: 1, resource: { buffer: values.buffer } },
            { binding: 2, resource: { buffer: out.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    scatter(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        indiciesStorage: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        gradStorage: BackendStorage, gradShape: Shape, gradStride: Stride,
        dim: number): void {
        const out = gpuStorage(outStorage);
        let indicies = gpuStorage(indiciesStorage);
        let grad = gpuStorage(gradStorage);
        const copies: WebGPUStorage[] = [];
        if (out.buffer === indicies.buffer) {
            const copy = this.empty(indicies.numel);
            this.copyStorage(indicies, copy);
            indicies = copy;
            copies.push(copy);
        }
        if (out.buffer === grad.buffer) {
            const copy = this.empty(grad.numel);
            this.copyStorage(grad, copy);
            grad = copy;
            copies.push(copy);
        }
        if (dim < 0 || dim >= gradShape.length)
            throw new Error('scatter received dim outside dimensions of grad');
        if (indiciesShape.length !== outShape.length || indiciesShape.length !== gradShape.length)
            throw new Error('indicies, out, and grad have differing ranks inside scatter');
        for (let i = 0; i < indiciesShape.length; i++) {
            if (i !== dim && (indiciesShape[i]! < 0 || indiciesShape[i]! > outShape[i]!))
                throw new Error('scatter received incompatible shapes for indicies and out');
        }

        const size = product(gradShape);
        if (size === 0) {
            copies.forEach(copy => this.release(copy));
            return;
        }
        const params = this.params([
            size, gradShape.length, indiciesShape.length, dim,
            ...this.pad(gradShape), ...this.pad(gradStride),
            ...this.pad(indiciesStride), ...this.pad(outStride),
        ]);
        this.submit('scatter', SCATTER_SHADER, [
            { binding: 0, resource: { buffer: indicies.buffer } },
            { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: out.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
        copies.forEach(copy => this.release(copy));
    }

    embedding(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        indiciesStorage: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        tableStorage: BackendStorage, tableShape: Shape, tableStride: Stride,
        features: number): void {
        const out = gpuStorage(outStorage), indicies = gpuStorage(indiciesStorage), table = gpuStorage(tableStorage);
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

        const size = product(outShape);
        if (size === 0) return;
        const aligned = contiguous(outShape, outStride) && contiguous(indiciesShape, indiciesStride) && contiguous(tableShape, tableStride);
        if (aligned) {
            const params = this.params([size, features], true);
            this.submit('embedding:contiguous', EMBEDDING_CONTIGUOUS_SHADER, [
                { binding: 0, resource: { buffer: indicies.buffer } },
                { binding: 1, resource: { buffer: table.buffer } },
                { binding: 2, resource: { buffer: out.buffer } },
                { binding: 3, resource: { buffer: params } },
            ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
            return;
        }
        const params = this.params([
            size, outShape.length, indiciesShape.length, 0,
            ...this.pad(outShape), ...this.pad(outStride),
            ...this.pad(indiciesStride), ...this.pad(tableStride),
        ]);
        this.submit('embedding', EMBEDDING_SHADER, [
            { binding: 0, resource: { buffer: indicies.buffer } },
            { binding: 1, resource: { buffer: table.buffer } },
            { binding: 2, resource: { buffer: out.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    embeddingBackward(tableStorage: BackendStorage, tableShape: Shape, tableStride: Stride,
        indiciesStorage: BackendStorage, indiciesShape: Shape, indiciesStride: Stride,
        gradStorage: BackendStorage, gradShape: Shape, gradStride: Stride,
        features: number): void {
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

        const table = gpuStorage(tableStorage);
        let indicies = gpuStorage(indiciesStorage);
        let grad = gpuStorage(gradStorage);
        const copies: WebGPUStorage[] = [];
        if (table.buffer === indicies.buffer) {
            const copy = this.empty(indicies.numel);
            this.copyStorage(indicies, copy);
            indicies = copy;
            copies.push(copy);
        }
        if (table.buffer === grad.buffer) {
            const copy = this.empty(grad.numel);
            this.copyStorage(grad, copy);
            grad = copy;
            copies.push(copy);
        }

        const size = product(gradShape);
        if (size === 0) {
            copies.forEach(copy => this.release(copy));
            return;
        }
        const aligned = contiguous(gradShape, gradStride) && contiguous(indiciesShape, indiciesStride) && contiguous(tableShape, tableStride);
        if (aligned) {
            const params = this.params([size, features], true);
            this.submit('embeddingBackward:contiguous', EMBEDDING_BACKWARD_CONTIGUOUS_SHADER, [
                { binding: 0, resource: { buffer: indicies.buffer } },
                { binding: 1, resource: { buffer: grad.buffer } },
                { binding: 2, resource: { buffer: table.buffer } },
                { binding: 3, resource: { buffer: params } },
            ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
            copies.forEach(copy => this.release(copy));
            return;
        }
        const params = this.params([
            size, gradShape.length, indiciesShape.length, 0,
            ...this.pad(gradShape), ...this.pad(gradStride),
            ...this.pad(indiciesStride), ...this.pad(tableStride),
        ]);
        this.submit('embeddingBackward', EMBEDDING_BACKWARD_SHADER, [
            { binding: 0, resource: { buffer: indicies.buffer } },
            { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: table.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
        copies.forEach(copy => this.release(copy));
    }

    layernorm(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        gammaStorage: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        betaStorage: BackendStorage, betaShape: Shape, betaStride: Stride,
        meanStorage: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStdStorage: BackendStorage, invStdShape: Shape, invStdStride: Stride,
        epsilon: number): void {
        if (!same(outShape, inShape)) throw new Error('layernorm input/output shape mismatch');
        if (inShape.length < 1) throw new Error('layernorm expects an input with rank at least 1');
        if (inShape.length > MAX_DIMS) throw new Error(`WebGPU tensors support at most ${MAX_DIMS} dimensions`);
        const features = inShape[inShape.length - 1]!;
        if (features <= 0) throw new Error('layernorm requires a non-empty normalized dimension');
        const rows = product(inShape) / features;
        if (gammaShape.length !== 1 || gammaShape[0] !== features ||
            betaShape.length !== 1 || betaShape[0] !== features)
            throw new Error('layernorm gamma/beta shape mismatch');
        if (meanShape.length !== 1 || meanShape[0] !== rows ||
            invStdShape.length !== 1 || invStdShape[0] !== rows)
            throw new Error('layernorm saved-stat shape mismatch');
        if (!Number.isFinite(epsilon) || epsilon < 0)
            throw new Error('layernorm expects a finite non-negative epsilon');
        if (rows === 0) return;

        const output = gpuStorage(outStorage), input = gpuStorage(inputStorage);
        const gamma = gpuStorage(gammaStorage), beta = gpuStorage(betaStorage);
        const mean = gpuStorage(meanStorage), invStd = gpuStorage(invStdStorage);
        const params = this.params([
            rows, features, inShape.length, floatBits(epsilon),
            ...this.pad(outStride), ...this.pad(inShape), ...this.pad(inStride),
            gammaStride[0]!, betaStride[0]!, meanStride[0]!, invStdStride[0]!,
        ]);
        this.submit('layernorm', LAYERNORM_SHADER, [
            { binding: 0, resource: { buffer: input.buffer } },
            { binding: 1, resource: { buffer: gamma.buffer } },
            { binding: 2, resource: { buffer: beta.buffer } },
            { binding: 3, resource: { buffer: output.buffer } },
            { binding: 4, resource: { buffer: mean.buffer } },
            { binding: 5, resource: { buffer: invStd.buffer } },
            { binding: 6, resource: { buffer: params } },
        ], [rows], [params]);
    }

    layernormBackward(
        gradInputStorage: BackendStorage, gradInputShape: Shape, gradInputStride: Stride,
        gradGammaStorage: BackendStorage, gradGammaShape: Shape, gradGammaStride: Stride,
        gradBetaStorage: BackendStorage, gradBetaShape: Shape, gradBetaStride: Stride,
        gradOutputStorage: BackendStorage, gradOutputShape: Shape, gradOutputStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        gammaStorage: BackendStorage, gammaShape: Shape, gammaStride: Stride,
        meanStorage: BackendStorage, meanShape: Shape, meanStride: Stride,
        invStdStorage: BackendStorage, invStdShape: Shape, invStdStride: Stride,
    ): void {
        if (!same(gradInputShape, inShape) || !same(gradOutputShape, inShape))
            throw new Error('layernormBackward input-gradient shape mismatch');
        if (inShape.length < 1 || inShape.length > MAX_DIMS)
            throw new Error(`WebGPU layernorm expects a rank between 1 and ${MAX_DIMS}`);
        const features = inShape[inShape.length - 1]!;
        if (features <= 0) throw new Error('layernormBackward requires a non-empty normalized dimension');
        const rows = product(inShape) / features;
        if (gammaShape.length !== 1 || gammaShape[0] !== features ||
            gradGammaShape.length !== 1 || gradGammaShape[0] !== features ||
            gradBetaShape.length !== 1 || gradBetaShape[0] !== features)
            throw new Error('layernormBackward parameter shape mismatch');
        if (meanShape.length !== 1 || meanShape[0] !== rows ||
            invStdShape.length !== 1 || invStdShape[0] !== rows)
            throw new Error('layernormBackward saved-stat shape mismatch');

        const gradInput = gpuStorage(gradInputStorage), gradGamma = gpuStorage(gradGammaStorage);
        const gradBeta = gpuStorage(gradBetaStorage), gradOutput = gpuStorage(gradOutputStorage);
        const input = gpuStorage(inputStorage), gamma = gpuStorage(gammaStorage);
        const mean = gpuStorage(meanStorage), invStd = gpuStorage(invStdStorage);
        if (rows > 0) {
            const inputParams = this.params([
                rows, features, inShape.length, 0,
                ...this.pad(gradInputStride), ...this.pad(inShape),
                ...this.pad(gradOutputStride), ...this.pad(inStride),
                meanStride[0]!, invStdStride[0]!, gammaStride[0]!,
            ]);
            this.submit('layernormBackward:input', LAYERNORM_BACKWARD_INPUT_SHADER, [
                { binding: 0, resource: { buffer: gradOutput.buffer } },
                { binding: 1, resource: { buffer: input.buffer } },
                { binding: 2, resource: { buffer: gamma.buffer } },
                { binding: 3, resource: { buffer: mean.buffer } },
                { binding: 4, resource: { buffer: invStd.buffer } },
                { binding: 5, resource: { buffer: gradInput.buffer } },
                { binding: 6, resource: { buffer: inputParams } },
            ], [rows], [inputParams]);
        }

        const paramParams = this.params([
            rows, features, inShape.length, 0,
            ...this.pad(inShape), ...this.pad(gradOutputStride), ...this.pad(inStride),
            meanStride[0]!, invStdStride[0]!, gradGammaStride[0]!, gradBetaStride[0]!,
        ]);
        this.submit('layernormBackward:params', LAYERNORM_BACKWARD_PARAMS_SHADER, [
            { binding: 0, resource: { buffer: gradOutput.buffer } },
            { binding: 1, resource: { buffer: input.buffer } },
            { binding: 2, resource: { buffer: mean.buffer } },
            { binding: 3, resource: { buffer: invStd.buffer } },
            { binding: 4, resource: { buffer: gradGamma.buffer } },
            { binding: 5, resource: { buffer: gradBeta.buffer } },
            { binding: 6, resource: { buffer: paramParams } },
        ], [Math.ceil(features / WORKGROUP_SIZE)], [paramParams]);
    }
}

export async function tryCreateWebGpuBackend(): Promise<WebGpuBackend | null> {
    let gpu: { requestAdapter(): Promise<GPUAdapter | null> } | undefined;
    const browserGpu = (globalThis.navigator as Navigator & { gpu?: typeof gpu } | undefined)?.gpu;
    if (browserGpu) {
        gpu = browserGpu;
    } else if (typeof process !== 'undefined' && process.versions?.node) {
        let module: { globals: object; create(args: unknown[]): unknown };
        try {
            module = await import(/* @vite-ignore */ 'webgpu');
        } catch (error) {
            if (process.env?.TSGRAD_DEBUG_BACKEND) console.warn('Unable to load Node WebGPU provider', error);
            return null;
        }
        Object.assign(globalThis, module.globals);
        gpu = module.create([]) as unknown as typeof gpu;
    }
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    return new WebGpuBackend(await adapter.requestDevice());
}
