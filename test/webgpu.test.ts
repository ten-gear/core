import { beforeAll, describe, expect, test } from 'vitest';
import { setBackendForTesting } from '../src/backend/index.js';
import { tryCreateWebGpuBackend, WebGpuBackend } from '../src/backend/webgpu/webgpu_backend.js';
import { Tensor } from '../src/tensor.js';
import { conv1d, embedding, embeddingBackward, gather, logsoftmax, scatter, softmax, tocontiguous } from '../src/tensor_functions.js';
import { readStorageAsync, TensorData } from '../src/tensor_data.js';

const backend = await tryCreateWebGpuBackend();

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i]!);
}

describe.skipIf(!backend)('WebGPU backend', () => {
    beforeAll(() => setBackendForTesting(backend as WebGpuBackend));

    test('keeps chained map, zip, and reduce operations on device', async () => {
        const input = new Tensor(new TensorData(new Float32Array([-2, 1, -4, 3]), [2, 2]));
        const result = input.relu().add(2).mul(input.relu()).sum(1);
        expectClose(await result.toFloat32(), [3, 15]);
    });

    test('supports broadcast and permuted views', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([10, 20]), [2, 1]));
        expectClose(await a.add(b).toFloat32(), [11, 12, 13, 24, 25, 26]);
        expectClose(await a.permute(1, 0).tocontiguous().toFloat32(), [1, 4, 2, 5, 3, 6]);
    });

    test('covers unary, comparison, and reduction operators', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [2, 2]));
        expectClose(await a.neg().exp().toFloat32(), [Math.exp(-1), Math.exp(-2), Math.exp(-3), Math.exp(-4)]);
        expectClose(await a.sigmoid().toFloat32(), [0.7310586, 0.880797, 0.952574, 0.982014]);
        expectClose(await a.gelu().toFloat32(), [0.8413447, 1.9544997, 2.9959503, 3.9998733]);
        expectClose(await a.silu().toFloat32(), [0.7310586, 1.7615942, 2.8577224, 3.9280552]);
        expectClose(await a.sqrt().toFloat32(), [1, Math.sqrt(2), Math.sqrt(3), 2]);
        expectClose(await a.sub(2).div(2).toFloat32(), [-0.5, 0, 0.5, 1]);
        expectClose(await a.lt(3).toFloat32(), [1, 1, 0, 0]);
        expectClose(await a.eq(3).toFloat32(), [0, 0, 1, 0]);
        expectClose(await a.max(1).toFloat32(), [2, 4]);

        const geluInput = new Tensor(new TensorData(new Float32Array([-1, 0, 1, 4]), [4]));
        geluInput._requiresGrad(true);
        geluInput.gelu().backward();
        expectClose(await geluInput.grad!.toFloat32(), [-0.08331547, 0.5, 1.08331547, 1.00050365]);

        const siluInput = new Tensor(new TensorData(new Float32Array([-1, 0, 1, 4]), [4]));
        siluInput._requiresGrad(true);
        siluInput.silu().backward();
        expectClose(await siluInput.grad!.toFloat32(), [0.07232949, 0.5, 0.92767051, 1.05266461]);
    });

    test('runs tiled matmul and convolution', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));
        expectClose(await a.matmul(b).toFloat32(), [58, 64, 139, 154]);

        const input = new TensorData(new Float32Array([1, 2, 3, 4]), [1, 1, 4]);
        const weight = new TensorData(new Float32Array([1, 0, -1]), [1, 1, 3]);
        expectClose(await readStorageAsync(conv1d(input, weight)._storage), [-2, -2, 3, 4]);
    });

    test('runs matmul backward through strided views', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));
        a._requiresGrad(true);
        b._requiresGrad(true);
        a.matmul(b).sum(0).sum(1).backward();
        expectClose(await a.grad!.toFloat32(), [15, 19, 23, 15, 19, 23]);
        expectClose(await b.grad!.toFloat32(), [5, 5, 7, 7, 9, 9]);
    });

    test('updates parameters with SGD kernels', async () => {
        const param = backend!.alloc(3);
        const grad = backend!.alloc(3);
        backend!.copyFromHost(param, new Float32Array([1, 2, 3]));
        backend!.copyFromHost(grad, new Float32Array([0.5, -1, 2]));
        backend!.sgd(param, [3], [1], grad, [3], [1], 0.1);
        expectClose(await backend!.copyToHostAsync(param), [0.95, 2.1, 2.8]);
        backend!.release(param);
        backend!.release(grad);
    });

    test('updates parameters and moments with AdamW', async () => {
        const param = backend!.alloc(3);
        const grad = backend!.alloc(3);
        const m = backend!.alloc(3, 0);
        const v = backend!.alloc(3, 0);
        backend!.copyFromHost(param, new Float32Array([1, 2, 3]));
        backend!.copyFromHost(grad, new Float32Array([0.5, -1, 2]));
        backend!.adamw(param, [3], [1], grad, [3], [1], m, [3], [1], v, [3], [1], 0.1, 0.9, 0.999, 1e-8, 0.01, 1);
        expectClose(await backend!.copyToHostAsync(param), [0.899, 2.098, 2.897]);
        expectClose(await backend!.copyToHostAsync(m), [0.05, -0.1, 0.2]);
        expectClose(await backend!.copyToHostAsync(v), [0.00025, 0.001, 0.004]);
        backend!.release(param);
        backend!.release(grad);
        backend!.release(m);
        backend!.release(v);
    });

    test('gathers through strided index and value views', async () => {
        const values = new TensorData(new Float32Array([10, 11, 12, 20, 21, 22]), [2, 3]);
        const indices = new TensorData(new Float32Array([1, 0, 0, 1]), [2, 2]);
        expectClose(await readStorageAsync(gather(indices, values, 1)._storage), [11, 10, 20, 21]);

        const column = new TensorData(new Float32Array([0, 1, 2, 0]), [2, 2]);
        expectClose(await readStorageAsync(gather(column, values.permute(1, 0), 0)._storage), [10, 21, 12, 20]);

        const wide = new TensorData(new Float32Array([1, 0, 2, 0, 1, 1]), [2, 3]);
        const table = new TensorData(new Float32Array([10, 20, 11, 21, 12, 22, 13, 23]), [4, 2]);
        expectClose(await readStorageAsync(gather(wide.permute(1, 0), table, 0)._storage), [11, 20, 10, 21, 12, 21]);

        const out = backend!.alloc(4);
        const indexStorage = backend!.alloc(4);
        const valueStorage = backend!.alloc(6);
        backend!.copyFromHost(indexStorage, new Float32Array([1, 0, 0, 1]));
        backend!.copyFromHost(valueStorage, new Float32Array([10, 11, 12, 20, 21, 22]));
        backend!.gather(out, [2, 2], [1, 2], indexStorage, [2, 2], [2, 1], valueStorage, [2, 3], [3, 1], 1);
        expectClose(await backend!.copyToHostAsync(out), [11, 20, 10, 21]);
        backend!.release(out);
        backend!.release(indexStorage);
        backend!.release(valueStorage);
    });

    test('scatter-adds duplicate gradient entries', async () => {
        const indices = new TensorData(new Float32Array([1, 0, 0, 1]), [2, 2]);
        const grad = new TensorData(new Float32Array([1, 2, 3, 4]), [2, 2]);
        const out = TensorData.fill([2, 3], 0);
        scatter(out, indices, grad, 1);
        expectClose(await readStorageAsync(out._storage), [2, 1, 0, 3, 4, 0]);

        const flipped = TensorData.fill([2, 3], 0);
        scatter(flipped, indices, grad.permute(1, 0), 1);
        expectClose(await readStorageAsync(flipped._storage), [3, 1, 0, 2, 4, 0]);

        const repeated = TensorData.fill([1, 3], 100);
        scatter(repeated, new TensorData(new Float32Array([0, 0]), [1, 2]), new TensorData(new Float32Array([5, 7]), [1, 2]), 1);
        expectClose(await readStorageAsync(repeated._storage), [112, 100, 100]);
    });

    test('embedding copies rows through a strided table', async () => {
        const stored = new TensorData(new Float32Array([
            0.1, 0.4, 0.7, 1.0, 1.3,
            0.2, 0.5, 0.8, 1.1, 1.4,
            0.3, 0.6, 0.9, 1.2, 1.5,
        ]), [3, 5]);
        const table = stored.permute(1, 0);
        const ids = new TensorData(new Float32Array([4, 1, 4]), [3]);
        const plain = new TensorData(new Float32Array([
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
            1.0, 1.1, 1.2,
            1.3, 1.4, 1.5,
        ]), [5, 3]);
        expectClose(await readStorageAsync(embedding(ids, plain)._storage), [
            1.3, 1.4, 1.5, 0.4, 0.5, 0.6, 1.3, 1.4, 1.5,
        ]);
        expectClose(await readStorageAsync(embedding(ids, table)._storage), [
            1.3, 1.4, 1.5, 0.4, 0.5, 0.6, 1.3, 1.4, 1.5,
        ]);

        const grid = new TensorData(new Float32Array([4, 1, 4, 2]), [2, 2]);
        expectClose(await readStorageAsync(embedding(grid.permute(1, 0), table)._storage), [
            1.3, 1.4, 1.5, 1.3, 1.4, 1.5,
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]);
    });

    test('embedding backward adds repeated token rows', async () => {
        const ids = new TensorData(new Float32Array([4, 1, 4]), [3]);
        const grad = new TensorData(new Float32Array([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]), [3, 3]);
        const expected = [
            0, 0, 0,
            4, 5, 6,
            0, 0, 0,
            0, 0, 0,
            8, 10, 12,
        ];
        const table = new TensorData(new Float32Array(15), [5, 3]);
        embeddingBackward(table, ids, grad);
        expectClose(await readStorageAsync(table._storage), expected);

        const viewed = new TensorData(new Float32Array(15), [3, 5]).permute(1, 0);
        const gradView = new TensorData(new Float32Array([
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
        ]), [3, 3]).permute(1, 0);
        embeddingBackward(viewed, ids, gradView);
        expectClose(await readStorageAsync(tocontiguous(viewed)._storage), expected);
    });

    test('runs layernorm forward and backward on the last dimension', async () => {
        const inputValues = new Float32Array([0.3, -1.2, 2.1, 1.5, 0.2, -0.7]);
        const gammaValues = new Float32Array([1.1, -0.4, 0.8]);
        const betaValues = new Float32Array([0.2, -0.3, 0.5]);
        const upstream = new Float32Array([0.5, -1, 0.25, 1.5, 0.75, -0.5]);
        const expectedOutput = new Float32Array(6);
        const expectedInputGrad = new Float32Array(6);
        const expectedGammaGrad = new Float32Array(3);
        const expectedBetaGrad = new Float32Array(3);

        for (let row = 0; row < 2; row++) {
            const base = row * 3;
            const mean = (inputValues[base]! + inputValues[base + 1]! + inputValues[base + 2]!) / 3;
            const variance = (
                (inputValues[base]! - mean) ** 2 +
                (inputValues[base + 1]! - mean) ** 2 +
                (inputValues[base + 2]! - mean) ** 2
            ) / 3;
            const invStd = 1 / Math.sqrt(variance + 1e-5);
            let sumQ = 0;
            let sumQXHat = 0;
            for (let col = 0; col < 3; col++) {
                const index = base + col;
                const xHat = (inputValues[index]! - mean) * invStd;
                const q = upstream[index]! * gammaValues[col]!;
                expectedOutput[index] = gammaValues[col]! * xHat + betaValues[col]!;
                expectedGammaGrad[col]! += upstream[index]! * xHat;
                expectedBetaGrad[col]! += upstream[index]!;
                sumQ += q;
                sumQXHat += q * xHat;
            }
            for (let col = 0; col < 3; col++) {
                const index = base + col;
                const xHat = (inputValues[index]! - mean) * invStd;
                const q = upstream[index]! * gammaValues[col]!;
                expectedInputGrad[index] = invStd * (q - sumQ / 3 - xHat * sumQXHat / 3);
            }
        }

        const input = new Tensor(new TensorData(inputValues, [1, 2, 3]));
        const gamma = new Tensor(new TensorData(gammaValues, [3]));
        const beta = new Tensor(new TensorData(betaValues, [3]));
        input._requiresGrad(true);
        gamma._requiresGrad(true);
        beta._requiresGrad(true);
        const output = input.layernorm(gamma, beta);
        expectClose(await output.toFloat32(), expectedOutput);
        output.backward(new Tensor(new TensorData(upstream, [1, 2, 3])));
        expectClose(await input.grad!.toFloat32(), expectedInputGrad);
        expectClose(await gamma.grad!.toFloat32(), expectedGammaGrad);
        expectClose(await beta.grad!.toFloat32(), expectedBetaGrad);
    });

    test('runs softmax and logsoftmax forward and backward', async () => {
        const values = new Float32Array([0.3, -1.2, 2.1, 1.5, 0.2, -0.7]);
        const upstream = new Float32Array([0.5, -1, 0.25, 1.5, 0.75, -0.5]);
        const probabilities = new Float32Array(6);
        const softmaxGrad = new Float32Array(6);
        const logsoftmaxGrad = new Float32Array(6);
        for (let row = 0; row < 2; row++) {
            const offset = row * 3;
            const maximum = Math.max(values[offset]!, values[offset + 1]!, values[offset + 2]!);
            let sum = 0;
            for (let col = 0; col < 3; col++)
                sum += probabilities[offset + col] = Math.exp(values[offset + col]! - maximum);
            let dot = 0;
            let gradSum = 0;
            for (let col = 0; col < 3; col++) {
                probabilities[offset + col]! /= sum;
                dot += upstream[offset + col]! * probabilities[offset + col]!;
                gradSum += upstream[offset + col]!;
            }
            for (let col = 0; col < 3; col++) {
                softmaxGrad[offset + col] =
                    probabilities[offset + col]! * (upstream[offset + col]! - dot);
                logsoftmaxGrad[offset + col] =
                    upstream[offset + col]! - probabilities[offset + col]! * gradSum;
            }
        }

        const softmaxInput = new Tensor(new TensorData(Float32Array.from(values), [2, 3]));
        softmaxInput._requiresGrad(true);
        const softmaxOutput = softmax(softmaxInput, -1);
        expectClose(await softmaxOutput.toFloat32(), probabilities);
        softmaxOutput.backward(new Tensor(new TensorData(Float32Array.from(upstream), [2, 3])));
        expectClose(await softmaxInput.grad!.toFloat32(), softmaxGrad);

        const logsoftmaxInput = new Tensor(new TensorData(Float32Array.from(values), [2, 3]));
        logsoftmaxInput._requiresGrad(true);
        const logsoftmaxOutput = logsoftmax(logsoftmaxInput, 1);
        expectClose(await logsoftmaxOutput.exp().toFloat32(), probabilities);
        logsoftmaxOutput.backward(new Tensor(new TensorData(Float32Array.from(upstream), [2, 3])));
        expectClose(await logsoftmaxInput.grad!.toFloat32(), logsoftmaxGrad);

        const strided = new Tensor(
            new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]),
        ).permute(1, 0);
        const denominator = Math.exp(-2) + Math.exp(-1) + 1;
        const dimProbabilities = [Math.exp(-2), Math.exp(-1), 1].map(value => value / denominator);
        expectClose(await softmax(strided, 0).toFloat32(), [
            dimProbabilities[0]!, dimProbabilities[0]!,
            dimProbabilities[1]!, dimProbabilities[1]!,
            dimProbabilities[2]!, dimProbabilities[2]!,
        ]);
    });

    test('reuses pooled buffers without violating queue order', async () => {
        const first = backend!.alloc(4, 7);
        const firstBuffer = first.buffer;
        backend!.release(first);
        const second = backend!.alloc(4, 0);
        expect(second.buffer).toBe(firstBuffer);
        expectClose(await backend!.copyToHostAsync(second), [0, 0, 0, 0]);
        backend!.release(second);
    });

    test('readback cleanup does not destroy parameters for later submits', async () => {
        const input = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [4]));
        const first = input.add(1);
        const readback = first.toFloat32();
        let later = input;
        for (let i = 0; i < 20; i++) later = later.mul(1.01).add(0.25);
        expectClose(await readback, [2, 3, 4, 5]);
        const values = await later.toFloat32();
        expect(values.every(Number.isFinite)).toBe(true);
    });
});
