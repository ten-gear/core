import { expect, test } from "vitest";
import { CpuBackend } from "../src/backend/cpu_backend.js";
import { getBackend, initBackend, setBackendForTesting } from "../src/backend/index.js";
import { FlashAttention as FlashAttentionLayer } from "../src/layers.js";
import { Tensor } from "../src/tensor.js";
import { flashAttention } from "../src/tensor_functions.js";
import { readStorage, TensorData } from "../src/tensor_data.js";

await initBackend({ backend: "cpu" });

type AttentionReference = {
    output: Float32Array;
    gradQuery: Float32Array;
    gradKey: Float32Array;
    gradValue: Float32Array;
};

function referenceAttention(
    query: Float32Array, key: Float32Array, value: Float32Array,
    gradOutput: Float32Array, time: number, headDim: number,
    scale: number, causal: boolean,
): AttentionReference {
    const output = new Float32Array(time * headDim);
    const probabilities = new Float32Array(time * time);
    for (let i = 0; i < time; i++) {
        const keyLimit = causal ? i + 1 : time;
        const scores = new Float32Array(keyLimit);
        let maximum = -Infinity;
        for (let j = 0; j < keyLimit; j++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++)
                dot += query[i * headDim + d]! * key[j * headDim + d]!;
            scores[j] = scale * dot;
            maximum = Math.max(maximum, scores[j]!);
        }
        let denominator = 0;
        for (let j = 0; j < keyLimit; j++)
            denominator += Math.exp(scores[j]! - maximum);
        for (let j = 0; j < keyLimit; j++) {
            const probability = Math.exp(scores[j]! - maximum) / denominator;
            probabilities[i * time + j] = probability;
            for (let d = 0; d < headDim; d++)
                output[i * headDim + d]! += probability * value[j * headDim + d]!;
        }
    }

    const gradQuery = new Float32Array(query.length);
    const gradKey = new Float32Array(key.length);
    const gradValue = new Float32Array(value.length);
    for (let i = 0; i < time; i++) {
        let delta = 0;
        for (let d = 0; d < headDim; d++)
            delta += output[i * headDim + d]! * gradOutput[i * headDim + d]!;
        const keyLimit = causal ? i + 1 : time;
        for (let j = 0; j < keyLimit; j++) {
            let gradValueDot = 0;
            for (let d = 0; d < headDim; d++)
                gradValueDot += gradOutput[i * headDim + d]! * value[j * headDim + d]!;
            const probability = probabilities[i * time + j]!;
            const gradScore = probability * (gradValueDot - delta);
            for (let d = 0; d < headDim; d++) {
                gradQuery[i * headDim + d]! += scale * gradScore * key[j * headDim + d]!;
                gradKey[j * headDim + d]! += scale * gradScore * query[i * headDim + d]!;
                gradValue[j * headDim + d]! += probability * gradOutput[i * headDim + d]!;
            }
        }
    }
    return { output, gradQuery, gradKey, gradValue };
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, digits = 4): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++)
        expect(actual[i]).toBeCloseTo(expected[i]!, digits);
}

test("fused causal attention matches a materialized reference forward and backward", () => {
    const previous = getBackend();
    setBackendForTesting(new CpuBackend());
    try {
        const time = 3, headDim = 32, scale = 1 / Math.sqrt(headDim);
        const queryValues = Float32Array.from(
            { length: time * headDim }, (_, i) => Math.sin(i * 0.17) * 0.3,
        );
        const keyValues = Float32Array.from(
            { length: time * headDim }, (_, i) => Math.cos(i * 0.11) * 0.2,
        );
        const valueValues = Float32Array.from(
            { length: time * headDim }, (_, i) => ((i % 11) - 5) * 0.07,
        );
        const gradValues = Float32Array.from(
            { length: time * headDim }, (_, i) => Math.sin(i * 0.07) * 0.4,
        );
        const expected = referenceAttention(
            queryValues, keyValues, valueValues, gradValues, time, headDim, scale, true,
        );

        const query = new Tensor(new TensorData(queryValues, [1, 1, time, headDim]));
        const key = new Tensor(new TensorData(keyValues, [1, 1, time, headDim]));
        const value = new Tensor(new TensorData(valueValues, [1, 1, time, headDim]));
        query._requiresGrad(true);
        key._requiresGrad(true);
        value._requiresGrad(true);
        const output = flashAttention(query, key, value, true, scale);
        expectClose(readStorage(output.data._storage), expected.output);
        output.backward(new Tensor(new TensorData(gradValues, [1, 1, time, headDim])));
        expectClose(readStorage(query.grad!.data._storage), expected.gradQuery);
        expectClose(readStorage(key.grad!.data._storage), expected.gradKey);
        expectClose(readStorage(value.grad!.data._storage), expected.gradValue);
    }
    finally {
        setBackendForTesting(previous);
    }
});

test("FlashAttention layer supports non-causal self and explicit QKV attention", () => {
    const previous = getBackend();
    setBackendForTesting(new CpuBackend());
    try {
        const input = new Tensor(TensorData.fill([1, 1, 2, 32], 0.25));
        const layer = new FlashAttentionLayer(false);
        expectClose(
            readStorage(layer.forward(input).data._storage),
            readStorage(flashAttention(input, input, input).data._storage),
        );
        expectClose(
            readStorage(layer.forward(input, input, input).data._storage),
            readStorage(flashAttention(input, input, input).data._storage),
        );
    }
    finally {
        setBackendForTesting(previous);
    }
});
