import { expect, test } from "vitest";
import { CpuBackend } from "../src/backend/cpu_backend.js";
import { getBackend, initBackend, setBackendForTesting } from "../src/backend/index.js";
import { RoPE as RoPELayer } from "../src/layers.js";
import { Tensor } from "../src/tensor.js";
import { rope } from "../src/tensor_functions.js";
import { readStorage, TensorData } from "../src/tensor_data.js";

await initBackend({ backend: "cpu" });

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++)
        expect(actual[i]).toBeCloseTo(expected[i]!);
}

test("RoPE applies an offset and backpropagates the inverse rotation", () => {
    const previous = getBackend();
    setBackendForTesting(new CpuBackend());
    try {
        const input = new Tensor(new TensorData(
            new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [1, 2, 4],
        ));
        const sin = new Tensor(new TensorData(
            new Float32Array([0, 0, 1, 0, 0, 1]), [3, 2],
        ));
        const cos = new Tensor(new TensorData(
            new Float32Array([1, 1, 0, 1, 1, 0]), [3, 2],
        ));
        input._requiresGrad(true);

        const output = rope(input, sin, cos, 1);
        expectClose(readStorage(output.data._storage), [-3, 2, 1, 4, 5, -8, 7, 6]);
        output.backward(new Tensor(TensorData.fill([1, 2, 4], 1)));
        expectClose(readStorage(input.grad!.data._storage), [1, 1, -1, 1, 1, 1, 1, -1]);
    }
    finally {
        setBackendForTesting(previous);
    }
});

test("RoPE layer creates frequency tables and accepts a forward offset", () => {
    const previous = getBackend();
    setBackendForTesting(new CpuBackend());
    try {
        const layer = new RoPELayer(4, 4);
        const input = new Tensor(new TensorData(
            new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [1, 2, 4],
        ));
        expectClose(
            readStorage(layer.forward(input, 1).data._storage),
            readStorage(rope(input, layer.sin, layer.cos, 1).data._storage),
        );
    }
    finally {
        setBackendForTesting(previous);
    }
});
