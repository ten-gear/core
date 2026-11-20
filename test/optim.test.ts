import { expect, test } from "vitest";
import { CpuBackend } from "../src/backend/cpu_backend.js";
import { getBackend, initBackend, setBackendForTesting } from "../src/backend/index.js";
import { Parameter } from "../src/module.js";
import { AdamW } from "../src/optim.js";
import { Scalar } from "../src/scalar.js";
import { Tensor } from "../src/tensor.js";
import { readStorage, TensorData } from "../src/tensor_data.js";

await initBackend({ backend: "cpu" });

test("AdamW updates tensor and scalar parameters with moment estimates", () => {
    const previous = getBackend();
    setBackendForTesting(new CpuBackend());
    try {
        const tensor = new Tensor(new TensorData(new Float32Array([1, -2]), [2]));
        tensor.grad = new Tensor(new TensorData(new Float32Array([0.5, -0.25]), [2]));
        const scalar = new Scalar(2);
        scalar.derivative = -0.25;
        const tensorParameter = new Parameter(tensor);
        const scalarParameter = new Parameter(scalar);
        const optimizer = new AdamW([tensorParameter, scalarParameter], 0.1, 0.9, 0.999, 1e-8, 0.01);

        optimizer.step();
        expect(readStorage(tensor.data._storage)[0]).toBeCloseTo(0.899);
        expect(readStorage(tensor.data._storage)[1]).toBeCloseTo(-1.898);
        expect(scalar.data).toBeCloseTo(2.098);

        tensor.grad = new Tensor(new TensorData(new Float32Array([0.25, -0.5]), [2]));
        scalar.derivative = 0.5;
        optimizer.step();

        const expected = (parameter: number, firstGrad: number, secondGrad: number): number => {
            const firstStep = parameter * 0.999 - 0.1 * firstGrad / (Math.abs(firstGrad) + 1e-8);
            const m = 0.9 * (0.1 * firstGrad) + 0.1 * secondGrad;
            const v = 0.999 * (0.001 * firstGrad * firstGrad) + 0.001 * secondGrad * secondGrad;
            const mHat = m / (1 - 0.9 ** 2);
            const vHat = v / (1 - 0.999 ** 2);
            return firstStep * 0.999 - 0.1 * mHat / (Math.sqrt(vHat) + 1e-8);
        };
        expect(readStorage(tensor.data._storage)[0]).toBeCloseTo(expected(1, 0.5, 0.25));
        expect(readStorage(tensor.data._storage)[1]).toBeCloseTo(expected(-2, -0.25, -0.5));
        expect(scalar.data).toBeCloseTo(expected(2, -0.25, 0.5));
    }
    finally {
        setBackendForTesting(previous);
    }
});
