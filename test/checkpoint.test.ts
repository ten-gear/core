import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, expect, test } from "vitest";
import { CpuBackend } from "../src/backend/cpu_backend.js";
import { initBackend, setBackendForTesting } from "../src/backend/index.js";
import { Module, Parameter } from "../src/module.js";
import { AdamW } from "../src/optim.js";
import { Scalar } from "../src/scalar.js";
import { Tensor } from "../src/tensor.js";
import { readStorage, TensorData, type TensorBuffer } from "../src/tensor_data.js";
import {
    loadLatestCheckpoint,
    saveCheckpoint,
    type CheckpointHyperparameters,
    type CheckpointModelConfig,
} from "../project/checkpoint.js";
import { Random, sampleBatch } from "../project/tinystories_data.js";

await initBackend({ backend: "cpu" });

beforeEach(() => {
    setBackendForTesting(new CpuBackend());
});

const modelConfig: CheckpointModelConfig = {
    vocab: 8,
    nEmbd: 4,
    nHead: 1,
    nLayer: 1,
    maxSeq: 4,
};

const hyperparameters: CheckpointHyperparameters = {
    learningRate: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.01,
    batchSize: 2,
    blockSize: 4,
};

class OneParameter extends Module {
    constructor(values: number[]) {
        super();
        const data = Float32Array.from(values);
        const tensor = new Tensor(new TensorData(data, [data.length]));
        tensor.grad = Tensor.zeros([data.length]);
        this.addParameter("weight", tensor);
    }
}

class ScalarParameter extends Module {
    constructor() {
        super();
        this.addParameter("bias", new Scalar(1));
    }
}

class ViewedParameter extends Module {
    constructor() {
        super();
        const data = TensorData.fill([2, 2], 1);
        this.addParameter("weight", new Tensor(data.permute(1, 0)));
    }
}

test("Random repeats its sequence after setState", () => {
    const first = new Random(1337);
    const values = [first.next(), first.next(), first.next()];
    for (const value of values) {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
    }
    const state = first.getState();
    const following = first.next();

    const second = new Random(1337);
    expect([second.next(), second.next(), second.next()]).toEqual(values);

    const restored = new Random(0);
    restored.setState(state);
    expect(restored.next()).toBe(following);
    expect(restored.getState()).toBe(first.getState());

    const tokens = Uint16Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const left = sampleBatch(tokens, 2, 3, new Random(5));
    const right = sampleBatch(tokens, 2, 3, new Random(5));
    expect(Array.from(left.input)).toEqual(Array.from(right.input));
    expect(Array.from(left.target)).toEqual(Array.from(right.target));
});

test("a checkpoint round-trips weights, Adam state, and the generator", async () => {
    await withRoot(async root => {
        const source = new OneParameter([1, -2, 0.5]);
        const sourceOptim = optimizerFor(source);
        const sourceWeight = parameter(source);
        (sourceWeight.value as Tensor).grad = new Tensor(new TensorData(new Float32Array([0.5, -0.25, 0.125]), [3]));
        sourceOptim.step();

        const weights = host(sourceWeight.value as Tensor);
        const first = hostBuffer(sourceOptim.m.get(sourceWeight));
        const second = hostBuffer(sourceOptim.v.get(sourceWeight));
        const adamStep = sourceOptim.steps.get(sourceWeight);

        const rng = new Random(99);
        rng.next();
        const rngState = rng.getState();
        const following = new Random(0);
        following.setState(rngState);
        const nextDraw = following.next();

        await saveCheckpoint({
            root,
            step: 4,
            seed: 99,
            rng,
            model: source,
            optim: sourceOptim,
            modelConfig,
            hyperparameters,
        });

        const restored = new OneParameter([0, 0, 0]);
        const restoredOptim = optimizerFor(restored);
        const loaded = await loadLatestCheckpoint(root, restored, restoredOptim, { model: modelConfig, hyperparameters });
        expect(loaded).toEqual({ step: 4, rngState });

        const restoredWeight = parameter(restored);
        expect(Array.from(host(restoredWeight.value as Tensor))).toEqual(Array.from(weights));
        expect(Array.from(hostBuffer(restoredOptim.m.get(restoredWeight)))).toEqual(Array.from(first));
        expect(Array.from(hostBuffer(restoredOptim.v.get(restoredWeight)))).toEqual(Array.from(second));
        expect(restoredOptim.steps.get(restoredWeight)).toBe(adamStep);

        const resumed = new Random(0);
        resumed.setState(loaded!.rngState);
        expect(resumed.next()).toBe(nextDraw);
    });
});

test("the checkpoint directory and latest pointer appear only after the files are synced", async () => {
    await withRoot(async root => {
        const finalDir = join(root, "step-0");
        const latest = join(root, "latest");
        const temporary = join(root, `.tmp-step-0-${process.pid}`);
        const model = new OneParameter([4, 5]);
        await saveCheckpoint({
            root,
            step: 0,
            seed: 1,
            rng: new Random(1),
            model,
            optim: optimizerFor(model),
            modelConfig,
            hyperparameters,
            beforePublish() {
                expect(existsSync(finalDir)).toBe(false);
                expect(existsSync(latest)).toBe(false);
                expect(existsSync(temporary)).toBe(true);
                for (const name of ["model.bin", "optimizer.bin", "training-state.json", "rng-state.bin"])
                    expect(existsSync(join(temporary, name))).toBe(true);
            },
        });

        expect(existsSync(finalDir)).toBe(true);
        expect(existsSync(temporary)).toBe(false);
        expect(readFileSync(latest, "utf8")).toBe("step-0\n");
        for (const name of ["model.bin", "optimizer.bin", "training-state.json", "rng-state.bin"])
            expect(existsSync(join(finalDir, name))).toBe(true);

        const restored = new OneParameter([0, 0]);
        const restoredOptim = optimizerFor(restored);
        const loaded = await loadLatestCheckpoint(root, restored, restoredOptim, { model: modelConfig, hyperparameters });
        expect(loaded).toEqual({ step: 0, rngState: new Random(1).getState() });
        expect(Array.from(host(parameter(restored).value as Tensor))).toEqual([4, 5]);
        expect(Array.from(hostBuffer(restoredOptim.m.get(parameter(restored))))).toEqual([0, 0]);
        expect(Array.from(hostBuffer(restoredOptim.v.get(parameter(restored))))).toEqual([0, 0]);
        expect(restoredOptim.steps.get(parameter(restored))).toBe(0);
    });
});

test("a failed publish keeps the previous latest pointer", async () => {
    await withRoot(async root => {
        const model = new OneParameter([1, 2]);
        await saveCheckpoint({
            root,
            step: 0,
            seed: 1,
            rng: new Random(1),
            model,
            optim: optimizerFor(model),
            modelConfig,
            hyperparameters,
        });

        const replacement = new OneParameter([3, 4]);
        await expect(saveCheckpoint({
            root,
            step: 1,
            seed: 1,
            rng: new Random(1),
            model: replacement,
            optim: optimizerFor(replacement),
            modelConfig,
            hyperparameters,
            beforePublish() {
                throw new Error("stop before publish");
            },
        })).rejects.toThrow("stop before publish");

        expect(existsSync(join(root, "step-1"))).toBe(false);
        expect(existsSync(join(root, `.tmp-step-1-${process.pid}`))).toBe(false);
        expect(readFileSync(join(root, "latest"), "utf8")).toBe("step-0\n");

        const restored = new OneParameter([0, 0]);
        const loaded = await loadLatestCheckpoint(root, restored, optimizerFor(restored), { model: modelConfig, hyperparameters });
        expect(loaded?.step).toBe(0);
        expect(Array.from(host(parameter(restored).value as Tensor))).toEqual([1, 2]);
    });
});

test("replacing a checkpoint keeps the saved bytes and drops the previous directory", async () => {
    await withRoot(async root => {
        const first = new OneParameter([1, 2]);
        await saveCheckpoint({
            root,
            step: 7,
            seed: 3,
            rng: new Random(3),
            model: first,
            optim: optimizerFor(first),
            modelConfig,
            hyperparameters,
        });
        const second = new OneParameter([8, 9]);
        await saveCheckpoint({
            root,
            step: 7,
            seed: 3,
            rng: new Random(4),
            model: second,
            optim: optimizerFor(second),
            modelConfig,
            hyperparameters,
        });

        expect(existsSync(`${join(root, "step-7")}.replacing`)).toBe(false);
        const restored = new OneParameter([0, 0]);
        const loaded = await loadLatestCheckpoint(root, restored, optimizerFor(restored), { model: modelConfig, hyperparameters });
        expect(loaded?.step).toBe(7);
        expect(loaded?.rngState).toBe(new Random(4).getState());
        expect(Array.from(host(parameter(restored).value as Tensor))).toEqual([8, 9]);
    });
});

test("resume rejects a hyperparameter mismatch before changing the model", async () => {
    await withRoot(async root => {
        const source = new OneParameter([1, 2, 3]);
        await saveCheckpoint({
            root,
            step: 2,
            seed: 1,
            rng: new Random(1),
            model: source,
            optim: optimizerFor(source),
            modelConfig,
            hyperparameters,
        });

        const fresh = new OneParameter([0, 0, 0]);
        await expect(loadLatestCheckpoint(root, fresh, optimizerFor(fresh), {
            model: modelConfig,
            hyperparameters: { ...hyperparameters, learningRate: 0.2 },
        })).rejects.toThrow(/hyperparameters\.learningRate/);
        expect(Array.from(host(parameter(fresh).value as Tensor))).toEqual([0, 0, 0]);
    });
});

test("scalar and non-contiguous parameters are refused before a checkpoint directory exists", async () => {
    await withRoot(async root => {
        const scalar = new ScalarParameter();
        await expect(saveCheckpoint({
            root,
            step: 0,
            seed: 1,
            rng: new Random(1),
            model: scalar,
            optim: new AdamW(scalar.parameters(), 0.1),
            modelConfig,
            hyperparameters,
        })).rejects.toThrow(/not a tensor/);

        const viewed = new ViewedParameter();
        await expect(saveCheckpoint({
            root,
            step: 0,
            seed: 1,
            rng: new Random(1),
            model: viewed,
            optim: new AdamW(viewed.parameters(), 0.1),
            modelConfig,
            hyperparameters,
        })).rejects.toThrow(/not contiguous row-major/);

        expect(existsSync(join(root, "step-0"))).toBe(false);
        expect(existsSync(join(root, "latest"))).toBe(false);
        expect(existsSync(join(root, `.tmp-step-0-${process.pid}`))).toBe(false);
    });
});

test("a missing latest pointer starts a new run", async () => {
    await withRoot(async root => {
        const model = new OneParameter([1]);
        expect(await loadLatestCheckpoint(root, model, optimizerFor(model), { model: modelConfig, hyperparameters })).toBeNull();
    });
});

function optimizerFor(model: Module): AdamW {
    return new AdamW(
        model.parameters(),
        hyperparameters.learningRate,
        hyperparameters.beta1,
        hyperparameters.beta2,
        hyperparameters.eps,
        hyperparameters.weightDecay,
    );
}

function parameter(model: Module): Parameter {
    const found = model.parameters()[0];
    if (found === undefined)
        throw new Error("expected a parameter");
    return found;
}

function host(tensor: Tensor): Float32Array {
    return new Float32Array(readStorage(tensor.data._storage));
}

function hostBuffer(buffer: number | TensorBuffer | undefined): Float32Array {
    if (typeof buffer === "number" || buffer === undefined)
        throw new Error("missing Adam moment");
    return new Float32Array(readStorage(buffer));
}

async function withRoot(run: (root: string) => Promise<void>): Promise<void> {
    const root = await mkdtemp(join(tmpdir(), "tengear-ckpt-"));
    try {
        await run(root);
    }
    finally {
        await rm(root, { recursive: true, force: true });
    }
}
