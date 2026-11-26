import { existsSync } from "node:fs";
import { mkdir, open, readFile, rename, rm, stat, type FileHandle } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { getBackend } from "../src/backend/index.js";
import { Module, Parameter } from "../src/module.js";
import { AdamW } from "../src/optim.js";
import { Tensor } from "../src/tensor.js";
import { readStorageAsync, TensorData, writeStorage, type TensorBuffer } from "../src/tensor_data.js";
import { Random } from "./tinystories_data.js";

export const checkpointDirectory = fileURLToPath(new URL("./checkpoints/", import.meta.url));

const CHECKPOINT_VERSION = 1;
const ZERO_CHUNK = new Float32Array(256 * 1024);
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export type CheckpointModelConfig = {
    vocab: number;
    nEmbd: number;
    nHead: number;
    nLayer: number;
    maxSeq: number;
};

export type CheckpointHyperparameters = {
    learningRate: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    batchSize: number;
    blockSize: number;
};

export type CheckpointSaveOptions = {
    root: string;
    step: number;
    seed: number;
    rng: Random;
    model: Module;
    optim: AdamW;
    modelConfig: CheckpointModelConfig;
    hyperparameters: CheckpointHyperparameters;
    beforePublish?: () => void | Promise<void>;
};

type ParameterRecord = {
    name: string;
    shape: number[];
    numel: number;
    adamStep: number;
};

type TrainingState = {
    version: number;
    step: number;
    seed: number;
    model: CheckpointModelConfig;
    hyperparameters: CheckpointHyperparameters;
    parameters: ParameterRecord[];
};

type SavedParameter = ParameterRecord & {
    tensor: Tensor;
    moment: TensorBuffer | null;
    variance: TensorBuffer | null;
};

export async function saveCheckpoint(options: CheckpointSaveOptions): Promise<string> {
    const { root, step, seed, rng, model, optim, modelConfig, hyperparameters, beforePublish } = options;
    assertNonNegativeInteger(step, "checkpoint step");
    if (!Number.isFinite(seed))
        throw new Error("checkpoint seed must be a finite number");

    await getBackend().synchronize();
    const parameters = collectParameters(model, optim);
    const name = `step-${step}`;
    const finalDir = join(root, name);
    const temporary = join(root, `.tmp-${name}-${process.pid}`);

    await mkdir(root, { recursive: true });
    await rm(temporary, { recursive: true, force: true });
    await mkdir(temporary);

    try {
        const state: TrainingState = {
            version: CHECKPOINT_VERSION,
            step,
            seed,
            model: modelConfig,
            hyperparameters,
            parameters: parameters.map(({ name: parameterName, shape, numel, adamStep }) => ({
                name: parameterName,
                shape,
                numel,
                adamStep,
            })),
        };
        await writeSyncedFile(join(temporary, "training-state.json"), Buffer.from(`${JSON.stringify(state, null, 2)}\n`, "utf8"));
        await writeSyncedFile(join(temporary, "rng-state.bin"), encodeRngState(rng.getState()));
        await writeModelFile(join(temporary, "model.bin"), parameters);
        await writeOptimizerFile(join(temporary, "optimizer.bin"), parameters);
        if (beforePublish !== undefined)
            await beforePublish();
        await replaceDirectory(temporary, finalDir);
        await publishLatest(root, name);
    }
    catch (error) {
        await rm(temporary, { recursive: true, force: true });
        throw error;
    }
    return finalDir;
}

export async function loadLatestCheckpoint(
    root: string,
    model: Module,
    optim: AdamW,
    expected: { model: CheckpointModelConfig; hyperparameters: CheckpointHyperparameters },
): Promise<{ step: number; rngState: number } | null> {
    const latestPath = join(root, "latest");
    if (!existsSync(latestPath))
        return null;

    const name = (await readFile(latestPath, "utf8")).trim();
    if (!/^step-(0|[1-9]\d*)$/.test(name))
        throw new Error(`latest checkpoint pointer is invalid: ${name}`);
    const directory = join(root, name);
    if (!existsSync(directory))
        throw new Error(`latest checkpoint ${name} is missing`);

    const state = parseTrainingState(JSON.parse(await readFile(join(directory, "training-state.json"), "utf8")));
    if (`step-${state.step}` !== name)
        throw new Error(`checkpoint directory ${name} does not match step ${state.step}`);
    assertSameNumbers("model", expected.model, state.model);
    assertSameNumbers("hyperparameters", expected.hyperparameters, state.hyperparameters);

    const named = new Map(model.namedParameters());
    if (named.size !== state.parameters.length)
        throw new Error(`checkpoint has ${state.parameters.length} parameters, model has ${named.size}`);
    for (const record of state.parameters) {
        if (!named.has(record.name))
            throw new Error(`checkpoint parameter ${record.name} is not in the model`);
    }
    for (const parameterName of named.keys()) {
        if (!state.parameters.some(record => record.name === parameterName))
            throw new Error(`model parameter ${parameterName} is missing from the checkpoint`);
    }

    await loadModelFile(join(directory, "model.bin"), state.parameters, named);
    await loadOptimizerFile(join(directory, "optimizer.bin"), state.parameters, named, optim);
    return { step: state.step, rngState: decodeRngState(await readFile(join(directory, "rng-state.bin"))) };
}

function collectParameters(model: Module, optim: AdamW): SavedParameter[] {
    const saved: SavedParameter[] = [];
    for (const [name, parameter] of model.namedParameters()) {
        const value = parameter.value;
        if (!(value instanceof Tensor))
            throw new Error(`parameter ${name} is not a tensor`);
        assertContiguous(name, value);
        const moment = momentBuffer(optim.m, parameter, name, "first");
        const variance = momentBuffer(optim.v, parameter, name, "second");
        if ((moment === null) !== (variance === null))
            throw new Error(`parameter ${name} is missing an Adam moment`);
        const adamStep = optim.steps.get(parameter) ?? 0;
        assertNonNegativeInteger(adamStep, `Adam step for ${name}`);
        saved.push({
            name,
            shape: [...value.data._shape],
            numel: value.data.size,
            adamStep,
            tensor: value,
            moment,
            variance,
        });
    }
    return saved;
}

function momentBuffer(
    moments: Map<Parameter, number | TensorBuffer>,
    parameter: Parameter,
    name: string,
    which: "first" | "second",
): TensorBuffer | null {
    const value = moments.get(parameter);
    if (value === undefined)
        return null;
    if (typeof value === "number")
        throw new Error(`parameter ${name} stores a scalar Adam ${which} moment`);
    return value;
}

function assertContiguous(name: string, tensor: Tensor): void {
    const expected = TensorData.calculateStrides(tensor.data._shape);
    const stride = tensor.data._stride;
    if (expected.length !== stride.length || expected.some((value, index) => value !== stride[index]))
        throw new Error(`parameter ${name} is not contiguous row-major`);
}

async function writeModelFile(path: string, parameters: SavedParameter[]): Promise<void> {
    const file = await open(path, "w");
    try {
        for (const parameter of parameters) {
            const values = await readStorageAsync(parameter.tensor.data._storage);
            if (values.length !== parameter.numel)
                throw new Error(`parameter ${parameter.name} storage has ${values.length} values, expected ${parameter.numel}`);
            await writeFloat32(file, values);
        }
        await file.sync();
    }
    finally {
        await file.close();
    }
}

async function writeOptimizerFile(path: string, parameters: SavedParameter[]): Promise<void> {
    const file = await open(path, "w");
    try {
        for (const parameter of parameters) {
            await writeMoment(file, parameter.moment, parameter.numel, `${parameter.name} first moment`);
            await writeMoment(file, parameter.variance, parameter.numel, `${parameter.name} second moment`);
        }
        await file.sync();
    }
    finally {
        await file.close();
    }
}

async function writeMoment(file: FileHandle, values: TensorBuffer | null, numel: number, label: string): Promise<void> {
    if (values === null) {
        await writeZeros(file, numel);
        return;
    }
    const host = await readStorageAsync(values);
    if (host.length !== numel)
        throw new Error(`${label} has ${host.length} values, expected ${numel}`);
    await writeFloat32(file, host);
}

async function loadModelFile(path: string, manifest: ParameterRecord[], named: Map<string, Parameter>): Promise<void> {
    await assertByteSize(path, manifest.reduce((total, record) => total + record.numel, 0) * 4, "model.bin");
    const file = await open(path, "r");
    try {
        for (const record of manifest) {
            const parameter = named.get(record.name);
            if (parameter === undefined)
                throw new Error(`checkpoint parameter ${record.name} is not in the model`);
            const tensor = parameter.value;
            if (!(tensor instanceof Tensor))
                throw new Error(`parameter ${record.name} is not a tensor`);
            assertContiguous(record.name, tensor);
            if (!sameShape(tensor.data._shape, record.shape) || tensor.data.size !== record.numel)
                throw new Error(`parameter ${record.name} shape does not match the checkpoint`);
            writeStorage(tensor.data._storage, await readFloat32(file, record.numel));
        }
    }
    finally {
        await file.close();
    }
}

async function loadOptimizerFile(
    path: string,
    manifest: ParameterRecord[],
    named: Map<string, Parameter>,
    optim: AdamW,
): Promise<void> {
    const floats = manifest.reduce((total, record) => total + record.numel, 0) * 2;
    await assertByteSize(path, floats * 4, "optimizer.bin");
    const file = await open(path, "r");
    try {
        const backend = getBackend();
        for (const record of manifest) {
            const parameter = named.get(record.name);
            if (parameter === undefined)
                throw new Error(`checkpoint parameter ${record.name} is not in the model`);
            const first = await readFloat32(file, record.numel);
            const second = await readFloat32(file, record.numel);
            const moment = backend.alloc(record.numel, 0);
            const variance = backend.alloc(record.numel, 0);
            writeStorage(moment, first);
            writeStorage(variance, second);
            replaceMoment(optim.m, parameter, moment);
            replaceMoment(optim.v, parameter, variance);
            optim.steps.set(parameter, record.adamStep);
        }
    }
    finally {
        await file.close();
    }
}

function replaceMoment(moments: Map<Parameter, number | TensorBuffer>, parameter: Parameter, next: TensorBuffer): void {
    const previous = moments.get(parameter);
    moments.set(parameter, next);
    if (previous !== undefined && typeof previous !== "number" && previous !== next)
        getBackend().release(previous);
}

async function replaceDirectory(source: string, destination: string): Promise<void> {
    try {
        await rename(source, destination);
        return;
    }
    catch (error) {
        if (!existsSync(destination))
            throw error;
    }

    const backup = `${destination}.replacing`;
    await rm(backup, { recursive: true, force: true });
    await rename(destination, backup);
    try {
        await rename(source, destination);
    }
    catch (error) {
        try {
            await rename(backup, destination);
        }
        catch {
            // The previous checkpoint remains at the backup path.
        }
        throw error;
    }
    await rm(backup, { recursive: true, force: true }).catch(() => {});
}

async function publishLatest(root: string, name: string): Promise<void> {
    const temporary = join(root, `.tmp-latest-${process.pid}`);
    await writeSyncedFile(temporary, Buffer.from(`${name}\n`, "utf8"));
    try {
        await rename(temporary, join(root, "latest"));
    }
    catch (error) {
        await rm(temporary, { force: true });
        throw error;
    }
}

async function writeSyncedFile(path: string, bytes: Buffer): Promise<void> {
    const file = await open(path, "w");
    try {
        await writeBytes(file, bytes);
        await file.sync();
    }
    finally {
        await file.close();
    }
}

async function writeFloat32(file: FileHandle, values: Float32Array): Promise<void> {
    const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
    if (!LITTLE_ENDIAN) {
        for (let offset = 0; offset < bytes.length; offset += 4)
            bytes.writeFloatLE(bytes.readFloatBE(offset), offset);
    }
    await writeBytes(file, bytes);
}

async function writeZeros(file: FileHandle, count: number): Promise<void> {
    let remaining = count;
    while (remaining > 0) {
        const length = Math.min(remaining, ZERO_CHUNK.length);
        await writeFloat32(file, ZERO_CHUNK.subarray(0, length));
        remaining -= length;
    }
}

async function writeBytes(file: FileHandle, bytes: Buffer): Promise<void> {
    let offset = 0;
    while (offset < bytes.length) {
        const { bytesWritten } = await file.write(bytes, offset, bytes.length - offset, null);
        if (bytesWritten === 0)
            throw new Error("failed to write checkpoint");
        offset += bytesWritten;
    }
}

async function readFloat32(file: FileHandle, count: number): Promise<Float32Array> {
    const bytes = new Uint8Array(count * 4);
    let offset = 0;
    while (offset < bytes.length) {
        const { bytesRead } = await file.read(bytes, offset, bytes.length - offset, null);
        if (bytesRead === 0)
            throw new Error("checkpoint file ended early");
        offset += bytesRead;
    }
    if (LITTLE_ENDIAN)
        return new Float32Array(bytes.buffer, bytes.byteOffset, count);
    const values = new Float32Array(count);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < count; index++)
        values[index] = view.getFloat32(index * 4, true);
    return values;
}

function encodeRngState(state: number): Buffer {
    const bytes = Buffer.alloc(8);
    bytes.writeDoubleLE(state, 0);
    return bytes;
}

function decodeRngState(bytes: Buffer): number {
    if (bytes.length !== 8)
        throw new Error(`rng-state.bin is ${bytes.length} bytes, expected 8`);
    const state = bytes.readDoubleLE(0);
    if (!Number.isFinite(state))
        throw new Error("rng-state.bin is not a finite number");
    return state;
}

async function assertByteSize(path: string, expected: number, label: string): Promise<void> {
    if (!existsSync(path))
        throw new Error(`${label} is missing`);
    const { size } = await stat(path);
    if (size !== expected)
        throw new Error(`${label} is ${size} bytes, expected ${expected}`);
}

function parseTrainingState(value: unknown): TrainingState {
    if (!isRecord(value))
        throw new Error("training-state.json must be an object");
    const version = readFinite(value, "version", "checkpoint");
    if (version !== CHECKPOINT_VERSION)
        throw new Error(`unsupported checkpoint version ${version}`);
    const step = readFinite(value, "step", "checkpoint");
    assertNonNegativeInteger(step, "checkpoint step");
    const seed = readFinite(value, "seed", "checkpoint");
    if (!isRecord(value.model))
        throw new Error("checkpoint model config is missing");
    if (!isRecord(value.hyperparameters))
        throw new Error("checkpoint hyperparameters are missing");
    return {
        version,
        step,
        seed,
        model: readModelConfig(value.model),
        hyperparameters: readHyperparameters(value.hyperparameters),
        parameters: parseManifest(value.parameters),
    };
}

function readModelConfig(value: Record<string, unknown>): CheckpointModelConfig {
    return {
        vocab: readFinite(value, "vocab", "model"),
        nEmbd: readFinite(value, "nEmbd", "model"),
        nHead: readFinite(value, "nHead", "model"),
        nLayer: readFinite(value, "nLayer", "model"),
        maxSeq: readFinite(value, "maxSeq", "model"),
    };
}

function readHyperparameters(value: Record<string, unknown>): CheckpointHyperparameters {
    return {
        learningRate: readFinite(value, "learningRate", "hyperparameters"),
        beta1: readFinite(value, "beta1", "hyperparameters"),
        beta2: readFinite(value, "beta2", "hyperparameters"),
        eps: readFinite(value, "eps", "hyperparameters"),
        weightDecay: readFinite(value, "weightDecay", "hyperparameters"),
        batchSize: readFinite(value, "batchSize", "hyperparameters"),
        blockSize: readFinite(value, "blockSize", "hyperparameters"),
    };
}

function parseManifest(value: unknown): ParameterRecord[] {
    if (!Array.isArray(value))
        throw new Error("checkpoint parameters must be a list");
    const names = new Set<string>();
    return value.map((entry, index) => {
        if (!isRecord(entry))
            throw new Error(`checkpoint parameter ${index} is invalid`);
        const name = entry.name;
        if (typeof name !== "string" || name.length === 0)
            throw new Error(`checkpoint parameter ${index} has an invalid name`);
        if (names.has(name))
            throw new Error(`checkpoint parameter ${name} is duplicated`);
        names.add(name);
        if (!Array.isArray(entry.shape) || entry.shape.some(dim => typeof dim !== "number" || !Number.isInteger(dim) || dim < 0))
            throw new Error(`checkpoint parameter ${name} has an invalid shape`);
        const shape = [...entry.shape] as number[];
        const numel = entry.numel;
        if (typeof numel !== "number" || !Number.isInteger(numel) || numel < 0)
            throw new Error(`checkpoint parameter ${name} has an invalid numel`);
        const expected = shape.reduce((total, dim) => total * dim, 1);
        if (numel !== expected)
            throw new Error(`checkpoint parameter ${name} numel ${numel} does not match its shape`);
        const adamStep = entry.adamStep;
        if (typeof adamStep !== "number" || !Number.isInteger(adamStep) || adamStep < 0)
            throw new Error(`checkpoint parameter ${name} has an invalid Adam step`);
        return { name, shape, numel, adamStep };
    });
}

function assertSameNumbers(label: string, expected: object, actual: object): void {
    const expectedRecord = expected as Record<string, number>;
    const actualRecord = actual as Record<string, number>;
    for (const [key, value] of Object.entries(expectedRecord)) {
        if (actualRecord[key] !== value)
            throw new Error(`checkpoint ${label}.${key} is ${actualRecord[key]}, script has ${value}`);
    }
}

function readFinite(record: Record<string, unknown>, key: string, label: string): number {
    const value = record[key];
    if (typeof value !== "number" || !Number.isFinite(value))
        throw new Error(`${label}.${key} must be a finite number`);
    return value;
}

function sameShape(left: number[], right: number[]): boolean {
    return left.length === right.length && left.every((value, index) => value === right[index]);
}

function assertNonNegativeInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value < 0)
        throw new Error(`${label} must be a non-negative integer, received ${value}`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
