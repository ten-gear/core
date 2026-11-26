import type { Parameter } from "../src/module.js";
import { Tensor, tensorScope } from "../src/tensor.js";
import { logsoftmax } from "../src/tensor_functions.js";
import { TensorData, writeStorage } from "../src/tensor_data.js";
import {
    ENDING_COUNT,
    chooseEnding,
    endingTargetMask,
    fitEnding,
    maskedMean,
    type HellaSwagExample,
} from "./hellaswag_data.js";

// One training step is 4 sequences of 1024 tokens. Keep each eval forward in that range.
export const HELLASWAG_POSITIONS_PER_FORWARD = 4 * 1024;

export type HellaSwagModel = {
    parameters(): Parameter[];
    hidden(ids: Tensor): Tensor;
    head: { forward(input: Tensor): Tensor };
};

export async function hellaswagAccuracy(
    model: HellaSwagModel,
    examples: readonly HellaSwagExample[],
    blockSize: number,
): Promise<number> {
    const losses = await hellaswagEndingLosses(model, examples, blockSize);
    let correct = 0;
    for (let index = 0; index < examples.length; index++) {
        if (chooseEnding(losses[index]!) === examples[index]!.label)
            correct++;
    }
    return correct / examples.length;
}

export async function hellaswagEndingLosses(
    model: HellaSwagModel,
    examples: readonly HellaSwagExample[],
    blockSize: number,
    positionsPerForward = HELLASWAG_POSITIONS_PER_FORWARD,
): Promise<number[][]> {
    if (examples.length === 0)
        throw new Error("HellaSwag evaluation set is empty");
    const groups = hellaswagForwardGroups(examples, blockSize, positionsPerForward);
    return withDetachedParameters(model, async () => {
        const losses: number[][] = [];
        let completed = 0;
        for (const group of groups) {
            const before = completed;
            losses.push(...await scoreGroup(model, group, blockSize));
            completed += group.length;
            logHellaswagProgress(before, completed, examples.length);
        }
        return losses;
    });
}

export function hellaswagForwardGroups(
    examples: readonly HellaSwagExample[],
    blockSize: number,
    positionsPerForward = HELLASWAG_POSITIONS_PER_FORWARD,
): HellaSwagExample[][] {
    if (!Number.isInteger(positionsPerForward) || positionsPerForward < ENDING_COUNT)
        throw new Error(`HellaSwag positions per forward must be at least ${ENDING_COUNT}, received ${positionsPerForward}`);
    const groups: HellaSwagExample[][] = [];
    let group: HellaSwagExample[] = [];
    let maxLength = 2;
    for (const example of examples) {
        const length = fittedLength(example, blockSize);
        const nextLength = group.length === 0 ? length : Math.max(maxLength, length);
        const positions = (group.length + 1) * ENDING_COUNT * (nextLength - 1);
        if (group.length > 0 && positions > positionsPerForward) {
            groups.push(group);
            group = [example];
            maxLength = length;
            continue;
        }
        group.push(example);
        maxLength = nextLength;
    }
    if (group.length > 0)
        groups.push(group);
    return groups;
}

function fittedLength(example: HellaSwagExample, blockSize: number): number {
    let length = 2;
    for (const ending of example.endings)
        length = Math.max(length, fitEnding(example.context, ending, blockSize).tokens.length);
    return length;
}

async function withDetachedParameters<T>(model: HellaSwagModel, run: () => Promise<T>): Promise<T> {
    const parameters = model.parameters();
    const original = parameters.map((parameter) => parameter.value);
    for (const parameter of parameters)
        parameter.update((parameter.value as Tensor).detach());
    try {
        return await run();
    } finally {
        for (let index = 0; index < parameters.length; index++)
            parameters[index]!.update(original[index]);
    }
}

function logHellaswagProgress(before: number, completed: number, total: number): void {
    const previousMark = Math.floor(before / 1000);
    const nextMark = Math.floor(completed / 1000);
    for (let mark = previousMark + 1; mark <= nextMark; mark++)
        console.log(`hellaswag ${mark * 1000}/${total}`);
    if (completed === total && completed % 1000 !== 0)
        console.log(`hellaswag ${completed}/${total}`);
}

async function scoreGroup(
    model: HellaSwagModel,
    examples: readonly HellaSwagExample[],
    blockSize: number,
): Promise<number[][]> {
    const fitted = examples.map((example) => example.endings.map((ending) => fitEnding(example.context, ending, blockSize)));
    let length = 2;
    for (const rows of fitted)
        for (const row of rows)
            length = Math.max(length, row.tokens.length);
    const time = length - 1;
    const batch = examples.length * ENDING_COUNT;
    const input = new Float32Array(batch * time);
    const positions: number[] = [];
    const targets: number[] = [];
    const counts: number[] = [];

    for (let exampleIndex = 0; exampleIndex < fitted.length; exampleIndex++) {
        const rows = fitted[exampleIndex]!;
        for (let ending = 0; ending < ENDING_COUNT; ending++) {
            const sequence = rows[ending]!;
            const row = exampleIndex * ENDING_COUNT + ending;
            for (let t = 0; t < sequence.tokens.length - 1; t++)
                input[row * time + t] = sequence.tokens[t]!;
            const mask = endingTargetMask(sequence.tokens.length, sequence.contextLength, time);
            let count = 0;
            for (let t = 0; t < time; t++) {
                if (mask[t] === 0) continue;
                positions.push(row * time + t);
                targets.push(sequence.tokens[t + 1]!);
                count++;
            }
            if (count === 0)
                throw new Error("HellaSwag ending has no scored tokens");
            counts.push(count);
        }
    }

    const nll = await tensorScope(() => {
        const hidden = model.hidden(idsToTensor(input, [batch, time])).tocontiguous();
        const shape = hidden.shape();
        if (shape.length !== 3 || shape[0] !== batch || shape[1] !== time)
            throw new Error(`HellaSwag hidden state has shape ${shape.join("x")}, expected ${batch}x${time}xwidth`);
        const picked = hidden.view(batch * time, shape[2]!).embedding(idsToTensor(Float32Array.from(positions), [positions.length]));
        const logits = model.head.forward(picked);
        const losses = logsoftmax(logits, -1).gather(idsToTensor(Float32Array.from(targets), [targets.length]).view(targets.length, 1), 1).neg();
        return losses.toFloat32().then((values) => Float32Array.from(values));
    });
    if (nll.length !== positions.length)
        throw new Error(`HellaSwag loss has ${nll.length} values, expected ${positions.length}`);

    const means: number[][] = [];
    let offset = 0;
    for (let exampleIndex = 0; exampleIndex < examples.length; exampleIndex++) {
        const rowMeans: number[] = [];
        for (let ending = 0; ending < ENDING_COUNT; ending++) {
            const count = counts[exampleIndex * ENDING_COUNT + ending]!;
            const scored = new Uint8Array(count);
            scored.fill(1);
            rowMeans.push(maskedMean(nll.subarray(offset, offset + count), scored));
            offset += count;
        }
        means.push(rowMeans);
    }
    if (offset !== nll.length)
        throw new Error(`HellaSwag scored ${offset} tokens, expected ${nll.length}`);
    return means;
}

function idsToTensor(values: Float32Array, shape: number[]): Tensor {
    const data = TensorData.empty(shape);
    writeStorage(data._storage, values);
    return new Tensor(data);
}
