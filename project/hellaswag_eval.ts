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

export type TokenForward = {
    forward(ids: Tensor): Tensor;
};

export async function hellaswagAccuracy(
    model: TokenForward,
    examples: readonly HellaSwagExample[],
    blockSize: number,
): Promise<number> {
    if (examples.length === 0)
        throw new Error("HellaSwag evaluation set is empty");
    let correct = 0;
    for (let index = 0; index < examples.length; index++) {
        const prediction = await scoreExample(model, examples[index]!, blockSize);
        if (prediction === examples[index]!.label)
            correct++;
        if ((index + 1) % 1000 === 0 || index + 1 === examples.length)
            console.log(`hellaswag ${index + 1}/${examples.length}`);
    }
    return correct / examples.length;
}

async function scoreExample(model: TokenForward, example: HellaSwagExample, blockSize: number): Promise<number> {
    const fitted = example.endings.map((ending) => fitEnding(example.context, ending, blockSize));
    const length = Math.max(...fitted.map((row) => row.tokens.length));
    const time = length - 1;
    const input = new Float32Array(ENDING_COUNT * time);
    const target = new Float32Array(ENDING_COUNT * time);
    const masks: Uint8Array[] = [];
    for (let row = 0; row < ENDING_COUNT; row++) {
        const sequence = fitted[row]!;
        const padded = new Uint16Array(length);
        padded.set(sequence.tokens);
        for (let t = 0; t < time; t++) {
            input[row * time + t] = padded[t]!;
            target[row * time + t] = padded[t + 1]!;
        }
        masks.push(endingTargetMask(sequence.tokens.length, sequence.contextLength, time));
    }

    const nll = await tensorScope(() => {
        const logits = model.forward(idsToTensor(input, ENDING_COUNT, time));
        const losses = logsoftmax(logits, -1).gather(idsToTensor(target, ENDING_COUNT, time).view(ENDING_COUNT, time, 1), 2).neg();
        return losses.toFloat32().then((values) => Float32Array.from(values));
    });
    if (nll.length !== ENDING_COUNT * time)
        throw new Error(`HellaSwag loss has ${nll.length} values, expected ${ENDING_COUNT * time}`);

    const means: number[] = [];
    for (let row = 0; row < ENDING_COUNT; row++)
        means.push(maskedMean(nll.subarray(row * time, (row + 1) * time), masks[row]!));
    return chooseEnding(means);
}

function idsToTensor(values: Float32Array, batch: number, time: number): Tensor {
    const data = TensorData.empty([batch, time]);
    writeStorage(data._storage, values);
    return new Tensor(data);
}
