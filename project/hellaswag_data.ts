import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const dataDirectory = fileURLToPath(new URL("./data/hellaswag/", import.meta.url));
export const SOURCE_BYTES = 6_315_951;
export const EXAMPLE_COUNT = 10_042;
export const ENDING_COUNT = 4;

export const validationParquetPath = join(dataDirectory, "validation.parquet");
export const validationBinPath = join(dataDirectory, "validation.bin");
export const validationMetaPath = join(dataDirectory, "validation.json");

export type HellaSwagMeta = {
    sourceBytes: number;
    examples: number;
    encoding: "gpt2";
};

export type HellaSwagExample = {
    label: number;
    context: Uint16Array;
    endings: [Uint16Array, Uint16Array, Uint16Array, Uint16Array];
};

export type FittedEnding = {
    tokens: Uint16Array;
    contextLength: number;
};

export function encodeExamples(examples: readonly HellaSwagExample[]): Buffer {
    let bytes = 0;
    for (const example of examples)
        bytes += recordBytes(example);
    const buffer = Buffer.allocUnsafe(bytes);
    let offset = 0;
    for (const example of examples)
        offset = writeExample(buffer, offset, example);
    if (offset !== buffer.length)
        throw new Error("HellaSwag encoding did not fill the buffer");
    return buffer;
}

export function decodeExamples(bytes: Buffer): HellaSwagExample[] {
    const examples: HellaSwagExample[] = [];
    let offset = 0;
    while (offset < bytes.length) {
        const example = readExample(bytes, offset);
        examples.push(example.example);
        offset = example.offset;
    }
    return examples;
}

export function fitEnding(context: Uint16Array, ending: Uint16Array, blockSize: number): FittedEnding {
    if (!Number.isInteger(blockSize) || blockSize < 2)
        throw new Error("blockSize must be at least 2");
    if (ending.length === 0)
        throw new Error("HellaSwag ending is empty");
    let contextStart = 0;
    let endingStart = 0;
    const overflow = context.length + ending.length - blockSize;
    if (overflow > 0) {
        const dropContext = Math.min(overflow, context.length);
        contextStart = dropContext;
        const rest = overflow - dropContext;
        if (rest > 0)
            endingStart = rest;
    }
    const contextLength = context.length - contextStart;
    const endingLength = ending.length - endingStart;
    const tokens = new Uint16Array(contextLength + endingLength);
    tokens.set(context.subarray(contextStart), 0);
    tokens.set(ending.subarray(endingStart), contextLength);
    if (tokens.length < 2)
        throw new Error("HellaSwag sequence is shorter than 2 tokens");
    return { tokens, contextLength };
}

// Target index t predicts original token t + 1. Ending tokens start at contextLength.
export function endingTargetMask(tokenLength: number, contextLength: number, time: number): Uint8Array {
    const mask = new Uint8Array(time);
    for (let t = 0; t < time; t++) {
        const original = t + 1;
        if (original >= contextLength && original < tokenLength)
            mask[t] = 1;
    }
    return mask;
}

export function maskedMean(nll: ArrayLike<number>, mask: ArrayLike<number>): number {
    let sum = 0;
    let count = 0;
    const length = Math.min(nll.length, mask.length);
    for (let i = 0; i < length; i++) {
        if (mask[i] === 0) continue;
        const value = nll[i]!;
        if (!Number.isFinite(value))
            throw new Error("HellaSwag loss is not finite");
        sum += value;
        count++;
    }
    if (count === 0)
        throw new Error("HellaSwag ending has no scored tokens");
    return sum / count;
}

export function chooseEnding(meanLosses: readonly number[]): number {
    if (meanLosses.length !== ENDING_COUNT)
        throw new Error(`HellaSwag expects ${ENDING_COUNT} endings, received ${meanLosses.length}`);
    let best = 0;
    for (let i = 1; i < meanLosses.length; i++) {
        const loss = meanLosses[i]!;
        if (!Number.isFinite(loss))
            throw new Error("HellaSwag loss is not finite");
        if (loss < meanLosses[best]!)
            best = i;
    }
    if (!Number.isFinite(meanLosses[0]!))
        throw new Error("HellaSwag loss is not finite");
    return best;
}

export async function hellaswagIsEncoded(): Promise<boolean> {
    try {
        const meta = JSON.parse(await readFile(validationMetaPath, "utf8")) as HellaSwagMeta;
        if (meta.encoding !== "gpt2" || meta.sourceBytes !== SOURCE_BYTES || meta.examples !== EXAMPLE_COUNT)
            return false;
        const info = await stat(validationBinPath);
        return info.size > 0;
    }
    catch {
        return false;
    }
}

export async function loadHellaSwagExamples(): Promise<HellaSwagExample[]> {
    const meta = JSON.parse(await readFile(validationMetaPath, "utf8")) as HellaSwagMeta;
    if (meta.encoding !== "gpt2" || meta.sourceBytes !== SOURCE_BYTES || meta.examples !== EXAMPLE_COUNT)
        throw new Error("HellaSwag validation metadata does not match the pinned split");
    const examples = decodeExamples(await readFile(validationBinPath));
    if (examples.length !== EXAMPLE_COUNT)
        throw new Error(`HellaSwag validation has ${examples.length} examples, expected ${EXAMPLE_COUNT}`);
    return examples;
}

function recordBytes(example: HellaSwagExample): number {
    let bytes = 1 + 2 + example.context.length * 2;
    for (const ending of example.endings)
        bytes += 2 + ending.length * 2;
    return bytes;
}

function writeExample(buffer: Buffer, offset: number, example: HellaSwagExample): number {
    if (!Number.isInteger(example.label) || example.label < 0 || example.label >= ENDING_COUNT)
        throw new Error(`HellaSwag label ${example.label} is outside 0..${ENDING_COUNT - 1}`);
    buffer.writeUInt8(example.label, offset);
    offset += 1;
    offset = writeTokens(buffer, offset, example.context);
    for (const ending of example.endings)
        offset = writeTokens(buffer, offset, ending);
    return offset;
}

function writeTokens(buffer: Buffer, offset: number, tokens: Uint16Array): number {
    if (tokens.length > 0xffff)
        throw new Error(`HellaSwag token list is longer than ${0xffff}`);
    buffer.writeUInt16LE(tokens.length, offset);
    offset += 2;
    for (let i = 0; i < tokens.length; i++) {
        const id = tokens[i]!;
        if (!Number.isInteger(id) || id < 0 || id > 0xffff)
            throw new Error(`HellaSwag token id ${id} does not fit in uint16`);
        buffer.writeUInt16LE(id, offset);
        offset += 2;
    }
    return offset;
}

function readExample(bytes: Buffer, offset: number): { example: HellaSwagExample; offset: number } {
    const label = readU8(bytes, offset);
    offset += 1;
    if (label >= ENDING_COUNT)
        throw new Error(`HellaSwag label ${label} is outside 0..${ENDING_COUNT - 1}`);
    const context = readTokens(bytes, offset);
    offset = context.offset;
    const endings: Uint16Array[] = [];
    for (let i = 0; i < ENDING_COUNT; i++) {
        const ending = readTokens(bytes, offset);
        endings.push(ending.tokens);
        offset = ending.offset;
    }
    return {
        example: {
            label,
            context: context.tokens,
            endings: endings as HellaSwagExample["endings"],
        },
        offset,
    };
}

function readTokens(bytes: Buffer, offset: number): { tokens: Uint16Array; offset: number } {
    const length = readU16(bytes, offset);
    offset += 2;
    const tokens = new Uint16Array(length);
    for (let i = 0; i < length; i++) {
        tokens[i] = readU16(bytes, offset);
        offset += 2;
    }
    return { tokens, offset };
}

function readU8(bytes: Buffer, offset: number): number {
    if (offset + 1 > bytes.length)
        throw new Error("HellaSwag validation ended early");
    return bytes.readUInt8(offset);
}

function readU16(bytes: Buffer, offset: number): number {
    if (offset + 2 > bytes.length)
        throw new Error("HellaSwag validation ended early");
    return bytes.readUInt16LE(offset);
}
