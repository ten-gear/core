import { open } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const dataDirectory = fileURLToPath(new URL("./data/tinystories/", import.meta.url));

export type TokenSplit = "train" | "valid";

export type TokenBatch = {
    input: Float32Array;
    target: Float32Array;
};

export function tokenPath(split: TokenSplit): string {
    return join(dataDirectory, `${split}.bin`);
}

export async function loadTokenFile(path: string): Promise<Uint16Array> {
    const file = await open(path, "r");
    try {
        const { size } = await file.stat();
        if (size === 0 || size % 2 !== 0)
            throw new Error(`${path} is not a flat uint16 token file`);
        const tokens = new Uint16Array(size / 2);
        const bytes = new Uint8Array(tokens.buffer);
        const chunk = 64 * 1024 * 1024;
        let offset = 0;
        while (offset < bytes.length) {
            const length = Math.min(chunk, bytes.length - offset);
            const { bytesRead } = await file.read(bytes, offset, length, offset);
            if (bytesRead === 0)
                throw new Error(`${path} ended after ${offset} of ${bytes.length} bytes`);
            offset += bytesRead;
        }
        return tokens;
    }
    finally {
        await file.close();
    }
}

// One training row is blockSize + 1 consecutive ids:
// input = ids[start : start + blockSize], target = ids[start + 1 : start + blockSize + 1].
export function copyTokenWindow(
    tokens: Uint16Array,
    start: number,
    blockSize: number,
    input: Float32Array,
    target: Float32Array,
    row = 0,
): void {
    if (start < 0 || start + blockSize + 1 > tokens.length)
        throw new Error(`token window ${start} is outside a file of ${tokens.length} tokens`);
    const offset = row * blockSize;
    for (let t = 0; t < blockSize; t++) {
        input[offset + t] = tokens[start + t]!;
        target[offset + t] = tokens[start + t + 1]!;
    }
}

export function sampleBatch(tokens: Uint16Array, batchSize: number, blockSize: number): TokenBatch {
    if (!Number.isInteger(batchSize) || batchSize < 1)
        throw new Error("batchSize must be a positive integer");
    if (!Number.isInteger(blockSize) || blockSize < 1)
        throw new Error("blockSize must be a positive integer");
    const maxStart = tokens.length - blockSize - 1;
    if (maxStart < 0)
        throw new Error(`need at least ${blockSize + 1} tokens, found ${tokens.length}`);

    const input = new Float32Array(batchSize * blockSize);
    const target = new Float32Array(batchSize * blockSize);
    for (let row = 0; row < batchSize; row++) {
        const start = Math.floor(Math.random() * (maxStart + 1));
        copyTokenWindow(tokens, start, blockSize, input, target, row);
    }
    return { input, target };
}
