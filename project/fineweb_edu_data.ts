import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { TokenSplit } from "./tinystories_data.js";

export const dataDirectory = fileURLToPath(new URL("./data/fineweb-edu/", import.meta.url));

export { loadTokenFile, Random, sampleBatch } from "./tinystories_data.js";
export type { TokenBatch, TokenSplit } from "./tinystories_data.js";

export function tokenPath(split: TokenSplit): string {
    return join(dataDirectory, `${split}.bin`);
}
