import { installShards, readParquetRows, type ShardSource } from "./gpt2_token_io.js";
import { dataDirectory } from "./smol_smoltalk_data.js";
import type { TokenSplit } from "./tinystories_data.js";

const ROOT = "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data";

// The official test split is written as valid.bin so tokenPath("valid") matches the trainer.
const SHARDS: { split: TokenSplit; name: string; bytes: number }[] = [
    { split: "train", name: "train-00000-of-00004.parquet", bytes: 229_710_350 },
    { split: "train", name: "train-00001-of-00004.parquet", bytes: 230_298_184 },
    { split: "train", name: "train-00002-of-00004.parquet", bytes: 230_587_255 },
    { split: "train", name: "train-00003-of-00004.parquet", bytes: 231_879_938 },
    { split: "valid", name: "test-00000-of-00001.parquet", bytes: 48_177_022 },
];

const SOURCES: ShardSource[] = SHARDS.map((shard) => ({
    split: shard.split,
    name: shard.name,
    bytes: shard.bytes,
    url: `${ROOT}/${shard.name}`,
    documents: conversationDocuments,
}));

function describe(value: unknown): string {
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (Array.isArray(value)) return `an array of ${value.length}`;
    if (typeof value === "object")
        return `an object with keys ${Object.keys(value).slice(0, 8).join(", ")}`;
    return typeof value;
}

// One conversation becomes role/content turns. installShards appends <|endoftext|> after each document.
function renderConversation(messages: unknown, path: string): string | undefined {
    if (messages === null || messages === undefined) return undefined;
    if (!Array.isArray(messages))
        throw new Error(`${path} messages is ${describe(messages)}, expected a list of {role, content}`);
    const turns: string[] = [];
    for (let i = 0; i < messages.length; i++) {
        const message: unknown = messages[i];
        if (typeof message !== "object" || message === null)
            throw new Error(`${path} message ${i} is ${describe(message)}, expected {role, content}`);
        const record = message as Record<string, unknown>;
        const role = record.role;
        const content = record.content;
        if (typeof role !== "string" || typeof content !== "string")
            throw new Error(`${path} message ${i} is ${describe(message)}, expected {role, content}`);
        if (role.length === 0 && content.length === 0) continue;
        turns.push(`${role}\n${content}`);
    }
    if (turns.length === 0) return undefined;
    return turns.join("\n");
}

async function* conversationDocuments(path: string): AsyncGenerator<string> {
    for await (const row of readParquetRows(path, ["messages"])) {
        const text = renderConversation(row.messages, path);
        if (text !== undefined) yield text;
    }
}

export async function prepareSmolSmoltalk(): Promise<void> {
    console.log("Smol-SmolTalk");
    await installShards(dataDirectory, SOURCES);
}

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("prepare_smol_smoltalk")) {
    await prepareSmolSmoltalk();
}
