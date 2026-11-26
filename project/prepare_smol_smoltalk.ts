import { installShards, type ShardSource } from "./gpt2_token_io.js";
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
    format: "chat-messages",
}));

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
