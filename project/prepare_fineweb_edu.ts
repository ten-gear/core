import { installShards, readParquetRows, type ShardSource } from "./gpt2_token_io.js";
import { dataDirectory } from "./fineweb_edu_data.js";
import type { TokenSplit } from "./tinystories_data.js";

const ROOT = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT";

// sample-10BT has no validation split. The last shard is a slice of the same random sample,
// so it is held out as valid.bin. Shards 000 through 012 are train.
const SHARDS: { split: TokenSplit; name: string; bytes: number }[] = [
    { split: "train", name: "000_00000.parquet", bytes: 2_152_819_114 },
    { split: "train", name: "001_00000.parquet", bytes: 2_152_222_432 },
    { split: "train", name: "002_00000.parquet", bytes: 2_151_796_315 },
    { split: "train", name: "003_00000.parquet", bytes: 2_152_437_524 },
    { split: "train", name: "004_00000.parquet", bytes: 2_152_338_550 },
    { split: "train", name: "005_00000.parquet", bytes: 2_152_189_947 },
    { split: "train", name: "006_00000.parquet", bytes: 2_152_689_867 },
    { split: "train", name: "007_00000.parquet", bytes: 2_150_686_637 },
    { split: "train", name: "008_00000.parquet", bytes: 2_151_274_846 },
    { split: "train", name: "009_00000.parquet", bytes: 2_151_913_277 },
    { split: "train", name: "010_00000.parquet", bytes: 2_152_798_864 },
    { split: "train", name: "011_00000.parquet", bytes: 2_152_323_681 },
    { split: "train", name: "012_00000.parquet", bytes: 2_152_069_689 },
    { split: "valid", name: "013_00000.parquet", bytes: 540_632_672 },
];

const SOURCES: ShardSource[] = SHARDS.map((shard) => ({
    split: shard.split,
    name: shard.name,
    bytes: shard.bytes,
    url: `${ROOT}/${shard.name}`,
    documents: finewebDocuments,
}));

async function* finewebDocuments(path: string): AsyncGenerator<string> {
    for await (const row of readParquetRows(path, ["text"])) {
        const text = row.text;
        if (text === null || text === undefined || text === "") continue;
        if (typeof text !== "string")
            throw new Error(`${path} text column is ${typeof text}, expected a string`);
        yield text;
    }
}

export async function prepareFinewebEdu(): Promise<void> {
    console.log("FineWeb-Edu sample-10BT");
    await installShards(dataDirectory, SOURCES);
}

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("prepare_fineweb_edu")) {
    await prepareFinewebEdu();
}
