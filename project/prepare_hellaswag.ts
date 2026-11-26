import { readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { createGpt2Encoder, download, readParquetRows, SEPARATOR } from "./gpt2_token_io.js";
import {
    EXAMPLE_COUNT,
    SOURCE_BYTES,
    encodeExamples,
    validationBinPath,
    validationMetaPath,
    validationParquetPath,
    type HellaSwagExample,
    type HellaSwagMeta,
} from "./hellaswag_data.js";

const VALIDATION_URL = "https://huggingface.co/api/datasets/Rowan/hellaswag/parquet/default/validation/0.parquet";

export async function prepareHellaSwag(): Promise<void> {
    console.log("HellaSwag validation");
    await download(VALIDATION_URL, validationParquetPath, SOURCE_BYTES);
    await encodeValidation();
}

async function encodeValidation(): Promise<void> {
    const existing = await readMeta(validationMetaPath);
    const binStat = await stat(validationBinPath).catch(() => undefined);
    if (existing?.sourceBytes === SOURCE_BYTES && existing.encoding === "gpt2" && existing.examples === EXAMPLE_COUNT && binStat !== undefined) {
        console.log(`validation: ${existing.examples.toLocaleString()} examples already encoded`);
        return;
    }

    const encoder = createGpt2Encoder();
    const examples: HellaSwagExample[] = [];
    let row = 0;
    for await (const record of readParquetRows(validationParquetPath, ["ctx", "endings", "label"])) {
        const context = encoder.encode(readString(record.ctx, row, "ctx"), [SEPARATOR]);
        const endings = readEndings(record.endings, row).map((ending) => encoder.encode(` ${ending}`, [SEPARATOR]));
        if (endings.some((tokens) => tokens.length === 0))
            throw new Error(`HellaSwag row ${row} has an empty ending`);
        examples.push({
            label: readLabel(record.label, row),
            context: Uint16Array.from(context),
            endings: [
                Uint16Array.from(endings[0]!),
                Uint16Array.from(endings[1]!),
                Uint16Array.from(endings[2]!),
                Uint16Array.from(endings[3]!),
            ],
        });
        row++;
    }
    if (examples.length !== EXAMPLE_COUNT)
        throw new Error(`HellaSwag validation has ${examples.length} rows, expected ${EXAMPLE_COUNT}`);

    const tmp = `${validationBinPath}.tmp`;
    await rm(tmp, { force: true });
    try {
        await writeFile(tmp, encodeExamples(examples));
        await rm(validationBinPath, { force: true });
        await rename(tmp, validationBinPath);
        const meta: HellaSwagMeta = { sourceBytes: SOURCE_BYTES, examples: examples.length, encoding: "gpt2" };
        await writeFile(validationMetaPath, `${JSON.stringify(meta, null, 2)}\n`);
    }
    catch (error) {
        await rm(tmp, { force: true });
        throw error;
    }
    console.log(`validation: wrote ${examples.length.toLocaleString()} examples to ${validationBinPath}`);
}

function readString(value: unknown, row: number, field: string): string {
    if (typeof value !== "string")
        throw new Error(`HellaSwag row ${row} ${field} is ${describe(value)}, expected a string`);
    return value;
}

function readEndings(value: unknown, row: number): [string, string, string, string] {
    if (!Array.isArray(value) || value.length !== 4 || value.some((item) => typeof item !== "string"))
        throw new Error(`HellaSwag row ${row} endings are ${describe(value)}, expected four strings`);
    return value as [string, string, string, string];
}

function readLabel(value: unknown, row: number): number {
    const text = typeof value === "number" && Number.isInteger(value) ? String(value) : value;
    if (typeof text !== "string" || !/^[0-3]$/.test(text))
        throw new Error(`HellaSwag row ${row} label is ${describe(value)}, expected 0, 1, 2, or 3`);
    return Number(text);
}

function describe(value: unknown): string {
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (Array.isArray(value)) return `an array of ${value.length}`;
    if (typeof value === "object")
        return `an object with keys ${Object.keys(value).slice(0, 8).join(", ")}`;
    return typeof value;
}

async function readMeta(path: string): Promise<HellaSwagMeta | undefined> {
    try {
        return JSON.parse(await readFile(path, "utf8")) as HellaSwagMeta;
    }
    catch {
        return undefined;
    }
}

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("prepare_hellaswag")) {
    await prepareHellaSwag();
}
