import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";
import { dirname, join } from "node:path";
import { spawn } from "node:child_process";
import { Tiktoken } from "js-tiktoken/lite";
import gpt2 from "js-tiktoken/ranks/gpt2";
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import type { TokenSplit } from "./tinystories_data.js";

export const SEPARATOR = "<|endoftext|>";
export const EOT = 50256;

const ENCODE_CHARS = 512 * 1024;
const TOKEN_BUFFER = 1 << 20;
const COPY_BUFFER = 8 * 1024 * 1024;
const ROW_WINDOW = 4096;

export type ShardRecord = {
    name: string;
    bytes: number;
    tokens: number;
};

export type TokenMeta = {
    sourceBytes: number;
    tokens: number;
    encoding: "gpt2";
    eot: number;
    shards: ShardRecord[];
};

export type ShardSource = {
    split: TokenSplit;
    name: string;
    bytes: number;
    url: string;
    documents(parquetPath: string): AsyncIterable<string>;
};

export function createGpt2Encoder(): Tiktoken {
    const encoder = new Tiktoken(gpt2);
    const special = gpt2.special_tokens[SEPARATOR];
    const encoded = encoder.encode(SEPARATOR, [SEPARATOR]);
    if (special !== EOT || encoded.length !== 1 || encoded[0] !== EOT)
        throw new Error(`GPT-2 encoder did not map ${SEPARATOR} to ${EOT}`);
    return encoder;
}

function run(command: string, args: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { stdio: "inherit" });
        child.on("error", reject);
        child.on("exit", (code) => {
            if (code === 0) resolve();
            else reject(new Error(`${command} exited with code ${code}`));
        });
    });
}

export async function download(url: string, destination: string, expectedBytes: number): Promise<void> {
    await mkdir(dirname(destination), { recursive: true });
    const current = await stat(destination).catch(() => undefined);
    if (current?.size === expectedBytes) {
        console.log(`already downloaded ${destination}`);
        return;
    }
    if (current && current.size > expectedBytes)
        await rm(destination);

    console.log(`downloading ${url}`);
    await run("curl.exe", [
        "-L",
        "--ssl-no-revoke",
        "--fail",
        "--retry", "8",
        "--retry-all-errors",
        "--retry-delay", "2",
        "-C", "-",
        "-o", destination,
        url,
    ]);

    const after = await stat(destination);
    if (after.size !== expectedBytes)
        throw new Error(`expected ${expectedBytes} bytes at ${destination}, received ${after.size}`);
}

class TokenWriter {
    tokens = 0;
    private count = 0;
    private position = 0;
    private readonly buffer = new Uint16Array(TOKEN_BUFFER);

    constructor(private readonly file: FileHandle) {}

    async push(ids: number[]): Promise<void> {
        for (let i = 0; i < ids.length; i++) {
            const id = ids[i]!;
            if (!Number.isInteger(id) || id < 0 || id > EOT)
                throw new Error(`token id ${id} is outside the GPT-2 vocabulary`);
            this.buffer[this.count++] = id;
            if (this.count === this.buffer.length)
                await this.flush();
        }
    }

    async finish(): Promise<void> {
        await this.flush();
        await this.file.close();
    }

    private async flush(): Promise<void> {
        if (this.count === 0) return;
        const view = new Uint8Array(this.buffer.buffer, 0, this.count * 2);
        let written = 0;
        while (written < view.length) {
            const { bytesWritten } = await this.file.write(view, written, view.length - written, this.position);
            if (bytesWritten === 0)
                throw new Error("failed to write token file");
            written += bytesWritten;
            this.position += bytesWritten;
        }
        this.tokens += this.count;
        this.count = 0;
    }
}

function shardBinPath(directory: string, name: string): string {
    return join(directory, "shards", `${name}.bin`);
}

function shardMetaPath(directory: string, name: string): string {
    return join(directory, "shards", `${name}.json`);
}

function isShardRecord(value: unknown): value is ShardRecord {
    if (typeof value !== "object" || value === null) return false;
    const shard = value as ShardRecord;
    return typeof shard.name === "string" && shard.name.length > 0
        && Number.isInteger(shard.bytes) && shard.bytes >= 0
        && Number.isInteger(shard.tokens) && shard.tokens >= 0;
}

function isTokenMeta(value: unknown): value is TokenMeta {
    if (typeof value !== "object" || value === null) return false;
    const meta = value as TokenMeta;
    return meta.encoding === "gpt2"
        && meta.eot === EOT
        && Number.isInteger(meta.sourceBytes) && meta.sourceBytes >= 0
        && Number.isInteger(meta.tokens) && meta.tokens >= 0
        && Array.isArray(meta.shards) && meta.shards.every(isShardRecord);
}

async function readTokenMeta(path: string): Promise<TokenMeta | undefined> {
    try {
        const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
        return isTokenMeta(parsed) ? parsed : undefined;
    }
    catch {
        return undefined;
    }
}

async function writeTokenMeta(path: string, meta: TokenMeta): Promise<void> {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(meta, null, 2));
}

function sameShards(left: ShardRecord[], right: ShardRecord[]): boolean {
    if (left.length !== right.length) return false;
    for (let i = 0; i < left.length; i++) {
        const a = left[i]!;
        const b = right[i]!;
        if (a.name !== b.name || a.bytes !== b.bytes || a.tokens !== b.tokens) return false;
    }
    return true;
}

async function preview(encoder: Tiktoken, bin: string): Promise<void> {
    const file = await open(bin, "r");
    try {
        const bytes = new Uint8Array(64);
        const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
        const ids = Array.from(new Uint16Array(bytes.buffer, 0, Math.floor(bytesRead / 2)));
        console.log(`${bin} starts: ${JSON.stringify(encoder.decode(ids).slice(0, 180))}`);
    }
    finally {
        await file.close();
    }
}

async function encodeShard(options: {
    encoder: Tiktoken;
    documents: AsyncIterable<string>;
    binPath: string;
    metaPath: string;
    name: string;
    sourceBytes: number;
    label: string;
}): Promise<ShardRecord> {
    const existing = await readTokenMeta(options.metaPath);
    const binStat = await stat(options.binPath).catch(() => undefined);
    const shard = existing?.shards[0];
    if (existing && existing.sourceBytes === options.sourceBytes && existing.tokens > 0
        && existing.shards.length === 1 && shard?.name === options.name && shard.bytes === options.sourceBytes
        && shard.tokens === existing.tokens && binStat?.size === existing.tokens * 2) {
        console.log(`${options.label}: ${existing.tokens.toLocaleString()} tokens already encoded`);
        return shard;
    }

    const tmp = `${options.binPath}.tmp`;
    await mkdir(dirname(options.binPath), { recursive: true });
    await rm(tmp, { force: true });
    const writer = new TokenWriter(await open(tmp, "w"));
    const pieces: string[] = [];
    let pieceChars = 0;
    let documents = 0;
    let closed = false;

    // Each piece ends with the special separator, so byte-pair encoding cannot merge across documents.
    const flushPieces = async (): Promise<void> => {
        if (pieceChars === 0) return;
        await writer.push(options.encoder.encode(pieces.join(""), [SEPARATOR]));
        pieces.length = 0;
        pieceChars = 0;
    };

    try {
        console.log(`${options.label}: encoding`);
        for await (const document of options.documents) {
            if (document.length === 0) continue;
            const piece = document.endsWith(SEPARATOR) ? document : document + SEPARATOR;
            pieces.push(piece);
            pieceChars += piece.length;
            documents++;
            if (pieceChars >= ENCODE_CHARS)
                await flushPieces();
            if (documents % 50_000 === 0) {
                console.log(`${options.label}: documents ${documents.toLocaleString()}  tokens ${writer.tokens.toLocaleString()}`);
            }
        }
        await flushPieces();
        await writer.finish();
        closed = true;
        if (writer.tokens < 1)
            throw new Error(`${options.label} produced no tokens`);
    }
    catch (error) {
        if (!closed)
            await writer.finish().catch(() => undefined);
        await rm(tmp, { force: true });
        throw error;
    }

    await rm(options.binPath, { force: true });
    await rename(tmp, options.binPath);
    const record: ShardRecord = { name: options.name, bytes: options.sourceBytes, tokens: writer.tokens };
    const meta: TokenMeta = {
        sourceBytes: options.sourceBytes,
        tokens: writer.tokens,
        encoding: "gpt2",
        eot: EOT,
        shards: [record],
    };
    await writeTokenMeta(options.metaPath, meta);
    console.log(`${options.label}: wrote ${writer.tokens.toLocaleString()} tokens to ${options.binPath}`);
    return record;
}

async function writeAt(file: FileHandle, buffer: Uint8Array, length: number, position: number): Promise<void> {
    let written = 0;
    while (written < length) {
        const { bytesWritten } = await file.write(buffer, written, length - written, position + written);
        if (bytesWritten === 0)
            throw new Error("failed to write token file");
        written += bytesWritten;
    }
}

async function concatShards(options: {
    encoder: Tiktoken;
    directory: string;
    split: TokenSplit;
    shards: ShardRecord[];
}): Promise<void> {
    if (options.shards.length === 0)
        throw new Error(`no ${options.split} shards to concatenate`);
    let sourceBytes = 0;
    let tokens = 0;
    for (const shard of options.shards) {
        sourceBytes += shard.bytes;
        tokens += shard.tokens;
    }
    const bin = join(options.directory, `${options.split}.bin`);
    const metaPath = join(options.directory, `${options.split}.json`);
    const existing = await readTokenMeta(metaPath);
    const binStat = await stat(bin).catch(() => undefined);
    if (existing && existing.sourceBytes === sourceBytes && existing.tokens === tokens
        && sameShards(existing.shards, options.shards) && binStat?.size === tokens * 2) {
        console.log(`${options.split}: ${tokens.toLocaleString()} tokens already encoded`);
        return;
    }

    const tmp = `${bin}.tmp`;
    await rm(tmp, { force: true });
    const output = await open(tmp, "w");
    let position = 0;
    try {
        console.log(`${options.split}: concatenating ${options.shards.length} shards`);
        for (const shard of options.shards) {
            const shardBin = shardBinPath(options.directory, shard.name);
            const info = await stat(shardBin);
            const expected = shard.tokens * 2;
            if (info.size !== expected)
                throw new Error(`${shardBin} is ${info.size} bytes, expected ${expected}`);
            console.log(`${options.split}: copying ${shard.name} (${shard.tokens.toLocaleString()} tokens)`);
            const input = await open(shardBin, "r");
            try {
                const chunk = new Uint8Array(COPY_BUFFER);
                let offset = 0;
                while (offset < info.size) {
                    const length = Math.min(chunk.length, info.size - offset);
                    const { bytesRead } = await input.read(chunk, 0, length, offset);
                    if (bytesRead === 0)
                        throw new Error(`${shardBin} ended after ${offset} of ${info.size} bytes`);
                    await writeAt(output, chunk, bytesRead, position);
                    offset += bytesRead;
                    position += bytesRead;
                }
            }
            finally {
                await input.close();
            }
        }
        if (position !== tokens * 2)
            throw new Error(`${options.split} concatenated ${position} bytes, expected ${tokens * 2}`);
    }
    catch (error) {
        await output.close().catch(() => undefined);
        await rm(tmp, { force: true });
        throw error;
    }
    await output.close();
    await rm(bin, { force: true });
    await rename(tmp, bin);
    await writeTokenMeta(metaPath, {
        sourceBytes,
        tokens,
        encoding: "gpt2",
        eot: EOT,
        shards: options.shards,
    });
    console.log(`${options.split}: wrote ${tokens.toLocaleString()} tokens to ${bin}`);
    await preview(options.encoder, bin);
}

export async function* readParquetRows(path: string, columns: string[]): AsyncGenerator<Record<string, unknown>> {
    const file = await asyncBufferFromFile(path);
    const metadata = await parquetMetadataAsync(file);
    const numRows = Number(metadata.num_rows);
    if (!Number.isSafeInteger(numRows) || numRows < 0)
        throw new Error(`${path} has an invalid row count`);
    console.log(`${path}: ${numRows.toLocaleString()} rows`);
    for (let rowStart = 0; rowStart < numRows; rowStart += ROW_WINDOW) {
        const rowEnd = Math.min(rowStart + ROW_WINDOW, numRows);
        const rows = await parquetReadObjects({
            file,
            metadata,
            columns,
            rowStart,
            rowEnd,
            compressors,
        });
        if (rows.length !== rowEnd - rowStart)
            throw new Error(`${path} returned ${rows.length} rows for ${rowStart}:${rowEnd}`);
        for (const row of rows) yield row;
    }
}

export async function installShards(directory: string, sources: ShardSource[]): Promise<void> {
    const encoder = createGpt2Encoder();
    await mkdir(directory, { recursive: true });
    const seen = new Set<string>();
    const grouped = new Map<TokenSplit, ShardRecord[]>();
    for (const source of sources) {
        if (!source.name.endsWith(".parquet"))
            throw new Error(`${source.name} is not a parquet file`);
        if (seen.has(source.name))
            throw new Error(`duplicate shard ${source.name}`);
        seen.add(source.name);
        const parquetPath = join(directory, "parquet", source.name);
        await download(source.url, parquetPath, source.bytes);
        const stem = source.name.slice(0, -".parquet".length);
        const record = await encodeShard({
            encoder,
            documents: source.documents(parquetPath),
            binPath: shardBinPath(directory, stem),
            metaPath: shardMetaPath(directory, stem),
            name: stem,
            sourceBytes: source.bytes,
            label: `${source.split} ${stem}`,
        });
        const list = grouped.get(source.split) ?? [];
        list.push(record);
        grouped.set(source.split, list);
    }
    for (const split of ["train", "valid"] as const) {
        const shards = grouped.get(split);
        if (shards === undefined || shards.length === 0)
            throw new Error(`no ${split} shards`);
        await concatShards({ encoder, directory, split, shards });
    }
}
