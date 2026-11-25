import { createReadStream } from "node:fs";
import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { spawn } from "node:child_process";
import { Tiktoken } from "js-tiktoken/lite";
import gpt2 from "js-tiktoken/ranks/gpt2";
import { dataDirectory, tokenPath, type TokenSplit } from "./tinystories_data.js";

const SEPARATOR = "<|endoftext|>";
const EOT = 50256;
const ENCODE_CHARS = 512 * 1024;
const TOKEN_BUFFER = 1 << 20;

const SOURCES: { split: TokenSplit; name: string; url: string; bytes: number }[] = [
    {
        split: "valid",
        name: "TinyStoriesV2-GPT4-valid.txt",
        url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        bytes: 22_502_601,
    },
    {
        split: "train",
        name: "TinyStoriesV2-GPT4-train.txt",
        url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        bytes: 2_227_753_162,
    },
];

type TokenMeta = {
    sourceBytes: number;
    tokens: number;
    encoding: "gpt2";
    eot: number;
};

function metaPath(split: TokenSplit): string {
    return join(dataDirectory, `${split}.json`);
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

function assertGpt2Encoder(enc: Tiktoken): void {
    const special = gpt2.special_tokens[SEPARATOR];
    const encoded = enc.encode(SEPARATOR, [SEPARATOR]);
    if (special !== EOT || encoded.length !== 1 || encoded[0] !== EOT)
        throw new Error(`GPT-2 encoder did not map ${SEPARATOR} to ${EOT}`);
}

async function download(url: string, destination: string, expectedBytes: number): Promise<void> {
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

    constructor(private readonly file: Awaited<ReturnType<typeof open>>) {}

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

async function encodeSplit(enc: Tiktoken, source: string, split: TokenSplit, sourceBytes: number): Promise<void> {
    const bin = tokenPath(split);
    const metaFile = metaPath(split);
    const existing = await readMeta(metaFile);
    const binStat = await stat(bin).catch(() => undefined);
    if (existing?.sourceBytes === sourceBytes && existing.encoding === "gpt2" && existing.eot === EOT
        && binStat?.size === existing.tokens * 2) {
        console.log(`${split}: ${existing.tokens.toLocaleString()} tokens already encoded`);
        return;
    }

    const tmp = `${bin}.tmp`;
    await rm(tmp, { force: true });
    const writer = new TokenWriter(await open(tmp, "w"));
    const pieces: string[] = [];
    let pieceChars = 0;
    let pending = "";
    let sawSeparator = false;
    let readBytes = 0;
    let nextLog = 50 * 1024 * 1024;

    const flushPieces = async (): Promise<void> => {
        if (pieceChars === 0) return;
        await writer.push(enc.encode(pieces.join(""), [SEPARATOR]));
        pieces.length = 0;
        pieceChars = 0;
    };

    const absorb = async (text: string): Promise<void> => {
        if (text.length === 0) return;
        pending += text;
        let start = 0;
        while (true) {
            const at = pending.indexOf(SEPARATOR, start);
            if (at === -1) break;
            sawSeparator = true;
            const end = at + SEPARATOR.length;
            pieces.push(pending.slice(start, end));
            pieceChars += end - start;
            start = end;
            if (pieceChars >= ENCODE_CHARS)
                await flushPieces();
        }
        pending = pending.slice(start);
        if (!sawSeparator && pending.length > 8 * 1024 * 1024)
            throw new Error(`${source} has no ${SEPARATOR} separator in the first 8MB`);
    };

    const decoder = new TextDecoder("utf-8");
    try {
        for await (const chunk of createReadStream(source, { highWaterMark: 1 << 20 })) {
            const bytes = chunk as Buffer;
            readBytes += bytes.length;
            await absorb(decoder.decode(bytes, { stream: true }));
            if (readBytes >= nextLog) {
                const pct = (100 * readBytes / sourceBytes).toFixed(1);
                console.log(`${split}: read ${pct}%  tokens ${writer.tokens.toLocaleString()}`);
                nextLog += 50 * 1024 * 1024;
            }
        }
        await absorb(decoder.decode());
        if (pending.length > 0) {
            pieces.push(pending);
            pieceChars += pending.length;
            pending = "";
        }
        await flushPieces();
        if (!sawSeparator)
            throw new Error(`${source} has no ${SEPARATOR} separators`);
        await writer.finish();
    }
    catch (error) {
        await writer.finish().catch(() => undefined);
        await rm(tmp, { force: true });
        throw error;
    }

    await rm(bin, { force: true });
    await rename(tmp, bin);
    const meta: TokenMeta = { sourceBytes, tokens: writer.tokens, encoding: "gpt2", eot: EOT };
    await writeFile(metaFile, JSON.stringify(meta, null, 2));
    console.log(`${split}: wrote ${writer.tokens.toLocaleString()} tokens to ${bin}`);
    await preview(enc, bin);
}

async function readMeta(path: string): Promise<TokenMeta | undefined> {
    try {
        return JSON.parse(await readFile(path, "utf8")) as TokenMeta;
    }
    catch {
        return undefined;
    }
}

async function preview(enc: Tiktoken, bin: string): Promise<void> {
    const file = await open(bin, "r");
    try {
        const bytes = new Uint8Array(64);
        const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
        const ids = Array.from(new Uint16Array(bytes.buffer, 0, Math.floor(bytesRead / 2)));
        console.log(`${bin} starts: ${JSON.stringify(enc.decode(ids).slice(0, 180))}`);
    }
    finally {
        await file.close();
    }
}

export async function prepareTinyStories(): Promise<void> {
    const enc = new Tiktoken(gpt2);
    assertGpt2Encoder(enc);
    await mkdir(dataDirectory, { recursive: true });
    for (const source of SOURCES) {
        const textPath = join(dataDirectory, source.name);
        await download(source.url, textPath, source.bytes);
        await encodeSplit(enc, textPath, source.split, source.bytes);
    }
}

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("prepare_tinystories")) {
    await prepareTinyStories();
}
