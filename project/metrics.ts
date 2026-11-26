import { existsSync } from "node:fs";
import { mkdir, open, type FileHandle } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { getBackend } from "../src/backend/index.js";
import { readStorageAsync, TensorData } from "../src/tensor_data.js";
import { Tensor } from "../src/tensor.js";

export const logDirectory = fileURLToPath(new URL("./logging/", import.meta.url));

export const METRIC_HEADER = "step,train_loss,val_loss,learning_rate,tokens_seen,tokens_per_sec,step_time_ms,grad_norm,weight_norm,peak_memory_mb,hellaswag";

export type MetricRow = {
    step: number;
    trainLoss: number;
    valLoss: number;
    learningRate: number;
    tokensSeen: number;
    tokensPerSec: number | null;
    stepTimeMs: number | null;
    gradNorm: number;
    weightNorm: number;
    peakMemoryMb: number;
    hellaswag: number;
};

export function shouldLogMetrics(step: number, steps: number, every: number): boolean {
    if (step === 0 || step === steps) return true;
    return step % every === 0;
}

export function metricSteps(steps: number, every: number): number[] {
    const logged: number[] = [];
    for (let step = 0; step <= steps; step++) {
        if (shouldLogMetrics(step, steps, every))
            logged.push(step);
    }
    return logged;
}

export function formatMetricRow(row: MetricRow): string {
    return [
        String(row.step),
        cell(row.trainLoss),
        cell(row.valLoss),
        cell(row.learningRate),
        cell(row.tokensSeen),
        cell(row.tokensPerSec),
        cell(row.stepTimeMs),
        cell(row.gradNorm),
        cell(row.weightNorm),
        cell(row.peakMemoryMb),
        cell(row.hellaswag),
    ].join(",");
}

export function peakMemoryMb(): number {
    const maxRssKb = process.resourceUsage().maxRSS;
    if (maxRssKb > 0)
        return maxRssKb / 1024;
    return process.memoryUsage().rss / (1024 * 1024);
}

export async function tensorL2(tensors: readonly Tensor[]): Promise<number> {
    await getBackend().synchronize();
    let sum = 0;
    for (const tensor of tensors) {
        assertContiguous(tensor);
        const values = await readStorageAsync(tensor.data._storage);
        if (values.length !== tensor.data.size)
            throw new Error(`tensor storage has ${values.length} values, expected ${tensor.data.size}`);
        for (let i = 0; i < values.length; i++) {
            const value = values[i]!;
            sum += value * value;
        }
    }
    return Math.sqrt(sum);
}

export class MetricsLog {
    private constructor(private readonly file: FileHandle) {}

    static async open(path: string, resume: boolean): Promise<MetricsLog> {
        await mkdir(dirname(path), { recursive: true });
        if (!resume || !existsSync(path)) {
            const file = await open(path, "w");
            await writeAll(file, Buffer.from(`${METRIC_HEADER}\n`));
            await file.sync();
            return new MetricsLog(file);
        }
        return new MetricsLog(await open(path, "a"));
    }

    async write(row: MetricRow): Promise<void> {
        await writeAll(this.file, Buffer.from(`${formatMetricRow(row)}\n`));
        await this.file.sync();
    }

    async close(): Promise<void> {
        await this.file.close();
    }
}

function cell(value: number | null): string {
    if (value === null) return "";
    if (!Number.isFinite(value))
        throw new Error("metric is not finite");
    return String(value);
}

function assertContiguous(tensor: Tensor): void {
    const expected = TensorData.calculateStrides(tensor.data._shape);
    const stride = tensor.data._stride;
    if (expected.length !== stride.length || expected.some((value, index) => value !== stride[index]))
        throw new Error("metric norm requires a contiguous row-major tensor");
}

async function writeAll(file: FileHandle, bytes: Buffer): Promise<void> {
    let offset = 0;
    while (offset < bytes.length) {
        const { bytesWritten } = await file.write(bytes, offset, bytes.length - offset, null);
        if (bytesWritten === 0)
            throw new Error("failed to write metrics");
        offset += bytesWritten;
    }
}
