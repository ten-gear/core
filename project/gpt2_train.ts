import { existsSync } from "node:fs";
import { join } from "node:path";
import { Model, Layer } from "../src/model.js";
import { Tensor, tensorScope } from "../src/tensor.js";
import { logsoftmax } from "../src/tensor_functions.js";
import { TensorData, writeStorage } from "../src/tensor_data.js";
import { getBackendKind, initBackend } from "../src/backend/index.js";
import { AdamW } from "../src/optim.js";
import * as layers from "../src/layers.js";
import { loadLatestCheckpoint, saveCheckpoint } from "./checkpoint.js";
import { hellaswagAccuracy } from "./hellaswag_eval.js";
import { hellaswagIsEncoded, loadHellaSwagExamples } from "./hellaswag_data.js";
import { MetricsLog, peakMemoryMb, shouldLogMetrics, tensorL2, type MetricRow } from "./metrics.js";
import { prepareHellaSwag } from "./prepare_hellaswag.js";
import { fixedBatch, loadTokenFile, Random, sampleBatch, type TokenSplit } from "./tinystories_data.js";

const WEIGHT_STD = 0.02;

export class TransformerBlock extends Layer {
    readonly norm1: layers.LayerNorm;
    readonly attn: layers.MHAFlashAttention;
    readonly norm2: layers.LayerNorm;
    readonly mlp: layers.MLP;
    constructor(nEmbd: number, nHead: number, maxSeq: number, init: layers.WeightInit, residual: layers.WeightInit, dropout = 0) {
        super();
        this.norm1 = new layers.LayerNorm(nEmbd)
        this.attn = new layers.MHAFlashAttention(nEmbd, nHead, maxSeq, dropout, init, residual)
        this.norm2 = new layers.LayerNorm(nEmbd)
        this.mlp = new layers.MLP(nEmbd, 4, init, residual)
    }
    forward(input: Tensor, offset = 0): Tensor {
        const x = input.add(this.attn.forward(this.norm1.forward(input), offset));
        return x.add(this.mlp.forward(this.norm2.forward(x)));
    }
}

export class GPT2 extends Model {
    readonly embed: layers.Embedding;
    readonly blocks: TransformerBlock[];
    readonly norm: layers.LayerNorm;
    readonly head: layers.Linear;

    constructor(vocab: number, nEmbd: number, nHead: number, nLayer: number, maxSeq: number) {
        super();
        const init = layers.normalParameter(WEIGHT_STD);
        const residual = layers.normalParameter(WEIGHT_STD / Math.sqrt(2 * nLayer));
        this.embed = new layers.Embedding(vocab, nEmbd, init)
        this.blocks = [];
        for (let i = 0; i < nLayer; i++) {
            const block = new TransformerBlock(nEmbd, nHead, maxSeq, init, residual);
            this.addModule(`block${i}`, block);
            this.blocks.push(block);
        }
        this.norm = new layers.LayerNorm(nEmbd)
        this.head = new layers.Linear(nEmbd, vocab, init)
    }

    hidden(ids: Tensor): Tensor {
        let x = this.embed.forward(ids);
        for (const block of this.blocks) x = block.forward(x);
        return this.norm.forward(x);
    }

    forward(ids: Tensor): Tensor {
        return this.head.forward(this.hidden(ids));
    }
}

export function crossEntropy(logits: Tensor, targets: Tensor): Tensor {
    const [batch, time] = targets.shape();
    const nll = logsoftmax(logits, -1).gather(targets.view(batch, time, 1), 2).neg();
    return nll.sum(0).sum(1).sum(2).div(batch * time);
}

const VOCAB = 50304;
const N_EMBD = 768;
const N_HEAD = 12;
const N_LAYER = 12;
const BLOCK_SIZE = 1024;
const BATCH_SIZE = 4;
const LEARNING_RATE = 6e-4;
const BETA1 = 0.9;
const BETA2 = 0.95;
const EPS = 1e-8;
const WEIGHT_DECAY = 0.1;
const SAVE_CHECKPOINTS = true;
const RESUME_CHECKPOINT = false;
const CHECKPOINT_EVERY = 100;
const MONITOR = true;
const MONITOR_EVERY = 100;

const checkpointSpec = {
    model: {
        vocab: VOCAB,
        nEmbd: N_EMBD,
        nHead: N_HEAD,
        nLayer: N_LAYER,
        maxSeq: BLOCK_SIZE,
    },
    hyperparameters: {
        learningRate: LEARNING_RATE,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        weightDecay: WEIGHT_DECAY,
        batchSize: BATCH_SIZE,
        blockSize: BLOCK_SIZE,
    },
};

export type Gpt2Dataset = {
    name: string;
    checkpoints: string;
    logs: string;
    prepare: () => Promise<void>;
    tokenPath: (split: TokenSplit) => string;
};

function readSteps(): number {
    const raw = process.argv[2];
    if (raw === undefined) return 1000;
    const steps = Number(raw);
    if (!Number.isInteger(steps) || steps < 1)
        throw new Error(`steps must be a positive integer, received ${raw}`);
    return steps;
}

function idsToTensor(values: Float32Array, batch: number, time: number): Tensor {
    const data = TensorData.empty([batch, time]);
    writeStorage(data._storage, values);
    return new Tensor(data);
}

async function ensureTokens(dataset: Gpt2Dataset): Promise<void> {
    if (existsSync(dataset.tokenPath("train")) && existsSync(dataset.tokenPath("valid"))) return;
    console.log(`${dataset.name} token files are missing. Downloading and encoding them.`);
    await dataset.prepare();
}

async function ensureHellaswag(): Promise<void> {
    if (await hellaswagIsEncoded()) return;
    console.log("HellaSwag validation is missing. Downloading and encoding it.");
    await prepareHellaSwag();
}

function windowLoss(model: GPT2, tokens: Uint16Array): Promise<number> {
    const batch = fixedBatch(tokens, BATCH_SIZE, BLOCK_SIZE);
    const loss = crossEntropy(
        model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE)),
        idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE),
    );
    return loss.itemAsync();
}

function parameterTensors(model: GPT2, field: "value" | "grad"): Tensor[] {
    const tensors: Tensor[] = [];
    for (const parameter of model.parameters()) {
        const value = parameter.value;
        if (!(value instanceof Tensor))
            throw new Error("parameter is not a tensor");
        if (field === "value")
            tensors.push(value);
        else if (value.grad instanceof Tensor)
            tensors.push(value.grad);
    }
    return tensors;
}

export async function trainGpt2(dataset: Gpt2Dataset): Promise<void> {
    const steps = readSteps();
    const seed = Math.random();
    if (SAVE_CHECKPOINTS && (!Number.isInteger(CHECKPOINT_EVERY) || CHECKPOINT_EVERY < 1))
        throw new Error(`CHECKPOINT_EVERY must be a positive integer, received ${CHECKPOINT_EVERY}`);
    if (MONITOR && (!Number.isInteger(MONITOR_EVERY) || MONITOR_EVERY < 1))
        throw new Error(`MONITOR_EVERY must be a positive integer, received ${MONITOR_EVERY}`);
    await ensureTokens(dataset);
    if (MONITOR)
        await ensureHellaswag();
    await initBackend({ backend: "auto" });
    console.log(`backend ${getBackendKind()}`);
    console.log(`checkpoints ${dataset.checkpoints}`);

    const trainTokens = await loadTokenFile(dataset.tokenPath("train"));
    const validTokens = await loadTokenFile(dataset.tokenPath("valid"));
    console.log(`train tokens ${trainTokens.length.toLocaleString()}  valid tokens ${validTokens.length.toLocaleString()}`);

    const model = new GPT2(VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE);
    const parameterCount = model.parameters().reduce((total, parameter) => {
        const value = parameter.value;
        return total + (value instanceof Tensor ? value.size() : 1);
    }, 0);
    console.log(`parameters ${parameterCount.toLocaleString()}`);
    const optim = new AdamW(model.parameters(), LEARNING_RATE, BETA1, BETA2, EPS, WEIGHT_DECAY);
    for (const parameter of model.parameters()) {
        const value = parameter.value;
        if (value instanceof Tensor && value.grad === undefined)
            value.grad = Tensor.zeros(value.shape());
    }

    const rng = new Random(seed);
    let startStep = 1;
    let resumed = false;
    if (RESUME_CHECKPOINT) {
        const loaded = await loadLatestCheckpoint(dataset.checkpoints, model, optim, checkpointSpec);
        if (loaded !== null) {
            rng.setState(loaded.rngState);
            startStep = loaded.step + 1;
            resumed = true;
            console.log(`resumed from step ${loaded.step}`);
        }
    }
    if (startStep > steps) {
        console.log(`checkpoint step ${startStep - 1} is already at the ${steps} step target`);
        return;
    }

    const hellaswag = MONITOR ? await loadHellaSwagExamples() : null;
    const metrics = MONITOR ? await MetricsLog.open(join(dataset.logs, "metrics.csv"), resumed) : null;
    model.train();
    console.log(`${dataset.name}  GPT-2  vocab ${VOCAB}  block ${BLOCK_SIZE}  batch ${BATCH_SIZE}  steps ${steps}`);
    const writeCheckpoint = (step: number) => saveCheckpoint({
        root: dataset.checkpoints,
        step,
        seed,
        rng,
        model,
        optim,
        modelConfig: checkpointSpec.model,
        hyperparameters: checkpointSpec.hyperparameters,
    });

    const logRow = async (step: number, trainLoss: number | null, stepTimeMs: number | null): Promise<void> => {
        if (metrics === null || hellaswag === null) return;
        let loss = trainLoss;
        if (step === 0) {
            model.train();
            loss = await tensorScope(() => {
                optim.zeroGrad();
                const batch = fixedBatch(trainTokens, BATCH_SIZE, BLOCK_SIZE);
                const probe = crossEntropy(
                    model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE)),
                    idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE),
                );
                probe.backward();
                return probe.itemAsync();
            });
        }
        if (loss === null)
            throw new Error("training loss is missing");
        const gradNorm = await tensorL2(parameterTensors(model, "grad"));
        const weightNorm = await tensorL2(parameterTensors(model, "value"));
        if (step === 0)
            optim.zeroGrad();
        model.eval();
        const valLoss = await tensorScope(() => windowLoss(model, validTokens));
        const accuracy = await hellaswagAccuracy(model, hellaswag, BLOCK_SIZE);
        model.train();
        const stepTokens = BATCH_SIZE * BLOCK_SIZE;
        const row: MetricRow = {
            step,
            trainLoss: loss,
            valLoss,
            learningRate: optim.lr,
            tokensSeen: step * stepTokens,
            tokensPerSec: stepTimeMs === null || stepTimeMs === 0 ? null : stepTokens / (stepTimeMs / 1000),
            stepTimeMs,
            gradNorm,
            weightNorm,
            peakMemoryMb: peakMemoryMb(),
            hellaswag: accuracy,
        };
        if (!Number.isFinite(row.trainLoss))
            throw new Error("training loss is missing");
        await metrics.write(row);
        const throughput = row.tokensPerSec === null ? "" : `  tokens/s ${row.tokensPerSec.toFixed(1)}`;
        console.log(`step ${step}  val ${valLoss.toFixed(4)}  hellaswag ${accuracy.toFixed(4)}${throughput}`);
    };

    try {
        if (SAVE_CHECKPOINTS && !resumed) {
            await writeCheckpoint(0);
            console.log("checkpoint step 0");
        }
        if (MONITOR && !resumed)
            await logRow(0, null, null);

        const started = performance.now();
        for (let step = startStep; step <= steps; step++) {
            const stepStarted = performance.now();
            const lossValue = await tensorScope(() => {
                optim.zeroGrad();
                const batch = sampleBatch(trainTokens, BATCH_SIZE, BLOCK_SIZE, rng);
                const logits = model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE));
                const loss = crossEntropy(logits, idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE));
                loss.backward();
                optim.step();
                return loss.itemAsync();
            });
            const stepTimeMs = performance.now() - stepStarted;
            console.log(`step ${step}  loss ${lossValue.toFixed(4)}`);

            if (MONITOR && shouldLogMetrics(step, steps, MONITOR_EVERY))
                await logRow(step, lossValue, stepTimeMs);

            if (SAVE_CHECKPOINTS && step % CHECKPOINT_EVERY === 0) {
                await writeCheckpoint(step);
                console.log(`checkpoint step ${step}`);
            }
        }
        const seconds = (performance.now() - started) / 1000;
        console.log(`finished ${steps} steps in ${seconds.toFixed(1)}s`);
    }
    finally {
        await metrics?.close();
    }
}
