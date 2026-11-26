import { existsSync } from "node:fs";
import { Model } from "../src/model.ts"
import { Tensor, tensorScope } from "../src/tensor.ts"
import { Layer } from "../src/model.js";
import { logsoftmax } from "../src/tensor_functions.ts"
import { TensorData, writeStorage } from "../src/tensor_data.js";
import { getBackendKind, initBackend } from "../src/backend/index.js";
import { AdamW } from "../src/optim.js";
import { destroyPool } from "../src/fast_ops.js";
import * as layers from "../src/layers.ts"
import { checkpointDirectory, loadLatestCheckpoint, saveCheckpoint } from "./checkpoint.js";
import { prepareTinyStories } from "./prepare_tinystories.js";
import { loadTokenFile, Random, sampleBatch, tokenPath } from "./tinystories_data.js";

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

    forward(ids: Tensor): Tensor {
        let x = this.embed.forward(ids);
        for (const block of this.blocks) x = block.forward(x);
        return this.head.forward(this.norm.forward(x));
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
const VAL_EVERY = 50;
const SAVE_CHECKPOINTS = true;
const RESUME_CHECKPOINT = false;
const CHECKPOINT_EVERY = 100;
const SEED = Math.random();

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

async function ensureTokens(): Promise<void> {
    if (existsSync(tokenPath("train")) && existsSync(tokenPath("valid"))) return;
    console.log("TinyStories token files are missing. Downloading and encoding them.");
    await prepareTinyStories();
}

async function train(): Promise<void> {
    const steps = readSteps();
    if (SAVE_CHECKPOINTS && (!Number.isInteger(CHECKPOINT_EVERY) || CHECKPOINT_EVERY < 1))
        throw new Error(`CHECKPOINT_EVERY must be a positive integer, received ${CHECKPOINT_EVERY}`);
    await ensureTokens();
    await initBackend({ backend: "auto" });
    console.log(`backend ${getBackendKind()}`);

    const trainTokens = await loadTokenFile(tokenPath("train"));
    const validTokens = await loadTokenFile(tokenPath("valid"));
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

    const rng = new Random(SEED);
    let startStep = 1;
    let resumed = false;
    if (RESUME_CHECKPOINT) {
        const loaded = await loadLatestCheckpoint(checkpointDirectory, model, optim, checkpointSpec);
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

    model.train();
    console.log(`GPT-2  vocab ${VOCAB}  block ${BLOCK_SIZE}  batch ${BATCH_SIZE}  steps ${steps}`);
    const writeCheckpoint = (step: number) => saveCheckpoint({
        root: checkpointDirectory,
        step,
        seed: SEED,
        rng,
        model,
        optim,
        modelConfig: checkpointSpec.model,
        hyperparameters: checkpointSpec.hyperparameters,
    });
    if (SAVE_CHECKPOINTS && !resumed) {
        await writeCheckpoint(0);
        console.log("checkpoint step 0");
    }

    const started = performance.now();
    for (let step = startStep; step <= steps; step++) {
        const lossValue = await tensorScope(() => {
            optim.zeroGrad();
            const batch = sampleBatch(trainTokens, BATCH_SIZE, BLOCK_SIZE, rng);
            const logits = model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE));
            const loss = crossEntropy(logits, idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE));
            loss.backward();
            optim.step();
            return loss.itemAsync();
        });
        console.log(`step ${step}  loss ${lossValue.toFixed(4)}`);

        if (step % VAL_EVERY === 0) {
            model.eval();
            const val = await tensorScope(() => {
                const batch = sampleBatch(validTokens, BATCH_SIZE, BLOCK_SIZE, rng);
                const loss = crossEntropy(
                    model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE)),
                    idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE),
                );
                return loss.itemAsync();
            });
            model.train();
            console.log(`step ${step}  val ${val.toFixed(4)}`);
        }

        if (SAVE_CHECKPOINTS && step % CHECKPOINT_EVERY === 0) {
            await writeCheckpoint(step);
            console.log(`checkpoint step ${step}`);
        }
    }
    const seconds = (performance.now() - started) / 1000;
    console.log(`finished ${steps} steps in ${seconds.toFixed(1)}s`);
}

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("train_gpt_2")) {
    try {
        await train();
    }
    finally {
        destroyPool();
    }
}
