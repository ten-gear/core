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
import { loadTokenFile, sampleBatch, tokenPath } from "./tinystories_data.js";
import { prepareTinyStories } from "./prepare_tinystories.js";

export class TransformerBlock extends Layer {
    readonly norm1: layers.LayerNorm;
    readonly attn: layers.MHAFlashAttention;
    readonly norm2: layers.LayerNorm;
    readonly mlp: layers.MLP;
    constructor(nEmbd: number, nHead: number, maxSeq: number, dropout = 0) {
        super();
        this.norm1 = new layers.LayerNorm(nEmbd)
        this.attn = new layers.MHAFlashAttention(nEmbd, nHead, maxSeq, dropout)
        this.norm2 = new layers.LayerNorm(nEmbd)
        this.mlp = new layers.MLP(nEmbd)
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
        this.embed = new layers.Embedding(vocab, nEmbd)
        this.blocks = [];
        for (let i = 0; i < nLayer; i++) {
            const block = new TransformerBlock(nEmbd, nHead, maxSeq);
            this.addModule(`block${i}`, block);
            this.blocks.push(block);
        }
        this.norm = new layers.LayerNorm(nEmbd)
        this.head = new layers.Linear(nEmbd, vocab)
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
const WEIGHT_DECAY = 0.1;
const VAL_EVERY = 50;

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
    await ensureTokens();
    await initBackend({ backend: "auto" });
    console.log(`backend ${getBackendKind()}`);

    const trainTokens = await loadTokenFile(tokenPath("train"));
    const validTokens = await loadTokenFile(tokenPath("valid"));
    console.log(`train tokens ${trainTokens.length.toLocaleString()}  valid tokens ${validTokens.length.toLocaleString()}`);

    const model = new GPT2(VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE);
    const optim = new AdamW(model.parameters(), LEARNING_RATE, BETA1, BETA2, 1e-8, WEIGHT_DECAY);
    for (const parameter of model.parameters()) {
        const value = parameter.value;
        if (value instanceof Tensor && value.grad === undefined)
            value.grad = Tensor.zeros(value.shape());
    }
    model.train();
    console.log(`GPT-2  vocab ${VOCAB}  block ${BLOCK_SIZE}  batch ${BATCH_SIZE}  steps ${steps}`);

    const started = performance.now();
    for (let step = 1; step <= steps; step++) {
        const lossValue = await tensorScope(() => {
            optim.zeroGrad();
            const batch = sampleBatch(trainTokens, BATCH_SIZE, BLOCK_SIZE);
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
                const batch = sampleBatch(validTokens, BATCH_SIZE, BLOCK_SIZE);
                const loss = crossEntropy(
                    model.forward(idsToTensor(batch.input, BATCH_SIZE, BLOCK_SIZE)),
                    idsToTensor(batch.target, BATCH_SIZE, BLOCK_SIZE),
                );
                return loss.itemAsync();
            });
            model.train();
            console.log(`step ${step}  val ${val.toFixed(4)}`);
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
