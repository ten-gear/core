import { parentPort } from "node:worker_threads";
import { createGpt2Encoder, encodeShard, type EncodeJob, type EncodeResult } from "./gpt2_token_io.js";
import { documentsFor } from "./prepare_documents.js";

if (parentPort === null)
    throw new Error("prepare_encode_worker must be started as a worker");

const port = parentPort;
const encoder = createGpt2Encoder();

port.on("message", (job: EncodeJob) => {
    void encodeShard({
        encoder,
        documents: documentsFor(job.format, job.parquetPath),
        binPath: job.binPath,
        metaPath: job.metaPath,
        name: job.name,
        sourceBytes: job.sourceBytes,
        label: job.label,
    }).then(
        (record) => {
            const result: EncodeResult = { id: job.id, ok: true, record };
            port.postMessage(result);
        },
        (error: unknown) => {
            const result: EncodeResult = {
                id: job.id,
                ok: false,
                error: error instanceof Error ? error.stack ?? error.message : String(error),
            };
            port.postMessage(result);
        },
    );
});
