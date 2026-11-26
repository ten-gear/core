import { parentPort } from "node:worker_threads";
import { download, type DownloadJob, type DownloadResult } from "./gpt2_token_io.js";

if (parentPort === null)
    throw new Error("prepare_download_worker must be started as a worker");

const port = parentPort;

port.on("message", (job: DownloadJob) => {
    void download(job.url, job.destination, job.bytes).then(
        () => {
            const result: DownloadResult = { id: job.id, ok: true };
            port.postMessage(result);
        },
        (error: unknown) => {
            const result: DownloadResult = {
                id: job.id,
                ok: false,
                error: error instanceof Error ? error.stack ?? error.message : String(error),
            };
            port.postMessage(result);
        },
    );
});
