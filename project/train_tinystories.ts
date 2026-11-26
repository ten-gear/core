import { join } from "node:path";
import { destroyPool } from "../src/fast_ops.js";
import { checkpointDirectory } from "./checkpoint.js";
import { logDirectory } from "./metrics.js";
import { trainGpt2 } from "./gpt2_train.js";
import { prepareTinyStories } from "./prepare_tinystories.js";
import { tokenPath } from "./tinystories_data.js";

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("train_tinystories")) {
    try {
        await trainGpt2({
            name: "TinyStories",
            checkpoints: join(checkpointDirectory, "tinystories"),
            logs: join(logDirectory, "tinystories"),
            prepare: prepareTinyStories,
            tokenPath,
        });
    }
    finally {
        destroyPool();
    }
}
