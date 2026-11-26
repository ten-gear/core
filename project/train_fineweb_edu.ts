import { join } from "node:path";
import { destroyPool } from "../src/fast_ops.js";
import { checkpointDirectory } from "./checkpoint.js";
import { logDirectory } from "./metrics.js";
import { trainGpt2 } from "./gpt2_train.js";
import { prepareFinewebEdu } from "./prepare_fineweb_edu.js";
import { tokenPath } from "./fineweb_edu_data.js";

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("train_fineweb_edu")) {
    try {
        await trainGpt2({
            name: "FineWeb-Edu",
            checkpoints: join(checkpointDirectory, "fineweb-edu"),
            logs: join(logDirectory, "fineweb-edu"),
            prepare: prepareFinewebEdu,
            tokenPath,
        });
    }
    finally {
        destroyPool();
    }
}
