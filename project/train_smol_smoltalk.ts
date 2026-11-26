import { join } from "node:path";
import { destroyPool } from "../src/fast_ops.js";
import { checkpointDirectory } from "./checkpoint.js";
import { logDirectory } from "./metrics.js";
import { trainGpt2 } from "./gpt2_train.js";
import { prepareSmolSmoltalk } from "./prepare_smol_smoltalk.js";
import { tokenPath } from "./smol_smoltalk_data.js";

function isDirectRun(script: string): boolean {
    const entry = process.argv[1]?.replaceAll("\\", "/");
    return entry !== undefined && (entry.endsWith(`/${script}.ts`) || entry.endsWith(`/${script}.js`));
}

if (isDirectRun("train_smol_smoltalk")) {
    try {
        await trainGpt2({
            name: "Smol-SmolTalk",
            checkpoints: join(checkpointDirectory, "smol-smoltalk"),
            logs: join(logDirectory, "smol-smoltalk"),
            prepare: prepareSmolSmoltalk,
            tokenPath,
        });
    }
    finally {
        destroyPool();
    }
}
