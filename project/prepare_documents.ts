import { readParquetRows, type DocumentFormat } from "./gpt2_token_io.js";

export function documentsFor(format: DocumentFormat, path: string): AsyncIterable<string> {
    if (format === "fineweb-text") return finewebDocuments(path);
    if (format === "chat-messages") return conversationDocuments(path);
    throw new Error(`unknown document format ${format}`);
}

async function* finewebDocuments(path: string): AsyncGenerator<string> {
    for await (const row of readParquetRows(path, ["text"])) {
        const text = row.text;
        if (text === null || text === undefined || text === "") continue;
        if (typeof text !== "string")
            throw new Error(`${path} text column is ${typeof text}, expected a string`);
        yield text;
    }
}

function describe(value: unknown): string {
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (Array.isArray(value)) return `an array of ${value.length}`;
    if (typeof value === "object")
        return `an object with keys ${Object.keys(value).slice(0, 8).join(", ")}`;
    return typeof value;
}

// One conversation becomes role/content turns. Encoding appends <|endoftext|> after each document.
function renderConversation(messages: unknown, path: string): string | undefined {
    if (messages === null || messages === undefined) return undefined;
    if (!Array.isArray(messages))
        throw new Error(`${path} messages is ${describe(messages)}, expected a list of {role, content}`);
    const turns: string[] = [];
    for (let i = 0; i < messages.length; i++) {
        const message: unknown = messages[i];
        if (typeof message !== "object" || message === null)
            throw new Error(`${path} message ${i} is ${describe(message)}, expected {role, content}`);
        const record = message as Record<string, unknown>;
        const role = record.role;
        const content = record.content;
        if (typeof role !== "string" || typeof content !== "string")
            throw new Error(`${path} message ${i} is ${describe(message)}, expected {role, content}`);
        if (role.length === 0 && content.length === 0) continue;
        turns.push(`${role}\n${content}`);
    }
    if (turns.length === 0) return undefined;
    return turns.join("\n");
}

async function* conversationDocuments(path: string): AsyncGenerator<string> {
    for await (const row of readParquetRows(path, ["messages"])) {
        const text = renderConversation(row.messages, path);
        if (text !== undefined) yield text;
    }
}
