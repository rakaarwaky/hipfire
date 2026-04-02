#!/usr/bin/env bun
// hipfire CLI — ollama-style UX for AMD GPU inference
// Usage:
//   hipfire serve                    → start daemon + HTTP server
//   hipfire run <model> [prompt]     → interactive inference
//   hipfire list                     → show local models

import { spawn } from "bun";
import { existsSync, readdirSync, statSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

const HIPFIRE_DIR = join(homedir(), ".hipfire");
const MODELS_DIR = join(HIPFIRE_DIR, "models");
const DEFAULT_PORT = 11435;
const TEMP_CORRECTION = 0.82;

Bun.spawnSync(["mkdir", "-p", MODELS_DIR]);

// ─── Daemon IPC ─────────────────────────────────────────

class Engine {
  private proc: ReturnType<typeof spawn> | null = null;
  private lines: string[] = [];
  private buffer = "";

  async start() {
    const bins = [
      resolve(__dirname, "../target/release/examples/daemon"),
      join(HIPFIRE_DIR, "bin", "daemon"),
    ];
    const bin = bins.find(p => existsSync(p));
    if (!bin) throw new Error("daemon not found. cargo build --release --features deltanet --example daemon -p engine");

    this.proc = spawn([bin], { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
  }

  async send(msg: object) {
    if (!this.proc?.stdin) throw new Error("not running");
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
    await this.proc.stdin.flush();
  }

  async recv(): Promise<any> {
    if (!this.proc?.stdout) throw new Error("not running");
    const reader = this.proc.stdout.getReader();
    while (true) {
      if (this.lines.length > 0) {
        reader.releaseLock();
        return JSON.parse(this.lines.shift()!);
      }
      const { value, done } = await reader.read();
      if (done) throw new Error("daemon closed");
      this.buffer += new TextDecoder().decode(value);
      const parts = this.buffer.split("\n");
      this.buffer = parts.pop() || "";
      this.lines.push(...parts.filter(l => l.trim()));
      reader.releaseLock();
    }
  }

  async *generate(msg: object): AsyncGenerator<any> {
    await this.send(msg);
    while (true) {
      const r = await this.recv();
      yield r;
      if (r.type === "done" || r.type === "error") break;
    }
  }

  async stop() {
    try { await this.send({ type: "unload" }); } catch {}
    this.proc?.kill();
  }
}

// ─── Commands ───────────────────────────────────────────

async function run(model: string, prompt: string, image?: string, temp = 0.3, maxTokens = 512) {
  const path = findModel(model);
  if (!path) { console.error(`Model not found: ${model}\nRun: hipfire pull ${model}`); process.exit(1); }

  if (image && !existsSync(image)) { console.error(`Image not found: ${image}`); process.exit(1); }

  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  await e.send({ type: "load", model: path });
  const loaded = await e.recv();
  if (loaded.type === "error") { console.error(loaded.message); process.exit(1); }
  const vlTag = loaded.vl ? " VL" : "";
  console.error(`[${loaded.arch}${vlTag}] ${loaded.dim}d ${loaded.layers}L ${loaded.vocab} vocab`);

  if (image && !loaded.vl) {
    console.error(`WARNING: --image passed but model does not have a vision encoder. Ignoring image.`);
    image = undefined;
  }

  const genMsg: any = {
    type: "generate", id: "run", prompt,
    temperature: temp * TEMP_CORRECTION, max_tokens: maxTokens,
  };
  if (image) {
    genMsg.image = resolve(image);
    console.error(`[VL: ${image}]`);
  }

  for await (const msg of e.generate(genMsg)) {
    if (msg.type === "token") process.stdout.write(msg.text);
    else if (msg.type === "done") console.error(`\n[${msg.tokens} tok, ${msg.tok_s} tok/s]`);
  }
  await e.stop();
}

async function serve(port: number) {
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  let current: string | null = null;

  // Single-flight: only one generation at a time (GPU is single-threaded)
  let busy = false;
  const queue: Array<{ resolve: () => void }> = [];
  async function acquireLock() {
    if (!busy) { busy = true; return; }
    await new Promise<void>(resolve => queue.push({ resolve }));
  }
  function releaseLock() {
    const next = queue.shift();
    if (next) next.resolve();
    else busy = false;
  }

  console.error(`[hipfire] http://localhost:${port}/v1/chat/completions`);

  Bun.serve({
    port,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/health") return Response.json({ status: "ok", model: current });
      if (url.pathname === "/v1/models") return Response.json({ data: list().map(m => ({ id: m.name })) });

      if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
        await acquireLock();
        try {
        const body = await req.json();
        const prompt = (body.messages || []).map((m: any) => m.content).join("\n");
        const path = findModel(body.model || "default");
        if (!path) { releaseLock(); return Response.json({ error: "model not found" }, { status: 404 }); }

        if (current !== path) {
          if (current) { await e.send({ type: "unload" }); await e.recv(); }
          await e.send({ type: "load", model: path }); await e.recv();
          current = path;
        }

        if (body.stream) {
          const enc = new TextEncoder();
          return new Response(new ReadableStream({
            async start(ctrl) {
              for await (const msg of e.generate({
                type: "generate", id: "api", prompt,
                temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
                max_tokens: body.max_tokens ?? 512,
              })) {
                if (msg.type === "token") {
                  ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: msg.text } }] })}\n\n`));
                } else if (msg.type === "done") {
                  ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`));
                  ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                  ctrl.close();
                }
              }
            }
          }), { headers: { "Content-Type": "text/event-stream" } });
        }

        let content = "";
        for await (const msg of e.generate({
          type: "generate", id: "api", prompt,
          temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
          max_tokens: body.max_tokens ?? 512,
        })) { if (msg.type === "token") content += msg.text; }
        releaseLock();
        return Response.json({ choices: [{ message: { role: "assistant", content } }] });
        } finally { releaseLock(); }
      }
      return Response.json({ error: "not found" }, { status: 404 });
    }
  });
}

// ─── Helpers ────────────────────────────────────────────

function findModel(name: string): string | null {
  if (existsSync(name)) return resolve(name);
  const dirs = [resolve(__dirname, "../models"), MODELS_DIR];
  for (const dir of dirs) {
    try { for (const f of readdirSync(dir)) { if (f.includes(name) && f.endsWith(".hfq")) return join(dir, f); } } catch {}
  }
  return null;
}

function list() {
  const models: { name: string; size: string }[] = [];
  for (const dir of [resolve(__dirname, "../models"), MODELS_DIR]) {
    try { for (const f of readdirSync(dir)) {
      if (f.endsWith(".hfq")) {
        const sz = (statSync(join(dir, f)).size / 1e6).toFixed(0);
        models.push({ name: f, size: `${sz}MB` });
      }
    }} catch {}
  }
  return models;
}

// ─── Main ───────────────────────────────────────────────

const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case "serve": await serve(parseInt(rest[0]) || DEFAULT_PORT); break;
  case "run": {
    const model = rest[0];
    if (!model) { console.error("Usage: hipfire run <model> [--image img.png] [prompt]"); process.exit(1); }
    const imgIdx = rest.indexOf("--image");
    const image = imgIdx >= 0 ? rest[imgIdx + 1] : undefined;
    const filtered = rest.slice(1).filter((_, i) => i !== imgIdx - 1 && i !== imgIdx);
    const prompt = filtered.join(" ") || (image ? "Describe this image." : "Hello");
    await run(model, prompt, image);
    break;
  }
  case "list": for (const m of list()) console.log(`${m.name.padEnd(40)} ${m.size}`); break;
  case "rm": { const p = findModel(rest[0] || ""); if (p) { require("fs").unlinkSync(p); console.log(`Removed ${p}`); } break; }
  default:
    console.log(`hipfire — LLM inference for AMD GPUs

  serve [port]          Start OpenAI-compatible server (default: ${DEFAULT_PORT})
  run <model> [prompt]  Generate text
  list                  Show local models
  rm <model>            Delete model
  pull <model>          Download model (coming soon)`);
}
