# SolidChat — native macOS app

A real SwiftUI app. No web view, no browser, no localhost UI server. It spawns the inference
engine as a child process and talks to it over the OpenAI API, and everything else — model
scanning, HuggingFace downloads, SSE streaming, markdown rendering, persistence, memory
stats — is native Swift.

```bash
./build.sh && open build/SolidChat.app
```

To keep it around:

```bash
cp -R build/SolidChat.app /Applications/
```

## Requirements

Command Line Tools only — **no Xcode needed**. `build.sh` runs `swiftc` over every source file as
one whole-module optimized build (~10 s), assembles the `.app` bundle, and ad-hoc signs it
(unsigned SwiftUI binaries get killed on launch).

## Engines

| Engine | Format | Notes |
|---|---|---|
| **llama.cpp** | GGUF | PrismML build — the only one that runs the ternary/`q2_0` models |
| **MLX** | safetensors | Apple's fast path, via `mlx_lm.server` from the repo venv |
| **vLLM** | safetensors, unquantized | Code path exists; vLLM has no Metal backend and is not installed |

Each model row offers a Load button only for the engines its format actually supports.

## What's in it

- **Chat** — streaming, native markdown with syntax-highlighted code blocks and real tables,
  `<think>` reasoning folded into a disclosure, per-message tok/s · tokens · TTFT, edit-and-resend,
  regenerate, stop. Return sends, Shift-Return newlines.
- **Models** — scans `~/Downloads/LM_Studio_Models`, sorts by estimated speed, shows quant, size,
  and a memory-fit dot. Per-model load config (context, GPU layers, flash attention, Q8 KV cache).
- **Download** — paste a HuggingFace repo URL, pick a quant, 8 parallel range requests with live
  speed and ETA. Resumable, cancellable, atomic install straight into the models folder.
- **Logs** — the engine's own stdout, which is where a failed load explains itself.
- **Settings** (⌘,) — samplers with explanations, path overrides, HuggingFace token.

Menu commands: New Chat ⌘N, Rescan ⌘R, Unload ⌘⇧U, Stop ⌘.

## Layout

```
Sources/SolidChat/
  Core/    Types, ModelScanner, EngineManager, ProcessRunner,
           ChatClient (+ThinkSplitter), Downloader, Store, SystemStats
  Views/   RootView, ChatView, MarkdownMessageView, ModelsView,
           DownloadsView, LogsView, SettingsView
  SolidChatApp.swift
build.sh          swiftc -> .app bundle
SolidChat.icns    generated icon
```

State lives in `~/Library/Application Support/SolidChat/` (conversations, settings, engine log).

## Notes

- One model at a time — 24 GB unified memory. Loading a second evicts the first.
- **Full GPU offload is the default** (`gpuLayers` 999). An auto-chosen offload ratio measured
  3.99 tok/s on a 26B-A4B where full offload measured 37.9.
- The `~N tok/s` figure is an estimate — `115 GB/s ÷ bytes-read-per-token`, using this M5's measured
  effective bandwidth and a per-family MoE read fraction. Not a benchmark. Both constants are in
  `Core/ModelScanner.swift` and are hardware-specific.
- `mlx_lm.server` treats the request's `model` field as a HuggingFace repo id and will re-download a
  model already on disk. `ChatClient` sends the local path for MLX instead. Verified: zero
  huggingface.co requests in the engine log during MLX generation.
- The HuggingFace token is stored in plain text. Use a read-only token.

The earlier web version still lives in the parent directory and still works; this app replaces it.

## Verification

`swiftc -typecheck` over all sources: **0 errors, 0 warnings**. Verified against real models:

| Check | Result |
|---|---|
| Model scan | 30 models, 19 GGUF / 11 MLX, mmproj excluded, quants correct |
| llama.cpp load → stream → unload | 201.7 tok/s, TTFT 0.04 s, child process confirmed dead after unload |
| MLX load → stream | reasoning captured, **0 huggingface.co requests** (no re-download) |
| HuggingFace inspect | 24 quants parsed, `Q3_K_XL` intact, bad repo strings rejected |
| Markdown parser | 19 adversarial inputs + every streaming prefix of a document |
| ThinkSplitter | fed one character at a time, split tags, two blocks, incomplete tag |

A 6-lens adversarial review confirmed 19 defects (5 rejected on verification); all were fixed.
The most serious: `presize()` gives a file its final length before any bytes arrive, so an
interrupted download left a full-size zero-filled file that the size-only resume check 
treated as complete — silently installing a corrupt multi-gigabyte model. Resume now trusts a
completion ledger (`.solidchat-resume.json` inside the staging directory) instead of file size.
