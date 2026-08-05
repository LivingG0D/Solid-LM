# Benchmark harness

Every number in the top-level README came from these three files.

| File | What it does |
|---|---|
| `tps.py` | Measures generation tok/s from any OpenAI-compatible server. Streams and counts SSE chunks, timing from the **first** token so prompt processing never inflates the result. Counts `content`, `reasoning_content` **and** `reasoning` — thinking models use the latter two, and missing them reports a fast model as producing nothing. |
| `run.sh` | Starts an engine, waits until it is genuinely ready, measures, stops it. |
| `chart.swift` | Renders the README charts from the measured values. |

## Why it is written this way

**`llama-server` answers `/health` with a 503 while it pages in.** A bare "did curl succeed" check
calls it ready far too early and you measure a cold model. `run.sh` greps for `"ok"`.

**mlx_lm treats the request's `model` field as a HuggingFace repo id.** Pass `local` and it tries
`GET huggingface.co/api/models/local` and 404s. Pass the model's local path as the third argument.

**One engine at a time.** On a 24 GB machine two resident models make every number a lie.

## Use

```bash
# llama.cpp-style server on :8321
source run.sh
export DYLD_LIBRARY_PATH=/path/to/llama-build
run_bench "label" 8321 /path/to/llama-server -m model.gguf -ngl 999 -c 4096 -fa on \
  --host 127.0.0.1 --port 8321 --jinja

# MLX — note the model path as argv[3]
python3 tps.py 8383 3 /path/to/mlx-model-dir
```

Output: `mean spread best chunks [server_reported=…]`. When the server reports its own
`predicted_per_second`, it is printed alongside so the chunk count can be checked rather than
trusted. On the runs in the README the two agreed exactly: **14.26 vs 14.26**.
