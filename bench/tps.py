#!/usr/bin/env python3
"""Measure generation tok/s from any OpenAI-compatible server, identically.

Streams and counts SSE chunks, timing from the FIRST chunk to the last so
prompt processing never inflates the number. Counts both `content` and
`reasoning_content` — thinking models put their tokens in the latter, and
ignoring it would report a fast model as producing nothing.

Where the server also reports its own timings (llama.cpp does), those are
printed alongside so the chunk-count method can be checked rather than trusted.
"""
import json
import sys
import time
import urllib.request

PROMPT = ("Write a detailed technical explanation of how virtual memory paging works "
          "in a modern operating system. Cover the TLB, page faults, and swapping.")


def run(port, max_tokens=200, path="/v1/chat/completions", model="local"):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    first = None
    chunks = 0
    usage_tokens = None
    server_tps = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if (u := obj.get("usage")):
                usage_tokens = u.get("completion_tokens") or usage_tokens
            if (t := obj.get("timings")):
                server_tps = t.get("predicted_per_second") or server_tps
            for choice in obj.get("choices") or []:
                d = choice.get("delta") or {}
                # Thinking models emit reasoning_content; both are generated tokens.
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    if first is None:
                        first = time.time()
                    chunks += 1
    end = time.time()
    if first is None or chunks == 0:
        return None
    gen = end - first
    return {
        "tps": chunks / gen if gen > 0 else 0,
        "chunks": chunks,
        "usage_tokens": usage_tokens,
        "server_tps": server_tps,
        "ttft": first - t0,
        "gen_seconds": gen,
    }


if __name__ == "__main__":
    port = int(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    # mlx_lm treats this field as a HuggingFace repo id and will hit the network
    # (and 404) unless it is the local model path.
    model = sys.argv[3] if len(sys.argv) > 3 else "local"
    try:
        warm = run(port, model=model)
    except Exception as e:
        print(f"FAILED ({e})"); sys.exit(1)
    if warm is None:
        print("FAILED"); sys.exit(1)
    results = [r for _ in range(runs) if (r := run(port, model=model))]
    if not results:
        print("FAILED"); sys.exit(1)
    tps = [r["tps"] for r in results]
    mean = sum(tps) / len(tps)
    spread = (max(tps) - min(tps)) / 2
    srv = [r["server_tps"] for r in results if r["server_tps"]]
    extra = f" server_reported={sum(srv)/len(srv):.2f}" if srv else ""
    print(f"{mean:.2f} {spread:.2f} {max(tps):.2f} {results[0]['chunks']}{extra}")
