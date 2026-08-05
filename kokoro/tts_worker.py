#!/usr/bin/env python3
"""Persistent Kokoro TTS worker for SolidChat.

Loads the model once and then answers requests forever, because a fresh
`mlx_audio.tts.generate` process spends ~2.5s loading before it says a word —
unacceptable when the caller wants a chat reply spoken sentence by sentence.

Protocol: one JSON object per line on stdin, one per line on stdout.
  ->  {"text": "...", "voice": "af_heart", "speed": 1.0, "out": "/tmp/x.wav"}
  <-  {"ok": true, "path": "/tmp/x.wav", "seconds": 3.2}
  <-  {"ok": false, "error": "..."}
On startup: {"ready": true, "voices": [...], "sampleRate": 24000}

Everything except these JSON lines goes to stderr, so a stray library print
can never corrupt the protocol.
"""
import contextlib
import json
import os
import sys
import wave
from pathlib import Path

MODEL = os.environ.get("SOLIDCHAT_TTS_MODEL", "prince-canuma/Kokoro-82M")
SAMPLE_RATE = 24000


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def discover_voices():
    """Voice names are the .safetensors files shipped inside the model snapshot."""
    try:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(MODEL, allow_patterns=["voices/*"]))
        names = sorted(p.stem for p in (root / "voices").glob("*.safetensors"))
        if names:
            return names
    except Exception as exc:  # noqa: BLE001 — never fatal, the UI can live without the list
        print(f"voice discovery failed: {exc}", file=sys.stderr)
    return ["af_heart"]


def main():
    # mlx_audio chatters on stdout; keep the protocol clean by redirecting
    # anything it prints during import and generation to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_audio.tts.generate import generate_audio  # noqa: PLC0415

        voices = discover_voices()
        # Warm the graph so the first real request is not the slow one.
        try:
            generate_audio(text="ready", model=MODEL, voice=voices[0],
                           file_prefix=str(Path(os.environ.get("TMPDIR", "/tmp")) / "solidchat-tts-warm"),
                           audio_format="wav", join_audio=True, verbose=False, save=True)
        except Exception as exc:  # noqa: BLE001
            print(f"warmup failed: {exc}", file=sys.stderr)

    emit({"ready": True, "voices": voices, "sampleRate": SAMPLE_RATE})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"ok": False, "error": f"bad request: {exc}"})
            continue

        if req.get("quit"):
            return

        text = (req.get("text") or "").strip()
        if not text:
            emit({"ok": False, "error": "empty text"})
            continue

        out = Path(req.get("out") or (Path(os.environ.get("TMPDIR", "/tmp")) / "solidchat-tts.wav"))
        # generate_audio appends "_000.wav" to the prefix it is given.
        prefix = str(out.with_suffix(""))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                generate_audio(
                    text=text,
                    model=MODEL,
                    voice=req.get("voice") or "af_heart",
                    speed=float(req.get("speed") or 1.0),
                    lang_code=req.get("lang") or "en",
                    file_prefix=prefix,
                    audio_format="wav",
                    join_audio=True,
                    verbose=False,
                    save=True,   # without this generate_audio returns without writing anything
                )
            # join_audio=True writes "<prefix>.wav"; the segmented path writes
            # "<prefix>_000.wav". Accept either rather than betting on one.
            produced = next((p for p in (Path(f"{prefix}.wav"), Path(f"{prefix}_000.wav"))
                             if p.exists()), None)
            if produced is None:
                emit({"ok": False, "error": "no audio produced"})
                continue
            if produced != out:
                produced.replace(out)
            with contextlib.closing(wave.open(str(out))) as w:
                seconds = w.getnframes() / float(w.getframerate() or SAMPLE_RATE)
            emit({"ok": True, "path": str(out), "seconds": round(seconds, 2)})
        except Exception as exc:  # noqa: BLE001 — one bad utterance must not kill the worker
            emit({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
