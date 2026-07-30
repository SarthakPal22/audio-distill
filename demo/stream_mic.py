"""Real-time keyword spotting from the microphone.

Streaming pattern: a 1-second **sliding window** over a ring buffer,
re-classified every `--hop-ms` (default 250 ms). Total per-hop budget =
front-end + model latency; the int8 student fits it with ~100x headroom,
which is the whole point of compressing the model.

Run:
  pip install -e ".[demo,deploy]"
  python -m demo.stream_mic --run-dir runs/best
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import torch

from phase0.data import CLIP_SAMPLES, LABELS, SAMPLE_RATE
from phase0.student import MelFrontend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/best")
    parser.add_argument("--hop-ms", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="min softmax confidence to report a keyword")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model = next((run_dir / n for n in
                  ("model_int8_static.onnx", "model_fp32.onnx")
                  if (run_dir / n).exists()), None)
    assert model, f"no ONNX model in {run_dir}; run tools.deploy first"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    frontend = MelFrontend().eval()

    ring: collections.deque[np.ndarray] = collections.deque(
        maxlen=CLIP_SAMPLES // 160)  # 10 ms blocks

    def on_audio(indata, frames, t, status) -> None:
        ring.append(indata[:, 0].copy())

    print(f"listening ({model.name}); say one of: yes no up down left right "
          f"stop go ... Ctrl-C to quit")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=160,
                        callback=on_audio):
        while True:
            time.sleep(args.hop_ms / 1000)
            if len(ring) < ring.maxlen:
                continue
            wave = torch.from_numpy(np.concatenate(ring)[-CLIP_SAMPLES:])
            with torch.no_grad():
                spec = frontend(wave.unsqueeze(0)).numpy()
            t0 = time.perf_counter()
            logits = session.run(None, {"spec": spec})[0][0]
            ms = (time.perf_counter() - t0) * 1000
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            best = int(probs.argmax())
            if probs[best] >= args.threshold:
                print(f"  {LABELS[best]:>10s}  ({probs[best]:.0%}, {ms:.1f} ms)")


if __name__ == "__main__":
    main()
