"""Deployment pipeline: ONNX export -> numerical parity check ->
post-training quantization -> honest CPU latency benchmark.

Concepts:

- **ONNX** (Open Neural Network Exchange): a portable graph format;
  `onnxruntime` (ORT) executes it with fused, vectorized CPU kernels,
  typically beating eager PyTorch at batch-1 inference.
- We export the **spectrogram classifier only** (input: 1x64x101 log-mel).
  The FFT front-end stays outside the graph — standard practice for
  keyword spotting, since DSP libraries compute mel features cheaply
  everywhere, while FFT ops inside ONNX graphs have patchy support.
- **Numerical parity**: after export we assert max |Δlogit| between
  PyTorch and ORT is tiny; a silent export bug usually shows up here.
- **Post-training quantization (PTQ)**: converting trained fp32 weights
  to int8 *without retraining* (contrast with quantization-aware
  training, QAT, which simulates quantization during training).
  - *Dynamic PTQ*: int8 weights, activations quantized on the fly per
    batch. Zero calibration needed; mainly helps big matmuls.
  - *Static PTQ*: weights **per-channel affine int8** (each output
    channel gets its own scale/zero-point — much tighter than one scale
    per tensor) plus activation ranges fixed ahead of time by running a
    **calibration set** through observers. Best latency: everything
    stays in int8 between ops.
- **Latency protocol**: 20 warm-up runs (first calls pay one-time
  allocation/autotune costs), then 500 timed single-clip runs, reporting
  median and p99 — tail latency is what a real-time budget must survive.

Usage:
  python -m tools.deploy --run-dir runs/cnn1_a0.3_T4_s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase0.data import PackedSpeechCommands
from phase0.evaluate import load_student
from phase0.student import N_FRAMES, N_MELS


class SpecNet(torch.nn.Module):
    """Wraps a student so forward() takes the (B, 1, 64, 101) spectrogram."""

    def __init__(self, student: torch.nn.Module):
        super().__init__()
        self.student = student

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.student.spectrogram_forward(spec)


def export_onnx(student: torch.nn.Module, path: Path) -> None:
    net = SpecNet(student).eval()
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(net, (dummy,), str(path), input_names=["spec"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"spec": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)


def make_spectrograms(student, waveforms: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return student.frontend(waveforms).numpy()


def ort_latency(session, spec: np.ndarray, runs: int = 500,
                warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        session.run(None, {"spec": spec})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"spec": spec})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": round(times[len(times) // 2], 3),
            "p99_ms": round(times[int(runs * 0.99)], 3)}


def ort_accuracy(session, student, packs_dir: str) -> float:
    ds = PackedSpeechCommands(packs_dir, "testing")
    loader = DataLoader(ds, batch_size=256, num_workers=2)
    correct = 0
    for waveforms, labels, *_ in loader:
        specs = make_spectrograms(student, waveforms)
        logits = session.run(None, {"spec": specs})[0]
        correct += int((logits.argmax(-1) == labels.numpy()).sum())
    return correct / len(ds)


def main() -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                          quantize_dynamic, quantize_static)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--packs-dir", default="data/packs")
    parser.add_argument("--calib-clips", type=int, default=512)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    student = load_student(run_dir / "student_best.pt")
    fp32_path = run_dir / "model_fp32.onnx"
    export_onnx(student, fp32_path)

    # --- parity: PyTorch vs ORT on the same random spectrograms
    sess_fp32 = ort.InferenceSession(str(fp32_path),
                                     providers=["CPUExecutionProvider"])
    spec = torch.randn(8, 1, N_MELS, N_FRAMES)
    with torch.no_grad():
        torch_logits = student.spectrogram_forward(spec).numpy()
    ort_logits = sess_fp32.run(None, {"spec": spec.numpy()})[0]
    parity = float(np.abs(torch_logits - ort_logits).max())
    assert parity < 1e-3, f"ONNX export diverges from PyTorch: {parity}"

    # --- dynamic PTQ (weights-only int8)
    dyn_path = run_dir / "model_int8_dynamic.onnx"
    quantize_dynamic(str(fp32_path), str(dyn_path), weight_type=QuantType.QInt8)

    # --- static PTQ with per-channel weights + observer-calibrated activations
    train_ds = PackedSpeechCommands(args.packs_dir, "training")
    calib_waves = torch.stack(
        [train_ds[i][0] for i in range(0, len(train_ds),
                                       max(1, len(train_ds) // args.calib_clips))])
    calib_specs = make_spectrograms(student, calib_waves)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.batches = iter(np.array_split(calib_specs, 16))

        def get_next(self):
            batch = next(self.batches, None)
            return None if batch is None else {"spec": batch}

    static_path = run_dir / "model_int8_static.onnx"
    quantize_static(str(fp32_path), str(static_path), Reader(),
                    per_channel=True, weight_type=QuantType.QInt8)

    # --- accuracy + latency for every variant
    report: dict = {"parity_max_abs_delta_logit": parity}
    one_clip = calib_specs[:1]
    for name, path in [("fp32", fp32_path), ("int8_dynamic", dyn_path),
                       ("int8_static", static_path)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        report[name] = {
            "size_kb": round(path.stat().st_size / 1024, 1),
            "test_accuracy": round(ort_accuracy(sess, student, args.packs_dir), 5),
            **ort_latency(sess, one_clip),
        }
        print(f"{name}: {report[name]}")

    (run_dir / "deploy.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run_dir}/deploy.json")


if __name__ == "__main__":
    main()
