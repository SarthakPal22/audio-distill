"""Deep evaluation of one trained student checkpoint.

Reports, on the held-out test split:

- top-1 accuracy and NLL (negative log-likelihood)
- **expected calibration error** (ECE) + reliability-diagram bins
- **teacher agreement** and **mean KL to teacher** (fidelity metrics),
  read from the cached teacher logits — the teacher never runs here
- the full **confusion matrix** (saved as .npy for plotting)
- single-clip CPU latency: median and p99 over 200 timed runs after 20
  warm-up runs (warm-up matters: the first calls pay one-time costs like
  memory-pool growth and kernel autotuning that say nothing about
  steady-state latency)

Everything is written to <run_dir>/eval.json + confusion.npy so plots and
tables can be regenerated without re-running inference.

Usage:
  python -m phase0.evaluate --run-dir runs/cnn1_a0.3_T4_s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase0 import metrics
from phase0.data import CLIP_SAMPLES, NUM_CLASSES, PackedSpeechCommands
from phase0.student import build_student
from phase0.utils import count_macs, pick_device


def load_student(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    student = build_student(ckpt.get("arch", "cnn"), ckpt.get("width", 1.0))
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    return student


@torch.no_grad()
def cpu_latency_ms(model: torch.nn.Module, runs: int = 200,
                   warmup: int = 20) -> dict[str, float]:
    clip = torch.randn(1, CLIP_SAMPLES)
    for _ in range(warmup):
        model(clip)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(clip)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": times[len(times) // 2],
            "p99_ms": times[int(len(times) * 0.99)]}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--packs-dir", default="data/packs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = pick_device(args.device)
    student = load_student(run_dir / "student_best.pt").to(device)

    test_ds = PackedSpeechCommands(args.packs_dir, "testing")
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    all_logits, all_labels, all_teacher = [], [], []
    for waveforms, labels, t_logits, _ in tqdm(loader, desc="test set"):
        all_logits.append(student(waveforms.to(device)).cpu())
        all_labels.append(labels)
        if t_logits.numel():
            all_teacher.append(t_logits)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    probs = logits.softmax(-1)
    preds = logits.argmax(-1)

    report: dict = {
        "accuracy": (preds == labels).float().mean().item(),
        "nll": F.cross_entropy(logits, labels).item(),
        "ece": metrics.ece(probs, labels),
        "reliability": metrics.reliability_bins(probs, labels),
        "params": student.num_parameters(),
        "macs": count_macs(student, torch.randn(1, CLIP_SAMPLES).to(device)),
    }
    if all_teacher:
        teacher_logits = torch.cat(all_teacher)
        report["teacher_accuracy"] = (
            teacher_logits.argmax(-1) == labels).float().mean().item()
        report["teacher_ece"] = metrics.ece(teacher_logits.softmax(-1), labels)
        report["teacher_agreement"] = metrics.agreement(logits, teacher_logits)
        report["mean_kl_to_teacher"] = metrics.mean_kl(logits, teacher_logits)

    report["cpu_latency"] = cpu_latency_ms(student.cpu())
    np.save(run_dir / "confusion.npy",
            metrics.confusion_matrix(preds, labels, NUM_CLASSES))
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))

    print(f"\naccuracy {report['accuracy']:.2%} | ECE {report['ece']:.4f} | "
          f"latency {report['cpu_latency']['median_ms']:.2f}ms (median)")
    if all_teacher:
        print(f"teacher accuracy {report['teacher_accuracy']:.2%} | "
              f"agreement {report['teacher_agreement']:.2%} | "
              f"mean KL {report['mean_kl_to_teacher']:.4f}")
    print(f"wrote {run_dir}/eval.json and confusion.npy")


if __name__ == "__main__":
    main()
