"""Generate every figure in docs/figures/ from runs/ artifacts.

Figures (each skipped gracefully if its runs are missing):
  alpha_sweep.png       accuracy vs hard-label weight alpha
  temperature_sweep.png accuracy vs softmax temperature T
  low_resource.png      distillation gap vs training-set fraction
  pareto.png            accuracy vs MACs frontier across architectures/widths
  reliability.png       reliability diagrams: baseline vs distilled vs teacher
  confusion.png         distilled-student confusion matrix (log color scale)
  training_curves.png   validation accuracy per epoch, baseline vs distilled

Usage:  python -m tools.plots
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase0.data import LABELS
from tools.aggregate import load_rows

FIG_DIR = Path("docs/figures")


def grouped(rows, fixed: dict, x_key: str, y_key: str = "test_accuracy"):
    """mean/std of y over seeds, for rows matching `fixed`, keyed by x."""
    buckets = defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in fixed.items()) and r[y_key] is not None:
            buckets[r[x_key]].append(r[y_key])
    xs = sorted(buckets)
    means = [statistics.mean(buckets[x]) for x in xs]
    stds = [statistics.stdev(buckets[x]) if len(buckets[x]) > 1 else 0 for x in xs]
    return xs, means, stds


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


BASE = {"arch": "cnn", "width": 1.0, "feature_beta": 0.0,
        "augment": "none", "fraction": 1.0}


def alpha_sweep(rows):
    xs, means, stds = grouped(rows, {**BASE, "temperature": 4.0}, "alpha")
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(xs, [m * 100 for m in means], yerr=[s * 100 for s in stds],
                marker="o", capsize=3)
    ax.set_xlabel(r"hard-label weight $\alpha$  (1.0 = no distillation)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Hard/soft loss blend (T=4)")
    ax.grid(alpha=0.3)
    save(fig, "alpha_sweep.png")


def temperature_sweep(rows):
    xs, means, stds = grouped(rows, {**BASE, "alpha": 0.3}, "temperature")
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(xs, [m * 100 for m in means], yerr=[s * 100 for s in stds],
                marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("temperature T")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(r"Posterior softening ($\alpha$=0.3)")
    ax.grid(alpha=0.3)
    save(fig, "temperature_sweep.png")


def low_resource(rows):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    plotted = 0
    for alpha, label, style in [(1.0, "baseline (hard labels)", "--o"),
                                (0.3, "distilled (T=4)", "-o")]:
        pts = grouped(rows, {**BASE, "alpha": alpha, "temperature": 4.0},
                      "fraction")
        # fraction=1.0 runs share BASE; include them
        if len(pts[0]) < 2:
            continue
        xs, means, stds = pts
        ax.errorbar([x * 100 for x in xs], [m * 100 for m in means],
                    yerr=[s * 100 for s in stds], fmt=style, capsize=3,
                    label=label)
        plotted += 1
    if plotted < 2:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("labeled training data (%)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Dark knowledge as a regularizer")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "low_resource.png")


def pareto(rows):
    pts = [(r["macs"] / 1e6, r["test_accuracy"] * 100,
            f'{r["arch"]} w={r["width"]:g}')
           for r in rows
           if r["alpha"] == 0.3 and r["fraction"] == 1.0
           and r["feature_beta"] == 0 and r["augment"] == "none"
           and r["seed"] == 0 and r["test_accuracy"] is not None]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for macs, acc, label in pts:
        ax.scatter(macs, acc, s=45)
        ax.annotate(label, (macs, acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("MACs per inference (millions, log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs compute (distilled students)")
    ax.grid(alpha=0.3)
    save(fig, "pareto.png")


def reliability(rows):
    candidates = {"baseline": {**BASE, "alpha": 1.0},
                  "distilled": {**BASE, "alpha": 0.3, "temperature": 4.0}}
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    plotted = 0
    for name, fixed in candidates.items():
        for r in rows:
            if all(r[k] == v for k, v in fixed.items()) and r["seed"] == 0:
                eval_path = Path("runs") / r["run"] / "eval.json"
                if not eval_path.exists():
                    break
                rel = json.loads(eval_path.read_text())["reliability"]
                ax.plot(rel["confidence"], rel["accuracy"], marker="o", ms=3,
                        label=f'{name} (ECE {json.loads(eval_path.read_text())["ece"]:.3f})')
                plotted += 1
                break
    if plotted < 2:
        plt.close(fig)
        return
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)
    save(fig, "reliability.png")


def confusion(rows):
    for r in rows:
        if (r["alpha"] == 0.3 and all(r[k] == v for k, v in BASE.items())
                and r["seed"] == 0):
            path = Path("runs") / r["run"] / "confusion.npy"
            if not path.exists():
                return
            m = np.load(path).astype(float)
            m_norm = m / m.sum(axis=1, keepdims=True)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(np.log10(m + 1), cmap="viridis")
            ax.set_xticks(range(35), LABELS, rotation=90, fontsize=6)
            ax.set_yticks(range(35), LABELS, fontsize=6)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title("Distilled student confusion (log10 counts)")
            fig.colorbar(im, shrink=0.8)
            save(fig, "confusion.png")
            # print the top confusions for the README error analysis
            np.fill_diagonal(m_norm, 0)
            top = np.dstack(np.unravel_index(np.argsort(m_norm, axis=None)[::-1][:8],
                                             m_norm.shape))[0]
            for t, p in top:
                print(f"  confusion: {LABELS[t]} -> {LABELS[p]} "
                      f"({m_norm[t, p]:.1%} of true '{LABELS[t]}')")
            return


def training_curves(rows):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    plotted = 0
    for alpha, label in [(1.0, "baseline"), (0.3, "distilled (T=4)")]:
        for r in rows:
            if (r["alpha"] == alpha and r["seed"] == 0
                    and all(r[k] == v for k, v in BASE.items())
                    and (alpha == 1.0 or r["temperature"] == 4.0)):
                path = Path("runs") / r["run"] / "epochs.csv"
                if not path.exists():
                    break
                with open(path) as f:
                    epochs = list(csv.DictReader(f))
                ax.plot([int(e["epoch"]) for e in epochs],
                        [float(e["val_accuracy"]) * 100 for e in epochs],
                        marker="o", ms=3, label=label)
                plotted += 1
                break
    if plotted < 2:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Soft labels accelerate early training")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "training_curves.png")


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no runs found")
        return
    for fn in (alpha_sweep, temperature_sweep, low_resource, pareto,
               reliability, confusion, training_curves):
        fn(rows)


if __name__ == "__main__":
    main()
