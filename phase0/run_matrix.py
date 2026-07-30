"""Sequential experiment-matrix runner.

Each experiment states its hypothesis in the group comment. Runs are
subprocesses (crash isolation); a run whose manifest.json already exists
is skipped, so the matrix is resumable. Groups ordered so the core
results land first.

Design note on the augmentation study: online distillation costs a
teacher forward per step (~840x the student), so the offline-vs-online
comparison runs on the 25% subset, where (a) wall clock is tractable and
(b) augmentation matters most. Low-resource runs get proportionally more
epochs (capped) to keep the optimization budget roughly constant.

Usage:  python -m phase0.run_matrix [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [0, 1]


def fraction_epochs(fraction: float, base: int = 8, cap: int = 60) -> int:
    return min(cap, round(base / fraction))


def build_configs() -> list[list[str]]:
    configs: list[list[str]] = []

    # A. Alpha sweep (T=4). Hypothesis: blending hard + soft labels beats
    # either alone; alpha=1.0 is the no-distillation baseline.
    for alpha in [0.0, 0.1, 0.3, 0.5, 1.0]:
        for seed in SEEDS:
            configs.append([f"--alpha={alpha}", "--temperature=4.0", f"--seed={seed}"])

    # B. Temperature sweep (alpha=0.3). Hypothesis: T>1 exposes dark
    # knowledge; T=1 under-softens, very large T over-flattens.
    for temp in [1.0, 2.0, 8.0]:
        for seed in SEEDS:
            configs.append([f"--alpha=0.3", f"--temperature={temp}", f"--seed={seed}"])

    # C. Feature-based KD. Hypothesis: representation alignment via a
    # projection head adds signal beyond logit matching.
    for seed in SEEDS:
        configs.append(["--alpha=0.3", "--temperature=4.0",
                        "--feature-beta=1.0", f"--seed={seed}"])

    # D. Architecture/width Pareto frontier (accuracy vs MACs), distilled.
    for arch, width in [("dscnn", 0.25), ("dscnn", 0.5), ("dscnn", 1.0),
                        ("cnn", 0.25), ("cnn", 0.5)]:
        configs.append([f"--arch={arch}", f"--width={width}",
                        "--alpha=0.3", "--temperature=4.0", "--seed=0"])
    configs.append(["--arch=dscnn", "--width=1.0", "--alpha=1.0", "--seed=0"])

    # E. Low-resource ablation. Hypothesis: the distillation gap widens as
    # labeled data shrinks (soft labels act as a similarity-structured
    # regularizer).
    for fraction in [0.25, 0.10, 0.02]:
        for alpha in [1.0, 0.3]:
            for seed in SEEDS:
                configs.append([f"--fraction={fraction}", f"--alpha={alpha}",
                                "--temperature=4.0", f"--seed={seed}",
                                f"--epochs={fraction_epochs(fraction)}"])

    # F. Augmentation x distillation. Hypothesis: offline (clean-cached)
    # soft labels stay useful under input augmentation despite staleness;
    # online recomputation buys little at large cost.
    for seed in SEEDS:
        configs.append(["--alpha=0.3", "--temperature=4.0",
                        "--augment=offline", f"--seed={seed}"])
    configs.append(["--alpha=0.3", "--temperature=4.0", "--augment=offline",
                    "--fraction=0.25", "--epochs=32", "--seed=0"])
    configs.append(["--alpha=0.3", "--temperature=4.0", "--augment=online",
                    "--fraction=0.25", "--epochs=32", "--seed=0"])
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = build_configs()
    print(f"{len(configs)} runs in matrix")
    for i, cfg in enumerate(configs, 1):
        # reconstruct the run name the same way distill.py does, to skip
        # completed runs (makes the matrix resumable after interruption)
        defaults = {"arch": "cnn", "width": 1.0, "alpha": 0.3,
                    "temperature": 4.0, "feature_beta": 0.0, "augment": "none",
                    "fraction": 1.0, "seed": 0}
        for flag in cfg:
            key, value = flag.lstrip("-").split("=")
            key = key.replace("-", "_")
            if key in defaults:
                defaults[key] = type(defaults[key])(value)
        ns = argparse.Namespace(**defaults)
        from phase0.distill import run_name
        manifest = Path("runs") / run_name(ns) / "manifest.json"
        if manifest.exists():
            print(f"[{i}/{len(configs)}] skip (done): {' '.join(cfg)}", flush=True)
            continue
        print(f"[{i}/{len(configs)}] {' '.join(cfg)}", flush=True)
        if args.dry_run:
            continue
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "phase0.distill", *cfg],
            capture_output=True, text=True)
        tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        status = "ok" if result.returncode == 0 else "FAILED"
        print(f"    {status} in {(time.time() - start) / 60:.1f} min — {tail}",
              flush=True)
        if result.returncode != 0:
            Path("matrix_failures.log").open("a").write(
                f"\n=== {' '.join(cfg)} ===\n{result.stderr[-3000:]}\n")


if __name__ == "__main__":
    main()
