"""Shared experiment infrastructure: seeding, device selection, run
manifests, and MAC counting.

Reproducibility vocabulary:

- **Seed**: the initial state of a pseudo-random number generator (RNG).
  Fixing seeds for every RNG that touches the experiment (weight init,
  data shuffling, augmentation sampling, dropout) makes runs repeatable,
  which is what lets us report mean +/- std over seeds instead of a
  single cherry-picked number.
- **Run manifest**: a JSON record written next to every checkpoint
  containing the exact git commit, CLI arguments, seed, and final
  metrics. Any number in the README can be traced back to the manifest
  (and therefore the code state) that produced it.
- **MACs (multiply-accumulate operations)**: the standard compute-cost
  metric for CNNs; one MAC = one multiply + one add. Parameter count
  measures model *size*; MACs measure *work per inference*. Two models
  with equal parameters can differ wildly in MACs, so we report both.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Seed every RNG in play. Returns a torch.Generator for DataLoader
    shuffling so that data order is decoupled from model-init randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA, and MPS generators
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: gives each background loading process a
    deterministic, distinct seed (derived from the base seed by PyTorch)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")     # NVIDIA GPU (Colab/Kaggle)
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon GPU
    return torch.device("cpu")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"git_sha": git_sha(), **payload}
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2))


def count_macs(model: torch.nn.Module, example_input: torch.Tensor) -> int:
    """Count multiply-accumulates via forward hooks on Conv2d/Linear.

    Dependency-free by design. The mel-spectrogram front-end (an FFT, not
    a matmul) is excluded; it is identical for every student, so it
    cancels out of all comparisons.
    """
    total = 0

    def conv_hook(module: torch.nn.Conv2d, inputs, output) -> None:
        nonlocal total
        out_h, out_w = output.shape[-2:]
        kh, kw = module.kernel_size
        total += (
            module.out_channels * out_h * out_w
            * (module.in_channels // module.groups) * kh * kw
        )

    def linear_hook(module: torch.nn.Linear, inputs, output) -> None:
        nonlocal total
        total += module.in_features * module.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(example_input)
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return total
