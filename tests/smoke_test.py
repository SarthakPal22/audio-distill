"""Fast sanity check with NO downloads: verifies shapes, the loss math,
and that a few gradient steps actually reduce the distillation loss on
synthetic data. Run from the repo root:

  python tests/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from phase0.data import CLIP_SAMPLES, NUM_CLASSES
from phase0.distill import distillation_loss
from phase0.student import StudentCNN


def main():
    torch.manual_seed(0)
    batch = 8

    student = StudentCNN()
    waveforms = torch.randn(batch, CLIP_SAMPLES)
    logits = student(waveforms)
    assert logits.shape == (batch, NUM_CLASSES), logits.shape
    print(f"forward pass OK: {tuple(waveforms.shape)} -> {tuple(logits.shape)}")
    print(f"student parameters: {student.num_parameters():,}")

    # fake teacher: random-but-fixed logits that agree with the labels,
    # so the loss has something learnable to move toward
    labels = torch.randint(0, NUM_CLASSES, (batch,))
    teacher_logits = torch.randn(batch, NUM_CLASSES)
    teacher_logits[torch.arange(batch), labels] += 5.0

    loss = distillation_loss(logits, teacher_logits, labels, alpha=0.3, temperature=4.0)
    assert torch.isfinite(loss), "loss is not finite"
    baseline = distillation_loss(logits, teacher_logits, labels, alpha=1.0, temperature=4.0)
    assert torch.isfinite(baseline)
    print(f"loss OK: distill={loss.item():.4f}, hard-only baseline={baseline.item():.4f}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    first_loss = None
    for step in range(30):
        out = student(waveforms)
        loss = distillation_loss(out, teacher_logits, labels, alpha=0.3, temperature=4.0)
        if first_loss is None:
            first_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first_loss, (first_loss, loss.item())
    print(f"training step OK: loss {first_loss:.4f} -> {loss.item():.4f} over 30 steps")
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
