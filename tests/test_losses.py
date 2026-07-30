"""Correctness properties of the distillation objective (no downloads)."""

import torch
import torch.nn.functional as F

from phase0.distill import distillation_loss, feature_loss

B, C = 16, 35


def test_alpha_one_reduces_to_cross_entropy():
    torch.manual_seed(0)
    s, t = torch.randn(B, C), torch.randn(B, C)
    y = torch.randint(0, C, (B,))
    assert torch.isclose(distillation_loss(s, t, y, alpha=1.0, temperature=4.0),
                         F.cross_entropy(s, y))


def test_soft_term_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    t = torch.randn(B, C)
    y = torch.randint(0, C, (B,))
    # KL(p || p) = 0, so at alpha=0 the loss must vanish
    assert distillation_loss(t.clone(), t, y, alpha=0.0, temperature=4.0).abs() < 1e-5


def test_loss_blends_linearly_in_alpha():
    torch.manual_seed(0)
    s, t = torch.randn(B, C), torch.randn(B, C)
    y = torch.randint(0, C, (B,))
    hard = distillation_loss(s, t, y, alpha=1.0, temperature=4.0)
    soft = distillation_loss(s, t, y, alpha=0.0, temperature=4.0)
    mid = distillation_loss(s, t, y, alpha=0.3, temperature=4.0)
    assert torch.isclose(mid, 0.3 * hard + 0.7 * soft, atol=1e-5)


def test_temperature_squared_keeps_gradients_comparable():
    torch.manual_seed(0)
    y = torch.randint(0, C, (B,))
    t = torch.randn(B, C)
    grads = []
    for temp in (2.0, 8.0):
        s = torch.randn(B, C, requires_grad=True)
        distillation_loss(s, t.clone(), y, alpha=0.0, temperature=temp).backward()
        grads.append(s.grad.norm().item())
    # without the T^2 factor these would differ by ~(8/2)^2 = 16x
    assert 0.2 < grads[0] / grads[1] < 5.0


def test_feature_loss_zero_at_perfect_alignment():
    proj = torch.nn.Linear(8, 8, bias=False)
    torch.nn.init.eye_(proj.weight)
    emb = torch.randn(B, 8)
    assert feature_loss(emb, proj, emb).abs() < 1e-6
    assert feature_loss(emb, proj, torch.randn(B, 8)) > 0
