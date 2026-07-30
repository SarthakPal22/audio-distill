"""Architecture invariants: shapes, widths, MACs, SpecAugment gating."""

import pytest
import torch

from phase0.data import CLIP_SAMPLES, NUM_CLASSES
from phase0.student import build_student
from phase0.utils import count_macs


@pytest.mark.parametrize("arch", ["cnn", "dscnn"])
@pytest.mark.parametrize("width", [0.25, 1.0])
def test_forward_shapes(arch, width):
    model = build_student(arch, width).eval()
    x = torch.randn(4, CLIP_SAMPLES)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (4, NUM_CLASSES)
    assert emb.shape == (4, model.embedding_dim)


def test_width_scales_parameters():
    small = build_student("cnn", 0.25).num_parameters()
    large = build_student("cnn", 1.0).num_parameters()
    assert small < large / 4


def test_dscnn_cheaper_than_cnn_at_equal_width():
    x = torch.randn(1, CLIP_SAMPLES)
    assert (count_macs(build_student("dscnn", 1.0), x)
            < count_macs(build_student("cnn", 1.0), x))


def test_spec_augment_only_in_training_mode():
    torch.manual_seed(0)
    model = build_student("cnn", spec_augment=True)
    x = torch.randn(2, CLIP_SAMPLES)
    model.eval()
    assert torch.equal(model.frontend(x), model.frontend(x))  # deterministic
    model.train()
    specs = model.frontend(x)
    assert (specs == 0).any()  # masks applied


def test_spectrogram_forward_matches_full_forward():
    model = build_student("cnn").eval()
    x = torch.randn(2, CLIP_SAMPLES)
    with torch.no_grad():
        assert torch.allclose(model(x), model.spectrogram_forward(model.frontend(x)),
                              atol=1e-5)
