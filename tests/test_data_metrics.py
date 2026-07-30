"""Dataset padding/subsetting invariants and metric sanity (no downloads)."""

import numpy as np
import torch

from phase0 import metrics
from phase0.data import CLIP_SAMPLES, WaveformAugment, _fix_length, stratified_subset


def test_fix_length_pads_and_trims():
    assert _fix_length(torch.ones(1000)).shape == (CLIP_SAMPLES,)
    assert _fix_length(torch.ones(20000)).shape == (CLIP_SAMPLES,)
    padded = _fix_length(torch.ones(1000))
    assert padded[1000:].abs().sum() == 0


def test_stratified_subset_is_deterministic_and_balanced():
    labels = np.repeat(np.arange(10), 100)
    a = stratified_subset(labels, 0.1, seed=7)
    b = stratified_subset(labels, 0.1, seed=7)
    assert np.array_equal(a, b)
    assert len(a) == 100
    counts = np.bincount(labels[a], minlength=10)
    assert counts.min() == counts.max() == 10  # class balance preserved


def test_waveform_augment_preserves_shape_and_range():
    import random
    random.seed(0)
    aug = WaveformAugment(noise_dir=None)  # shift-only
    out = aug(torch.rand(CLIP_SAMPLES) * 2 - 1)
    assert out.shape == (CLIP_SAMPLES,)
    assert out.abs().max() <= 1.0


def test_ece_zero_for_perfectly_calibrated_confident_model():
    labels = torch.arange(10)
    probs = torch.nn.functional.one_hot(labels, 10).float() * 0.999 + 1e-4
    assert metrics.ece(probs, labels) < 0.01


def test_agreement_and_kl_at_identity():
    logits = torch.randn(32, 35)
    assert metrics.agreement(logits, logits) == 1.0
    assert metrics.mean_kl(logits, logits) < 1e-6


def test_confusion_matrix_totals():
    preds = torch.tensor([0, 1, 1, 2])
    labels = torch.tensor([0, 1, 2, 2])
    m = metrics.confusion_matrix(preds, labels, 3)
    assert m.sum() == 4 and m[2, 1] == 1 and m[2, 2] == 1
