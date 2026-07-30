"""Dataset utilities for Google Speech Commands v2.

The dataset: ~105,000 one-second audio clips of people saying one of 35
short words ("yes", "no", "left", "stop", ...). The task is to classify
which word was said. It's the "MNIST of audio" — small enough to train on
a laptop, real enough to learn from.

Two dataset classes live here:

- `KeywordDataset` — reads individual WAV files via torchaudio's split
  walker. Used once, by `cache_teacher.py`, to build the packs below.
  (We decode with `soundfile` because torchaudio 2.9+ delegates decoding
  to the separate torchcodec/FFmpeg stack.)
- `PackedSpeechCommands` — reads a **memory-mapped pack**: one big
  `.npy` file per split holding every waveform as int16 PCM, plus the
  teacher's cached fp16 logits and embeddings. Memory-mapping means the
  OS pages audio in on demand — no 85k `open()` calls per epoch — which
  makes the student's epoch time bound by compute, not disk.

Augmentation (train time only, applied to the raw waveform):

- **Time shift**: roll the clip by up to ±100 ms. Keyword onsets are not
  aligned in real usage; this enforces (approximate) shift invariance.
- **Background noise injection**: mix in a random crop of the dataset's
  `_background_noise_` recordings at a random low gain — the standard
  Speech Commands recipe (p=0.8, gain ~ U(0, 0.1)).

The KD-specific subtlety: cached teacher logits were computed on *clean*
audio. Training the student on *augmented* audio against clean-cached
soft labels ("offline distillation") introduces teacher-label staleness;
recomputing teacher logits on the augmented batch every step ("online
distillation") is exact but pays the teacher's forward cost. We support
both to measure the gap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000          # samples per second of audio
CLIP_SAMPLES = SAMPLE_RATE    # every clip is padded/trimmed to exactly 1 second

# The 35 words in Speech Commands v2, in a fixed order. Index in this list
# is the class id used everywhere in this project.
LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes",
    "zero",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Acoustically confusable pairs — the words a small model mixes up first.
# Soft labels carry exactly this inter-class similarity structure.
CONFUSABLE_PAIRS = [("go", "no"), ("three", "tree"), ("forward", "four"),
                    ("bird", "bed"), ("off", "up")]


def _fix_length(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.numel() < CLIP_SAMPLES:
        return torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - waveform.numel()))
    return waveform[:CLIP_SAMPLES]


class KeywordDataset(Dataset):
    """File-reading dataset (slow path). Used to build packs and as a
    fallback when packs don't exist. subset: "training"|"validation"|"testing"."""

    def __init__(self, data_dir: str, subset: str):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=True, subset=subset
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        relpath, sample_rate, label, _speaker, _utt = self.dataset.get_metadata(idx)
        assert sample_rate == SAMPLE_RATE, f"unexpected sample rate {sample_rate}"
        audio, _ = soundfile.read(
            Path(self.dataset._archive) / relpath, dtype="float32"
        )
        return _fix_length(torch.from_numpy(audio)), LABEL_TO_INDEX[label]


class WaveformAugment:
    """Time shift + background-noise injection on int16-scale float waveforms.

    Uses Python's `random` module so DataLoader worker seeding (see
    utils.seed_worker) makes augmentation draws reproducible per run.
    """

    def __init__(self, noise_dir: Path | None, shift_ms: int = 100,
                 noise_prob: float = 0.8, noise_gain: float = 0.1):
        self.max_shift = SAMPLE_RATE * shift_ms // 1000
        self.noise_prob = noise_prob
        self.noise_gain = noise_gain
        self.noises: list[np.ndarray] = []
        if noise_dir is not None and noise_dir.exists():
            for wav in sorted(noise_dir.glob("*.wav")):
                audio, sr = soundfile.read(wav, dtype="float32")
                assert sr == SAMPLE_RATE
                self.noises.append(audio)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift != 0:
            waveform = torch.roll(waveform, shift)
            if shift > 0:
                waveform[:shift] = 0.0
            else:
                waveform[shift:] = 0.0
        if self.noises and random.random() < self.noise_prob:
            noise = random.choice(self.noises)
            start = random.randint(0, len(noise) - CLIP_SAMPLES)
            gain = random.uniform(0.0, self.noise_gain)
            waveform = waveform + gain * torch.from_numpy(
                noise[start:start + CLIP_SAMPLES].copy()
            )
        return waveform.clamp(-1.0, 1.0)


class PackedSpeechCommands(Dataset):
    """Memory-mapped pack: waveforms (int16), labels, cached teacher
    logits (fp16) and embeddings (fp16), all index-aligned.

    Returns (waveform fp32, label, teacher_logits fp32, teacher_embedding
    fp32). Logit/embedding tensors are zero-size if the pack was built
    without them.
    """

    def __init__(self, packs_dir: str | Path, split: str,
                 augment: WaveformAugment | None = None,
                 indices: np.ndarray | None = None):
        root = Path(packs_dir) / split
        self.meta = json.loads((root / "meta.json").read_text())
        self.waveforms = np.load(root / "waveforms.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy")
        logits_path = root / "teacher_logits.npy"
        emb_path = root / "teacher_emb.npy"
        self.teacher_logits = np.load(logits_path, mmap_mode="r") if logits_path.exists() else None
        self.teacher_emb = np.load(emb_path, mmap_mode="r") if emb_path.exists() else None
        self.augment = augment
        self.indices = np.arange(len(self.labels)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # int16 PCM -> float in [-1, 1], same scaling soundfile uses
        waveform = torch.from_numpy(
            self.waveforms[idx].astype(np.float32) / 32768.0
        )
        if self.augment is not None:
            waveform = self.augment(waveform)
        label = int(self.labels[idx])
        logits = (torch.from_numpy(self.teacher_logits[idx].astype(np.float32))
                  if self.teacher_logits is not None else torch.empty(0))
        emb = (torch.from_numpy(self.teacher_emb[idx].astype(np.float32))
               if self.teacher_emb is not None else torch.empty(0))
        return waveform, label, logits, emb


def stratified_subset(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Pick `fraction` of indices, preserving the class distribution
    (**stratified sampling**), deterministically per seed. This is how the
    low-resource ablation shrinks the labeled training set without also
    changing the class balance."""
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for c in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        keep = max(1, round(len(idx) * fraction))
        chosen.append(idx[:keep])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def noise_dir_for(data_dir: str) -> Path:
    return (Path(data_dir) / "SpeechCommands" / "speech_commands_v0.02"
            / "_background_noise_")
