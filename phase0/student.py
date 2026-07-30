"""Student architectures.

Every student is Frontend -> SpectrogramNet:

- `MelFrontend` turns the raw waveform into a log-mel spectrogram — a 2D
  time-frequency "image" (64 mel bands x 101 frames; 25 ms window, 10 ms
  hop) — and optionally applies **SpecAugment** during training:
  random time masks and frequency masks zeroed directly on the
  spectrogram, a strong regularizer that forces the network not to rely
  on any single band or instant.
- A spectrogram classifier ("SpectrogramNet") maps that image to 35
  logits. Keeping this half separate matters downstream: it is the part
  we export to ONNX and quantize, while the FFT front-end stays outside
  the graph.

Architectures:

- `StudentCNN` — plain VGG-style conv blocks. Simple, the pedagogical
  default.
- `DSCNN` — a **depthwise-separable CNN**, the standard efficient
  keyword-spotting architecture (Zhang et al. 2017, "Hello Edge").
  A depthwise-separable convolution factorizes a full KxK convolution
  into a per-channel spatial KxK ("depthwise") convolution followed by a
  1x1 cross-channel ("pointwise") convolution, cutting MACs by roughly
  the kernel area at equal accuracy for small models.

Both take a **width multiplier**: a scalar that scales every channel
count, tracing an accuracy-vs-MACs curve with one hyperparameter. Points
on that curve where no model is both smaller *and* more accurate form the
**Pareto frontier**.

For feature-based distillation, both expose `self.embedding_dim` and
return their penultimate global-average-pooled feature vector via
`forward(..., return_embedding=True)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from phase0.data import NUM_CLASSES, SAMPLE_RATE

N_MELS = 64
N_FRAMES = 101  # 1 s of audio at a 10 ms hop


class MelFrontend(nn.Module):
    """waveform (B, 16000) -> log-mel spectrogram (B, 1, 64, 101),
    with SpecAugment masking applied only in training mode."""

    def __init__(self, spec_augment: bool = False,
                 freq_mask: int = 8, time_mask: int = 20):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,        # 25 ms analysis window
            hop_length=160,   # one spectrogram column every 10 ms
            n_mels=N_MELS,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_augment = spec_augment
        self.freq_masker = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_masker = torchaudio.transforms.TimeMasking(time_mask)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = self.to_db(self.melspec(waveforms)).unsqueeze(1)
        if self.spec_augment and self.training:
            x = self.time_masker(self.freq_masker(x))
        return x


def _scaled(channels: int, width: float) -> int:
    return max(8, int(round(channels * width)))


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class _DSBlock(nn.Sequential):
    """Depthwise 3x3 (groups=channels) then pointwise 1x1. Stride on the
    depthwise conv downsamples the grid, which is where DS-CNNs bank
    their MAC savings."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _SpectrogramStudent(nn.Module):
    """Shared skeleton: spectrogram body + GAP + linear classifier, with a
    waveform front-end bolted on. Subclasses define `body` and `feat_dim`."""

    def __init__(self, body: nn.Module, feat_dim: int,
                 num_classes: int, spec_augment: bool):
        super().__init__()
        self.frontend = MelFrontend(spec_augment=spec_augment)
        self.body = body
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.embedding_dim = feat_dim

    def spectrogram_forward(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, 1, 64, 101) -> logits. This is the ONNX-exported graph."""
        return self.classifier(self.body(spec).mean(dim=(2, 3)))

    def forward(self, waveforms: torch.Tensor, return_embedding: bool = False):
        spec = self.frontend(waveforms)
        embedding = self.body(spec).mean(dim=(2, 3))  # global average pool
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class StudentCNN(_SpectrogramStudent):
    def __init__(self, num_classes: int = NUM_CLASSES, width: float = 1.0,
                 spec_augment: bool = False):
        chans = [_scaled(c, width) for c in (16, 32, 64, 128)]
        body = nn.Sequential(
            _ConvBlock(1, chans[0]),
            _ConvBlock(chans[0], chans[1]),
            _ConvBlock(chans[1], chans[2]),
            _ConvBlock(chans[2], chans[3]),
        )
        super().__init__(body, chans[3], num_classes, spec_augment)


class DSCNN(_SpectrogramStudent):
    def __init__(self, num_classes: int = NUM_CLASSES, width: float = 1.0,
                 spec_augment: bool = False):
        ch = _scaled(64, width)
        body = nn.Sequential(
            # aggressive strided stem: cuts the time-frequency grid early,
            # which is where most of the MAC savings come from
            nn.Conv2d(1, ch, kernel_size=(10, 4), stride=2, padding=(4, 1),
                      bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            _DSBlock(ch, ch),
            _DSBlock(ch, ch, stride=2),
            _DSBlock(ch, ch, stride=2),
            _DSBlock(ch, ch),
        )
        super().__init__(body, ch, num_classes, spec_augment)


ARCHITECTURES = {"cnn": StudentCNN, "dscnn": DSCNN}


def build_student(arch: str, width: float = 1.0,
                  spec_augment: bool = False) -> _SpectrogramStudent:
    return ARCHITECTURES[arch](width=width, spec_augment=spec_augment)
