"""The student: a tiny CNN (~0.2M parameters, ~400x smaller than the teacher).

Pipeline inside the model:
  raw waveform (16000 samples)
    -> mel spectrogram: a 2D "image" of the sound, time on the x-axis,
       pitch (mel-scaled frequency) on the y-axis
    -> log scale (loudness perception is logarithmic)
    -> a stack of small convolution blocks, exactly like an image classifier
    -> average over time/frequency -> linear layer -> 35 logits

Keeping the spectrogram computation *inside* the model means callers just
pass raw audio, same as the teacher.
"""

import torch
import torch.nn as nn
import torchaudio

from phase0.data import NUM_CLASSES, SAMPLE_RATE


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # halve time and frequency resolution
    )


class StudentCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, n_mels: int = 64):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,        # 25 ms analysis window
            hop_length=160,   # one spectrogram column every 10 ms
            n_mels=n_mels,    # 64 frequency bands
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms: (batch, 16000) -> logits (batch, num_classes)"""
        x = self.to_db(self.melspec(waveforms))   # (batch, 64 mels, 101 frames)
        x = x.unsqueeze(1)                        # add channel dim like a grayscale image
        x = self.features(x)                      # (batch, 128, 4, 6)
        x = x.mean(dim=(2, 3))                    # global average pool -> (batch, 128)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
