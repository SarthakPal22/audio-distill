"""Dataset utilities for Google Speech Commands v2.

The dataset: ~105,000 one-second audio clips of people saying one of 35
short words ("yes", "no", "left", "stop", ...). The task is to classify
which word was said. It's the "MNIST of audio" — small enough to train on
a laptop, real enough to learn from.

torchaudio downloads and extracts it automatically (~2.3 GB) on first use.
"""

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


class KeywordDataset(Dataset):
    """Wraps torchaudio's SPEECHCOMMANDS so every item is a fixed-size
    (waveform, class_index) pair, which lets PyTorch batch them together.

    subset: "training" | "validation" | "testing"
            (the official split shipped with the dataset)
    """

    def __init__(self, data_dir: str, subset: str):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=True, subset=subset
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        waveform, sample_rate, label, _speaker, _utt = self.dataset[idx]
        assert sample_rate == SAMPLE_RATE, f"unexpected sample rate {sample_rate}"

        # waveform arrives as shape (channels=1, num_samples). Drop the
        # channel dim and force the length to exactly 1 second: some clips
        # are slightly shorter, and batching requires equal lengths.
        waveform = waveform.squeeze(0)
        if waveform.numel() < CLIP_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, CLIP_SAMPLES - waveform.numel())
            )
        else:
            waveform = waveform[:CLIP_SAMPLES]

        return waveform, LABEL_TO_INDEX[label]
