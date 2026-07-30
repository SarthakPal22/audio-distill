"""Offline distillation, step 1: run the frozen teacher over every split
ONCE and cache its outputs to disk.

Why: with **online distillation** the teacher's forward pass runs inside
the student's training loop, dominating step time (the teacher is ~840x
the student). With **offline distillation** we amortize that cost — one
teacher pass, reused by every subsequent run. This is the same
pseudo-labeling pattern Distil-Whisper uses at scale, and it's what makes
a 37-run experiment matrix tractable on a laptop.

What gets written per split (data/packs/{training,validation,testing}/):

- waveforms.npy        (N, 16000) int16   — raw PCM, memory-mapped at train time
- labels.npy           (N,)       int64   — ground-truth class ids
- teacher_logits.npy   (N, 35)    fp16    — pre-softmax scores ("soft labels")
- teacher_emb.npy      (N, 768)   fp16    — mean-pooled final hidden state,
                                            for feature-based distillation
- meta.json                                — provenance (teacher id, git SHA)

fp16 halves disk size; logits round-trip through fp16 with negligible
distortion relative to the softmax temperature scales we use.

Usage:
  python -m phase0.cache_teacher                  # all three splits
  python -m phase0.cache_teacher --splits testing --max-items 64  # smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase0.data import CLIP_SAMPLES, KeywordDataset, LABELS, SAMPLE_RATE
from phase0.teacher import MODEL_ID, Teacher
from phase0.utils import git_sha, pick_device


class _FeatureDataset(KeywordDataset):
    """Moves the teacher's CPU-side mel-filterbank extraction into
    DataLoader worker processes, so it overlaps with the GPU forward
    (producer/consumer parallelism). Each worker owns its extractor."""

    def __init__(self, data_dir: str, subset: str):
        super().__init__(data_dir, subset)
        from transformers import ASTFeatureExtractor
        self.extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)

    def __getitem__(self, idx: int):
        waveform, label = super().__getitem__(idx)
        features = self.extractor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_values[0]
        return features, waveform, label


def pack_split(split: str, data_dir: str, out_dir: Path, teacher: Teacher,
               batch_size: int, max_items: int | None) -> None:
    dataset = _FeatureDataset(data_dir, split)
    n = len(dataset) if max_items is None else min(max_items, len(dataset))
    out = out_dir / split
    out.mkdir(parents=True, exist_ok=True)

    waveforms = np.lib.format.open_memmap(
        out / "waveforms.npy", mode="w+", dtype=np.int16, shape=(n, CLIP_SAMPLES))
    labels = np.zeros(n, dtype=np.int64)
    logits = np.lib.format.open_memmap(
        out / "teacher_logits.npy", mode="w+", dtype=np.float16, shape=(n, len(LABELS)))
    embs = np.lib.format.open_memmap(
        out / "teacher_emb.npy", mode="w+", dtype=np.float16, shape=(n, 768))

    # workers parallelize WAV decoding + feature extraction; order must stay
    # deterministic (shuffle=False) so cached rows stay index-aligned
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=6, prefetch_factor=4)
    start, done = time.time(), 0
    for batch_features, batch_waveforms, batch_labels in loader:
        if done >= n:
            break
        take = min(len(batch_labels), n - done)
        batch_waveforms = batch_waveforms[:take]
        batch_logits, batch_emb = teacher.forward_features(batch_features[:take])
        sl = slice(done, done + take)
        waveforms[sl] = (batch_waveforms.numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
        labels[sl] = batch_labels[:take].numpy()
        logits[sl] = batch_logits.cpu().numpy().astype(np.float16)
        embs[sl] = batch_emb.cpu().numpy().astype(np.float16)
        done += take
        if done % (batch_size * 10) < batch_size:
            rate = done / (time.time() - start)
            print(f"  {split}: {done}/{n} clips ({rate:.0f} clips/s, "
                  f"eta {(n - done) / max(rate, 1):.0f}s)", flush=True)

    np.save(out / "labels.npy", labels)
    for arr in (waveforms, logits, embs):
        arr.flush()
    (out / "meta.json").write_text(json.dumps({
        "split": split, "num_items": n, "teacher": MODEL_ID,
        "labels": LABELS, "git_sha": git_sha(),
        "seconds": round(time.time() - start, 1),
    }, indent=2))
    print(f"{split}: packed {n} clips in {time.time() - start:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data/packs")
    parser.add_argument("--splits", nargs="+",
                        default=["training", "validation", "testing"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=None,
                        help="cap items per split, for smoke runs")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    teacher = Teacher(device)
    for split in args.splits:
        pack_split(split, args.data_dir, Path(args.out_dir), teacher,
                   args.batch_size, args.max_items)


if __name__ == "__main__":
    main()
