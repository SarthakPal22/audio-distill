# audio-distill

Learning to compress audio AI models via **knowledge distillation**: training a
small, fast "student" model to imitate a large, accurate "teacher" model.

## Why distillation?

Big models are accurate but expensive to run. Distillation transfers most of a
big model's ability into a model small enough for phones, browsers, and cheap
servers. The trick: instead of training the small model only on true labels, we
also train it to match the big model's *full output distribution* (its "soft
labels"), which carries much more information per example.

## Roadmap

| Phase | Task | Teacher → Student | Status |
|-------|------|-------------------|--------|
| 0 | Keyword spotting (classify 1-second clips into 35 words) | AST transformer (86M params) → tiny CNN (~0.2M params) | **in progress** |
| 1 | Speech-to-text for one language (Hindi / Indian English) | Whisper-small/medium → 2-layer-decoder Whisper student | planned — see `phase1_whisper/` |
| 2 | Serve the student as a real-time transcription service (FastAPI, Docker) | — | planned |

Phase 0 exists to learn the machinery (training loops, spectrograms, the
distillation loss) on a dataset that trains in under an hour. Phase 1 is the
same recipe applied to a real speech-recognition model.

## Quickstart (Phase 0)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. no-download sanity check of the code itself
python tests/smoke_test.py

# 2. tiny end-to-end run (downloads dataset ~2.3 GB + teacher ~350 MB)
python -m phase0.distill --epochs 1 --max-batches 30

# 3. baseline: train the student WITHOUT a teacher
python -m phase0.distill --alpha 1.0 --out checkpoints/student_baseline.pt

# 4. the real thing: train the student WITH distillation
python -m phase0.distill --epochs 8

# 5. compare student vs teacher on the test set
python -m phase0.evaluate --checkpoint checkpoints/student_best.pt --with-teacher
```

Comparing step 3 vs step 4 answers the key question: *how much accuracy did the
teacher's knowledge buy us?*

## Repo layout

```
phase0/            keyword-spotting distillation (start here)
  data.py          Speech Commands dataset, padding, labels
  teacher.py       pretrained AST transformer (frozen, inference only)
  student.py       tiny CNN, ~400x smaller
  distill.py       the training loop and distillation loss
  evaluate.py      accuracy / size / latency comparison
phase1_whisper/    plan for Whisper distillation (Phase 1)
docs/
  PHASE0_GUIDE.md  step-by-step walkthrough with expected results
  GLOSSARY.md      every ML term used in this repo, in plain English
tests/
  smoke_test.py    quick sanity check, no downloads needed
```

## Results (fill in as you go)

| Model | Params | Test accuracy | CPU latency / clip |
|-------|--------|---------------|--------------------|
| Teacher (AST) | ~86M | | |
| Student, no distillation (`--alpha 1.0`) | ~0.2M | | |
| Student, distilled | ~0.2M | | |

## Reading list

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al. 2015, the original 9-page paper; readable
- [Distil-Whisper](https://github.com/huggingface/distil-whisper) — the recipe Phase 1 is modeled on
