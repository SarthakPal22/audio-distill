# Phase 0 walkthrough: distilling a keyword-spotting model

Follow these steps in order. Total hands-on time is small; most of the wall
clock is downloads and training runs you can leave unattended.

If any term is unfamiliar, look it up in [GLOSSARY.md](GLOSSARY.md).

## What you're building

A tiny CNN (~200K parameters) that classifies 1-second clips into 35 spoken
words, trained two ways:

1. **Baseline** — normal supervised training on the true labels.
2. **Distilled** — the same model also learns from a frozen 86M-parameter
   transformer teacher's soft labels.

The deliverable is the comparison: same model, same data — how much accuracy
does the teacher's knowledge add?

## Step 1 — Setup (5 min)

```bash
cd ~/Projects/audio-distill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke test needs no downloads and should end with `smoke test passed`.

## Step 2 — Read the code (30–60 min, the most important step)

Read in this order — each file's docstring explains its concepts:

1. `phase0/data.py` — what the dataset looks like
2. `phase0/student.py` — waveform → spectrogram → CNN → logits
3. `phase0/teacher.py` — what "soft labels" are and why they help
4. `phase0/distill.py` — the loss function and the training loop (the heart
   of the project; make sure the `distillation_loss` function makes sense)

## Step 3 — Tiny end-to-end run (15–30 min, mostly downloads)

```bash
python -m phase0.distill --epochs 1 --max-batches 30
```

First run downloads the dataset (~2.3 GB) and the teacher (~350 MB). With only
30 batches, expect low validation accuracy (maybe 10–40%) — the point is just
to see the loop work: loss printed per batch, validation at the end, a
checkpoint saved.

On an Apple Silicon Mac this uses your GPU automatically (`device: mps`).

## Step 4 — Train the baseline (no teacher)

```bash
python -m phase0.distill --alpha 1.0 --epochs 8 --out checkpoints/student_baseline.pt
```

`--alpha 1.0` means 100% hard-label loss, 0% teacher — plain supervised
training. Note the best validation accuracy it reaches.

Rough expectation: **88–94%**. If your Mac is slow here, this is the moment to
switch to a free GPU (Google Colab or Kaggle): upload the repo, `pip install
-r requirements.txt`, run the same commands.

## Step 5 — Train with distillation

```bash
python -m phase0.distill --alpha 0.3 --temperature 4.0 --epochs 8
```

Now 70% of the loss is "match the teacher". Expect a result **1–3 points
above your baseline**, with the gap largest early in training.

## Step 6 — Final comparison on the test set

```bash
python -m phase0.evaluate --checkpoint checkpoints/student_baseline.pt
python -m phase0.evaluate --checkpoint checkpoints/student_best.pt --with-teacher
```

Fill the results table in the README. Your headline will look something like:
"~400x smaller, much faster on CPU, and within a few points of the teacher's
~98% accuracy."

## Step 7 — Experiments (where the real learning happens)

Change one thing at a time and write down what happened:

- **Temperature sweep:** try `--temperature 1.0`, `2.0`, `4.0`, `8.0`. Which
  is best? Why might T=1 underperform? (Hint: reread the glossary entry.)
- **Alpha sweep:** try `--alpha 0.0` (pure teacher, ignores true labels
  entirely). Does it still work? What does that tell you?
- **Less data:** train baseline and distilled on 10% of the data
  (`--max-batches 130`). Distillation's advantage usually *grows* when data
  is scarce — confirm it.
- **Smaller student:** halve the channel sizes in `student.py`. How far can
  you shrink before accuracy collapses?

## Exercise for later (optional, mirrors Phase 1)

The teacher currently runs on every batch, every epoch — wasteful, since its
answers never change. Write a script that runs the teacher over the whole
training set **once**, saves the logits to disk, and modify `distill.py` to
load them instead. This "precompute the pseudo-labels" pattern is exactly how
Whisper distillation works at scale, and it will make your epochs several
times faster.

## When you're done

You understand: training loops, spectrograms, logits/softmax/temperature, the
KD loss, and train/val/test discipline. That's everything Phase 1 needs —
Whisper distillation is the same recipe with a sequence model and WER instead
of accuracy. See `phase1_whisper/README.md`.
