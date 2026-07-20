# Glossary

Every term this project uses, in plain English. Read top to bottom once, then
come back whenever a word in the code is unfamiliar.

## Models and training basics

- **Model / neural network** — a function with millions of adjustable numbers
  (**parameters** or **weights**) that maps an input (audio) to an output
  (class scores). Training = adjusting those numbers to reduce mistakes.
- **Parameters** — the adjustable numbers. "86M parameters" = 86 million of
  them. More parameters ≈ more capacity, more memory, slower inference.
- **Inference** — running a trained model to get predictions (no learning).
- **Training loop** — the repeated cycle: feed a batch in, measure the error,
  compute how each parameter should change, nudge them, repeat.
- **Batch** — a group of examples (e.g. 64 audio clips) processed together.
  GPUs are fast at doing many examples at once.
- **Epoch** — one full pass through the training dataset.
- **Loss** — a single number measuring how wrong the model is right now.
  Training minimizes it. Watching loss go down = watching learning happen.
- **Cross-entropy** — the standard loss for classification: heavily penalizes
  the model for assigning low probability to the correct class.
- **Gradient / backpropagation** — the gradient says, for each parameter,
  which direction to nudge it to reduce the loss. Backpropagation is the
  algorithm that computes gradients efficiently. `loss.backward()` in PyTorch.
- **Optimizer (AdamW)** — the rule for applying gradients to parameters.
  AdamW is the sensible default in 2026.
- **Learning rate** — how big each nudge is. Too high: training blows up.
  Too low: takes forever. `3e-4` is a classic starting point for AdamW.
- **Learning-rate schedule (cosine annealing)** — start at the full learning
  rate and smoothly decay toward zero; helps the model settle at the end.
- **Checkpoint** — the model's parameters saved to a file (`.pt`), so you can
  reload it later without retraining.
- **Overfitting** — the model memorizes training examples instead of learning
  general patterns; training accuracy rises while validation accuracy stalls.

## Data terms

- **Train / validation / test split** — train: what the model learns from.
  Validation: held-out data used *during* development to pick the best
  checkpoint and settings. Test: touched only at the very end, for the honest
  final number.
- **Label** — the correct answer for an example ("this clip says 'yes'").
- **Hard label vs soft label** — a hard label is just the answer ("yes").
  A soft label is a full probability distribution ("70% yes, 20% left,
  10% yeah…"), typically produced by a teacher model.
- **DataLoader** — PyTorch machinery that shuffles the dataset, groups
  examples into batches, and loads them in background workers.

## Distillation terms

- **Knowledge distillation (KD)** — training a small **student** model to
  match a large **teacher** model's outputs. From Hinton et al. 2015.
- **Teacher** — the big, accurate, frozen model. Never trained here; only
  queried.
- **Student** — the small model being trained.
- **Logits** — the raw scores a model outputs *before* they're converted to
  probabilities. One number per class; higher = more confident.
- **Softmax** — the function that turns logits into probabilities that sum
  to 1.
- **Temperature (T)** — dividing logits by T > 1 before softmax "softens"
  the probabilities (makes them less extreme), exposing the teacher's
  secondary opinions, which is exactly the knowledge worth transferring.
- **KL divergence** — a measure of how different two probability
  distributions are. The soft part of the distillation loss minimizes the KL
  divergence between student and teacher outputs.
- **Alpha (α)** — the blend weight in this repo: `α * hard_loss +
  (1-α) * soft_loss`. α = 1.0 means "ignore the teacher" (baseline).
- **Pseudo-labeling** — using a teacher's *predictions* on unlabeled data as
  if they were labels. This is how Phase 1 (Whisper) will work: the teacher
  transcribes audio, the student learns from those transcripts.

## Audio terms

- **Waveform** — raw audio: a long list of numbers (air-pressure samples).
  At 16 kHz there are 16,000 numbers per second.
- **Sample rate** — how many samples per second. Speech models almost always
  use 16 kHz.
- **Spectrogram** — audio converted to a 2D image: time on the x-axis,
  frequency (pitch) on the y-axis, brightness = energy. Lets us reuse image
  techniques (CNNs, vision transformers) on sound.
- **Mel scale** — a frequency axis warped to match human hearing (we're
  better at telling low pitches apart than high ones). A **mel spectrogram**
  is the standard input for speech models.
- **Keyword spotting** — classifying a short clip into one of N known words.
  The simplest speech task; our Phase 0.
- **ASR (automatic speech recognition) / speech-to-text** — transcribing
  arbitrary speech into text. Phase 1. Harder than keyword spotting because
  the output is a *sequence*, not a single class.
- **WER (word error rate)** — the standard ASR metric: what fraction of words
  the model gets wrong (insertions + deletions + substitutions). Lower is
  better.

## Hardware terms

- **GPU / CUDA** — graphics processors do the matrix math of neural nets
  10–100x faster than CPUs. CUDA is NVIDIA's interface to them (`device="cuda"`).
- **MPS** — Apple's GPU interface; on an Apple Silicon Mac, PyTorch uses your
  Mac's GPU with `device="mps"`.
- **Latency** — how long one prediction takes. The whole point of a distilled
  model is lower latency (and memory) at close-to-teacher accuracy.
