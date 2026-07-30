# Prompt for Fable: make audio-distill phenomenal

## Context

You are working in `~/Projects/audio-distill`, a knowledge-distillation (KD) project. Current state:

- **Phase 0 (scaffolded, never trained):** response-based KD on Google Speech Commands v2 (35-class keyword spotting, ~105k one-second 16 kHz clips). Teacher: frozen `MIT/ast-finetuned-speech-commands-v2` (86M-param Audio Spectrogram Transformer, ~98% test accuracy). Student: ~0.2M-param CNN over log-mel spectrograms (64 mels, 25 ms window / 10 ms hop). Loss: `alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))` per Hinton et al. 2015. Modules: `phase0/{data,teacher,student,distill,evaluate}.py`, plus `tests/smoke_test.py` and `docs/`.
- **Phase 1 (plan only):** Distil-Whisper-style encoder-preserving, decoder-shrinking distillation of `openai/whisper-small` for one language (Hindi or Indian-accented English), trained on teacher pseudo-labels.
- The results table in `README.md` is empty. No checkpoints exist. Hardware is an Apple Silicon Mac (MPS backend); assume Colab/Kaggle for anything needing CUDA.

## Mission

Turn this into a portfolio-grade **model-compression research repo**: rigorous experiments, modern KD techniques, deployable artifacts, and documentation that reads like a well-written systems/ML paper. It should demonstrate mastery of the full compression stack — response-based, feature-based, and data-based distillation, plus post-training quantization and on-device benchmarking — not just a single Hinton-loss training run.

## Non-negotiable style constraint (this is how I learn)

Preserve and extend the repo's pedagogical voice, but make it **maximally technical**: use the precise term of art for everything (e.g. "logit matching via temperature-scaled KL divergence", "dark knowledge", "posterior sharpening", "representation alignment via projection heads", "per-channel affine int8 quantization", "operator fusion"), and define each term inline the first time it appears — in module docstrings, in `docs/GLOSSARY.md`, and in the README. Every experiment must state its hypothesis, its controlled variables, and its conclusion. I want to be able to read this repo top to bottom and come out understanding compression research vocabulary.

## Workstream 1 — Empirical foundation (do this first; everything else depends on it)

1. Implement **offline distillation / pseudo-logit caching**: a `phase0/cache_teacher.py` script that runs the frozen teacher over the full training split once, memory-maps the (N, 35) fp16 logit tensor to disk, and a `--cached-logits` path in `distill.py` that eliminates the teacher's forward pass from the training loop. Report the epoch-time speedup. (This is the same amortization pattern Distil-Whisper uses for pseudo-labeling, so it directly de-risks Phase 1.)
2. Add **experiment reproducibility**: seeded RNG (`torch`, `numpy`, dataloader workers), deterministic flags where MPS allows, and a per-run JSON manifest (git SHA, args, seed, final metrics) written next to each checkpoint.
3. Run the actual matrix — baseline (`alpha=1.0`), distilled (`alpha=0.3, T=4`), and a **temperature sweep** (T ∈ {1, 2, 4, 8}) and **alpha sweep** (alpha ∈ {0.0, 0.1, 0.3, 0.5, 1.0}) — each with ≥2 seeds, on MPS locally or via a generated Colab/Kaggle notebook if wall-clock demands it. Fill the README table with mean ± std, not single runs.
4. Add the **low-resource ablation**: subsample the training set to {100%, 25%, 10%, 2%} and show the distillation gap widening as labeled data shrinks — the canonical evidence that soft labels act as a regularizer carrying inter-class similarity structure ("dark knowledge").

## Workstream 2 — Beyond logit matching (the techniques that make it impressive)

5. **Feature-based distillation (FitNets-style hint training):** align an intermediate student feature map to a teacher hidden state through a learned 1x1-conv/linear **projection head** (needed because dimensionalities differ), with an L2 or cosine alignment loss added to the objective. Ablate: logits-only vs logits+features.
6. **Data augmentation interaction:** implement **SpecAugment** (time and frequency masking on the log-mel spectrogram) and waveform-level augmentation (time shift, background noise from the dataset's `_background_noise_` folder). Study the KD-specific subtlety: when inputs are augmented, cached teacher logits are stale — support both "teacher sees augmented input" (online) and "clean-cached logits" (offline) modes and measure the difference.
7. **Calibration and agreement metrics** in `evaluate.py`: expected calibration error (ECE) with reliability diagrams, student–teacher top-1 agreement rate, and mean KL to the teacher on the test set. Distilled models are typically better calibrated than hard-label baselines — verify and plot it.
8. **Error analysis:** confusion matrix heatmap; show that residual student errors concentrate in acoustically confusable pairs (`go/no`, `three/tree`, `forward/four`), which is exactly the similarity structure soft labels encode.

## Workstream 3 — Architecture and efficiency engineering

9. Add a second student: a **depthwise-separable CNN (DS-CNN)** or **BC-ResNet**-style model (the standard efficient keyword-spotting architectures), and report **MACs/FLOPs** (via `fvcore` or `ptflops`) alongside parameter count. Build a small **accuracy-vs-MACs Pareto frontier** across student widths (channel multiplier ∈ {0.25, 0.5, 1.0}).
10. **Post-training quantization:** dynamic and static **int8 PTQ** of the best student (per-channel weight quantization, observer-based activation calibration), reporting accuracy delta and size on disk. Export to **ONNX**, validate numerical parity (max |Δlogit|), and benchmark with `onnxruntime` — median and p99 single-clip CPU latency with a proper warm-up, replacing the current naive timer.
11. Package a **real-time streaming demo**: microphone capture with a 1 s sliding window and hop-based inference, plus a minimal FastAPI `/classify` endpoint serving the ONNX model. This makes the compression story tangible.

## Workstream 4 — Engineering hygiene

12. Convert to a proper package (`pyproject.toml`), add type hints throughout, expand `tests/` (loss correctness: KD loss must reduce to CE at `alpha=1.0` and be minimized when student equals teacher; dataset shape/padding invariants; ONNX parity test), and a GitHub Actions CI running the no-download tests.
13. Lightweight experiment tracking: CSV/JSON logs plus matplotlib plots checked into `docs/figures/` (training curves, sweeps, Pareto frontier, reliability diagram). No heavyweight tracking service required.

## Workstream 5 — Documentation as the deliverable

14. Rewrite the README as a compact technical report: motivation, method (with the loss written in math), experimental setup, results tables with all sweep numbers, plots, ablation conclusions, limitations, and a "what I learned" appendix. Headline format: "N× smaller, M× faster on CPU int8, within K points of an 86M-parameter transformer."
15. Expand `docs/GLOSSARY.md` with every new term introduced (offline vs online distillation, projection head, ECE, PTQ vs QAT, depthwise-separable convolution, MACs, Pareto frontier, SpecAugment, pseudo-labeling, WER…).
16. Update `phase1_whisper/README.md` into a concrete execution plan informed by Phase 0's measured results: dataset choice (Kathbath/Shrutilipi vs Common Voice), pseudo-label confidence filtering criteria, per-token KD loss formulation, WER evaluation with `jiwer`, and a compute budget.

## Execution order and constraints

- Order: Workstream 1 → 2 → 3, with 4 and 5 woven in as you go. Do not skip the sweeps in Workstream 1 — an impressive repo with an empty results table is worthless.
- Keep dependencies minimal and MPS-compatible; anything CUDA-only goes in a clearly marked optional notebook.
- Long training runs: launch them, monitor, and continue other workstreams in parallel; never leave the results table partially filled at the end.
- Every commit message should name the technique it introduces, in technical terms.
