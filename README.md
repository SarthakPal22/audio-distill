# audio-distill

Compressing audio AI models via **knowledge distillation**: training small,
fast "student" models to imitate a large, accurate "teacher" — then
quantizing and deploying them.

**Teacher:** `MIT/ast-finetuned-speech-commands-v2`, an 86M-parameter Audio
Spectrogram Transformer at ~98% accuracy on Google Speech Commands v2
(35-word keyword spotting).
**Students:** 0.03–0.2M-parameter CNNs (plain and depthwise-separable),
~400–840x smaller.

## Method

The training objective is a three-term compression loss:

```
L = α · CE(z_s, y)                                   hard labels
  + (1−α) · T² · KL( softmax(z_t/T) ‖ softmax(z_s/T) )   logit matching
  + β · (1 − cos(P(h_s), h_t))                       feature alignment
```

- **Logit matching via temperature-scaled KL divergence** (Hinton et al.
  2015): temperature T > 1 softens both posteriors, exposing the teacher's
  "dark knowledge" — the inter-class similarity structure hard labels destroy.
- **Feature-based distillation** (FitNets-style): a learned **projection
  head** P aligns the student's penultimate embedding with the teacher's
  pooled hidden state (cosine distance), transferring representations, not
  just predictions.
- **Offline distillation**: the teacher runs *once* over the dataset
  (`phase0/cache_teacher.py`), writing fp16 logits + embeddings to
  memory-mapped packs; training then never pays the ~840x teacher forward.
  Online mode (`--augment online`) recomputes teacher logits on augmented
  audio for the staleness ablation.

## Repo layout

```
phase0/
  data.py           packed memmap dataset, waveform augmentation, stratified subsets
  teacher.py        frozen AST teacher (fp16 inference, logits + embeddings)
  student.py        MelFrontend (+SpecAugment), StudentCNN, DS-CNN, width multipliers
  distill.py        the KD training loop; per-run manifest.json + epochs.csv
  cache_teacher.py  one-time teacher pass -> data/packs/ (offline distillation)
  run_matrix.py     the full experiment matrix (sweeps, ablations; resumable)
  evaluate.py       ECE, reliability bins, teacher agreement, confusion matrix, latency
  metrics.py        calibration + fidelity metrics
tools/
  deploy.py         ONNX export -> parity check -> dynamic/static int8 PTQ -> benchmark
  aggregate.py      runs/*/manifest.json -> mean±std tables + docs/results.csv
  plots.py          all figures (sweeps, Pareto, reliability, confusion, curves)
demo/
  serve.py          FastAPI /classify endpoint over the ONNX student
  stream_mic.py     real-time sliding-window keyword spotting from the mic
tests/              no-download pytest suite (loss properties, shape/quant invariants)
```

## Workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # 19 no-download tests

# 1. one-time: cache teacher logits/embeddings (~1 h on Apple Silicon MPS)
python -m phase0.cache_teacher

# 2. single runs
python -m phase0.distill --alpha 1.0        # hard-label baseline
python -m phase0.distill --alpha 0.3 --temperature 4.0   # canonical KD
python -m phase0.distill --alpha 0.3 --feature-beta 1.0  # + feature alignment

# 3. or the whole matrix (~35 runs: α/T sweeps, low-resource, augmentation,
#    Pareto — each ≥2 seeds where it matters; resumable)
python -m phase0.run_matrix

# 4. analysis + deployment
python -m tools.aggregate                   # mean±std tables -> docs/results.csv
python -m tools.plots                       # figures -> docs/figures/
python -m phase0.evaluate --run-dir runs/<best>
python -m tools.deploy   --run-dir runs/<best>   # ONNX + int8 PTQ + latency
python -m demo.stream_mic --run-dir runs/<best>  # live mic demo
```

## Experiments the matrix answers

| Question | Design |
|---|---|
| Does distillation beat hard-label training? | α ∈ {0, 0.1, 0.3, 0.5, 1.0}, 2 seeds |
| How much softening is right? | T ∈ {1, 2, 4, 8} at α=0.3, 2 seeds |
| Do features add signal beyond logits? | β=1 vs β=0 ablation |
| Is dark knowledge a regularizer? | 100/25/10/2% data, gap vs baseline |
| Stale soft labels under augmentation? | offline vs online distillation at 25% data |
| Accuracy-vs-compute frontier? | {CNN, DS-CNN} × width {0.25, 0.5, 1.0}, MACs |
| Cost of int8? | dynamic + per-channel static PTQ, accuracy Δ + latency |

## Results

Infrastructure is complete and tested; the matrix has not been executed yet.
Run steps 1+3 above, then `python -m tools.aggregate` prints this table
ready to paste (mean ± std over seeds, per configuration).

| Model | Params | MACs | Test acc (%) | ECE | CPU latency |
|---|---|---|---|---|---|
| Teacher (AST, 86M) | 86M | — | ~98 (reference) | | |
| StudentCNN, hard labels | 0.10M | | *pending* | | |
| StudentCNN, distilled | 0.10M | | *pending* | | |
| DS-CNN, distilled, int8 | | | *pending* | | |

## Reading list

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al. 2015
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) — feature-based KD
- [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/abs/1711.07128) — DS-CNN
- [Distil-Whisper](https://github.com/huggingface/distil-whisper) — the Phase 1 recipe
- `docs/GLOSSARY.md` — every term used here, in plain English
