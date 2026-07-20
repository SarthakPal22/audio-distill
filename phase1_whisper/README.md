# Phase 1: distilling Whisper for one language (planned)

Do not start this until Phase 0 is done — it's the same recipe, scaled up.

## Goal

A distilled Whisper student for **one** language/domain (suggested: Hindi or
Indian-accented English), reported as: X% smaller, Yx faster, within Z% WER
of the teacher.

## The recipe (following Distil-Whisper)

1. **Teacher:** `openai/whisper-small` (or `-medium`). Frozen.
2. **Student architecture:** copy the teacher's encoder (frozen or lightly
   trained), keep only 2 decoder layers initialized from the teacher's first
   and last decoder layers. Most of Whisper's inference cost is the decoder,
   so this is where the speedup comes from.
3. **Data:** 100–500 hours of speech in the target language.
   - Hindi / Indic: AI4Bharat's Kathbath and Shrutilipi (open)
   - Alternative: Mozilla Common Voice for the target language
4. **Pseudo-labeling:** run the teacher over all the audio once, cache its
   transcripts (you practiced this pattern in the Phase 0 exercise). Filter
   out low-quality pseudo-labels (e.g. by teacher confidence / heuristics).
5. **Train:** cross-entropy on the pseudo-label tokens + KL divergence
   between student and teacher token distributions — the same two-part loss
   as Phase 0, applied per token of the transcript.
6. **Evaluate:** WER of teacher vs student on a held-out test set
   (jiwer library), plus latency and model size.

## Compute

- Pseudo-labeling: one pass of the teacher over the dataset (cache to disk).
- Training: roughly 20–60 GPU-hours on a T4/A10 class GPU.
  Free/cheap options: Kaggle (30 GPU-hrs/week free), Colab Pro, RunPod.

## Useful references

- https://github.com/huggingface/distil-whisper (paper + training code)
- https://huggingface.co/blog/fine-tune-whisper (Whisper training mechanics)
- https://ai4bharat.iitm.ac.in/datasets (Indic speech datasets)
