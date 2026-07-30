"""Minimal FastAPI service around the deployed ONNX student.

POST a WAV file to /classify and get top-5 keyword probabilities. The
server holds one onnxruntime session (int8 static if present, else fp32)
and computes the log-mel front-end in torch on CPU.

Run:
  pip install -e ".[demo,deploy]"
  uvicorn demo.serve:app --port 8000
  curl -F "file=@clip.wav" http://localhost:8000/classify
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile
import torch
from fastapi import FastAPI, HTTPException, UploadFile

from phase0.data import CLIP_SAMPLES, LABELS, SAMPLE_RATE
from phase0.student import MelFrontend

RUN_DIR = Path(os.environ.get("RUN_DIR", "runs/best"))

app = FastAPI(title="audio-distill keyword spotter")
frontend = MelFrontend().eval()
_model = next((RUN_DIR / n for n in
               ("model_int8_static.onnx", "model_fp32.onnx")
               if (RUN_DIR / n).exists()), None)
if _model is None:
    raise SystemExit(f"no ONNX model under {RUN_DIR}; run tools.deploy first "
                     "or set RUN_DIR")
session = ort.InferenceSession(str(_model), providers=["CPUExecutionProvider"])


@app.post("/classify")
async def classify(file: UploadFile):
    try:
        audio, sr = soundfile.read(io.BytesIO(await file.read()), dtype="float32")
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}")
    if sr != SAMPLE_RATE:
        raise HTTPException(400, f"expected {SAMPLE_RATE} Hz mono WAV, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wave = torch.zeros(CLIP_SAMPLES)
    wave[:min(len(audio), CLIP_SAMPLES)] = torch.from_numpy(
        audio[:CLIP_SAMPLES])
    with torch.no_grad():
        spec = frontend(wave.unsqueeze(0)).numpy()
    logits = session.run(None, {"spec": spec})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = probs.argsort()[::-1][:5]
    return {"model": _model.name,
            "top5": [{"label": LABELS[i], "prob": round(float(probs[i]), 4)}
                     for i in top]}
