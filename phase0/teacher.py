"""The teacher: a large pretrained Audio Spectrogram Transformer (AST).

This model (~86M parameters) was already fine-tuned on Speech Commands v2
by MIT and reaches ~98% accuracy. We never train it — we only *ask it
questions*. During distillation we feed it the same audio the student
sees and record its logits (raw pre-softmax scores). Those logits are the
"soft labels" the student learns to imitate.

Why soft labels beat hard labels: a hard label says only "this clip is
'go'". The teacher's logits also say "…but it sounds 60% like 'go', 30%
like 'no', 10% like 'low'" — the inter-class similarity structure
("dark knowledge") that hard labels destroy.

Engineering notes (they matter at 106k-clip caching scale):

- Feature extraction (waveform -> 1024x128 mel filterbank, CPU) and the
  transformer forward (GPU) are exposed separately, so callers can
  parallelize the CPU half across DataLoader workers while the GPU stays
  busy — a classic producer/consumer pipeline.
- On GPU/MPS the teacher runs in **fp16 (half precision)**: weights and
  activations use 16-bit floats, roughly doubling throughput. The tiny
  rounding noise this adds to logits is far below the softmax
  temperature scales KD operates at.
- We call the transformer trunk directly and pool one hidden state
  instead of requesting `output_hidden_states=True`, which would
  materialize all 13 layers' activations just to throw 12 away.
"""

from __future__ import annotations

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from phase0.data import LABELS, SAMPLE_RATE

MODEL_ID = "MIT/ast-finetuned-speech-commands-v2"


class Teacher:
    def __init__(self, device: torch.device, fp16: bool = True):
        self.device = device
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = ASTForAudioClassification.from_pretrained(MODEL_ID)
        if fp16 and device.type in ("cuda", "mps"):
            self.model.half()
        self.model.to(device)
        self.model.eval()

        # The teacher was trained with its own class ordering. Build a
        # column map so teacher logits line up with our LABELS order.
        teacher_label_to_id = {
            name.lower(): idx for name, idx in self.model.config.label2id.items()
        }
        missing = [w for w in LABELS if w not in teacher_label_to_id]
        if missing:
            raise RuntimeError(f"teacher checkpoint lacks classes: {missing}")
        self.columns = torch.tensor(
            [teacher_label_to_id[w] for w in LABELS], device=device
        )

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """CPU half: waveforms (B, 16000) -> mel filterbank (B, 1024, 128)."""
        return self.feature_extractor(
            [w.numpy() for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_values

    @torch.no_grad()
    def forward_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU half: features -> (logits (B, 35), embedding (B, 768)).

        The embedding is the mean over the final-layer (layer-normalized)
        token sequence — one vector summarizing what the teacher "heard".
        Feature-based distillation aligns the student's representation to
        it, transferring *how* the teacher represents audio, not only
        *what* it predicts.
        """
        features = features.to(self.device, dtype=self.model.dtype)
        trunk_out = self.model.audio_spectrogram_transformer(input_values=features)
        logits = self.model.classifier(trunk_out.pooler_output)
        embedding = trunk_out.last_hidden_state.mean(dim=1)
        return logits.float().index_select(1, self.columns), embedding.float()

    @torch.no_grad()
    def logits_and_embedding(
        self, waveforms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_features(self.extract_features(waveforms))

    @torch.no_grad()
    def logits(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms (B, 16000) on CPU -> logits (B, 35) aligned to LABELS."""
        return self.logits_and_embedding(waveforms)[0]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
