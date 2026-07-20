"""The teacher: a large pretrained Audio Spectrogram Transformer (AST).

This model (~86M parameters) was already fine-tuned on Speech Commands v2
by MIT and reaches ~98% accuracy. We never train it — we only *ask it
questions*. During distillation we feed it the same audio the student
sees and record its logits (raw pre-softmax scores). Those logits are the
"soft labels" the student learns to imitate.

Why soft labels beat hard labels: a hard label says only "this clip is
'go'". The teacher's logits also say "…but it sounds 60% like 'go', 30%
like 'no', 10% like 'low'" — the similarity structure between classes.
That extra signal is what makes distillation work.
"""

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from phase0.data import LABELS, SAMPLE_RATE

MODEL_ID = "MIT/ast-finetuned-speech-commands-v2"


class Teacher:
    def __init__(self, device: torch.device):
        self.device = device
        # The feature extractor converts raw waveforms into the
        # mel-spectrogram "image" format the AST expects.
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = ASTForAudioClassification.from_pretrained(MODEL_ID)
        self.model.to(device)
        self.model.eval()  # inference mode: disables dropout etc.

        # The teacher was trained with its own class ordering, which may
        # differ from ours (and may contain extra classes like
        # "_unknown_"). Build a column map so teacher logits line up with
        # our LABELS order.
        teacher_label_to_id = {
            name.lower(): idx for name, idx in self.model.config.label2id.items()
        }
        missing = [w for w in LABELS if w not in teacher_label_to_id]
        if missing:
            raise RuntimeError(f"teacher checkpoint lacks classes: {missing}")
        self.columns = torch.tensor(
            [teacher_label_to_id[w] for w in LABELS], device=device
        )

    @torch.no_grad()  # never compute gradients for the teacher: frozen + faster
    def logits(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms: (batch, 16000) float tensor on CPU.
        Returns logits of shape (batch, 35) aligned to data.LABELS."""
        inputs = self.feature_extractor(
            [w.numpy() for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        out = self.model(input_values=inputs.input_values.to(self.device))
        return out.logits.index_select(1, self.columns)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
