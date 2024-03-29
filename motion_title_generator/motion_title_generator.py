import os

import torch
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.mt5 import MT5Tokenizer

from utils.encode_decode import encode, generate

MODEL = "erikgrip2/mt5-finetuned-for-motion-title"
MAX_TEXT_TOKENS = 512
MAX_TITLE_TOKENS = 64


class MotionTitleGenerator:
    """Class to generate a title for a motion text."""

    def __init__(self):
        model = os.environ.get("HF_REPO_OR_ARTIFACT_PATH", MODEL)  # Set in Dockerfile
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = MT5Tokenizer.from_pretrained(model)

    def encode_text(self, text):
        """Use tokenizer to encode text."""
        return encode(text, tokenizer=self.tokenizer, max_tokens=MAX_TEXT_TOKENS)

    @torch.no_grad()
    def predict(self, text: str) -> str:
        """Generate a title for an input text."""
        enc = self.encode_text(text)
        return generate(self.model, self.tokenizer, enc, MAX_TITLE_TOKENS)
