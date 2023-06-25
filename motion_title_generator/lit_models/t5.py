from random import sample
from typing import Dict, List, Optional

from transformers.models.mt5 import MT5Tokenizer

from motion_title_generator.data.t5_encodings_dataset import (
    MAX_TEXT_TOKENS,
    MAX_TITLE_TOKENS,
)
from motion_title_generator.lit_models.base import BaseLitModel
from utils.encode_decode import generate


class MT5LitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """Lightning class to hold a MT5 model for conditional generation."""

    def __init__(self, model, args: Optional[Dict] = None):
        super().__init__(model, args=args)
        self.args = args if args is not None else {}
        self.model = model
        self.tokenizer = MT5Tokenizer.from_pretrained(model.model_name)
        self.max_text_tokens = self.args.get("max_title_tokens", MAX_TEXT_TOKENS)
        self.max_title_tokens = self.args.get("max_title_tokens", MAX_TITLE_TOKENS)
        self.validation_step_outputs: List[Dict] = []

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--max_text_tokens",
            type=int,
            default=MAX_TEXT_TOKENS,
            help="Number of tokens to use from text.",
        )
        parser.add_argument(
            "--num_title_tokens",
            type=int,
            default=MAX_TITLE_TOKENS,
            help="Max number of tokens to generate in summary.",
        )
        return parser

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """Forward pass through self.model."""
        loss, logits = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return loss, logits

    def training_step(self, batch, batch_idx):
        """Run forward pass and return loss."""
        loss, _ = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Run forward pass and return loss.

        Also save sample outputs to generate title on epoch end.
        """
        loss, _ = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({k: v[:1] for k, v in batch.items()})
        return loss

    def on_validation_epoch_end(self):
        """Generate title for a sample of validation set."""
        sample_output = sample(self.validation_step_outputs, 1)[0]
        self.validation_step_outputs.clear()

        generated_title = generate(
            self.model.model,
            self.tokenizer,
            sample_output,
            self.max_title_tokens,
        )
        self.logger.experiment.add_text(
            "actual title", sample_output["title"][0], global_step=self.global_step
        )
        self.logger.experiment.add_text(
            "generated title", generated_title, global_step=self.global_step
        )

    def test_step(self, batch, batch_idx):
        """Run forward pass and log loss."""
        loss, _ = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("test_loss", loss, prog_bar=True, logger=True)
