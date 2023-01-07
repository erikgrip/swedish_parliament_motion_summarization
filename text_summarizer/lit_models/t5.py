import argparse

from transformers.models.mt5 import MT5Tokenizer

from text_summarizer.lit_models.base import BaseLitModel
from text_summarizer.util import summarize, tokenize

MAX_TEXT_TOKENS = 512  # Keep in sync with Lit Dataset settings
MAX_TITLE_TOKENS = 64


class MT5LitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """Lightning class to hold a MT5 model for conditional generation."""

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args=args)
        self.model = model
        self.tokenizer = MT5Tokenizer.from_pretrained(model.model_name)
        self.max_title_tokens = self.args.get("max_title_tokens", MAX_TITLE_TOKENS)

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
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("train_loss:", loss, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "sample_text_encoding": batch["text_input_ids"][0],
            "sample_title": batch["title"][0],
        }

    def training_epoch_end(self, training_step_outputs):
        print("On training epoch end:")
        self.logger.experiment.add_text("log", "apa")
        print(training_step_outputs)
        sample_output = summarize(
            self.model,
            self.tokenizer,
            training_step_outputs[0]["sample_text"],
            self.max_title_tokens,
        )

        print(training_step_outputs[0]["sample_title"])
        print(sample_output)

    def validation_step(self, batch, batch_idx):
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("val_loss:", loss, prog_bar=True, logger=True)
        self.log(
            "sample prediction",
            batch["text_input_ids"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        loss, _ = self(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            labels=batch["title_mod_ids"],
            decoder_attention_mask=batch["title_attention_mask"],
        )
        self.log("test_loss:", loss, prog_bar=True, logger=True)
