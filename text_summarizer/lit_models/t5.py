import argparse

from transformers import MT5Tokenizer

from text_summarizer.lit_models.base import BaseLitModel


tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")


class T5LitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """Lightning class to hold a T5 model for conditional generation."""

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args=args)
        self.model = model

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
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask,
        )

        self.log("train_loss:", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask,
        )

        self.log("val_loss:", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask,
        )

        self.log("test_loss:", loss, prog_bar=True, logger=True)
        return loss
