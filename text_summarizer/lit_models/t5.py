import argparse


from text_summarizer.lit_models.base import BaseLitModel

MAX_TEXT_TOKENS = 512
MAX_SUMMARY_TOKENS = 64
TOKENIZER = MT5Tokenizer.from_pretrained("google/mt5-small")


class T5LitModel(BaseLitModel):  # pylint: disable=too-many-ancestors
    """Lightning class to hold a T5 model for conditional generation."""

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args=args)
        self.model = model
        print(args)
        import sys
        sys.exit(0)

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
        print(batch["text"])
        print(batch["label"])

        print(len(batch))
        encoded_texts = [self.encode(t, self.max_text_tokens) for t in batch["text"]]
        encoded_labels = [
            self.encode(l, self.max_summary_tokens, set_zero_to=-100)
            for l in batch["label"]
        ]
        input_ids = [e["text_input_ids"] for e in encoded_texts]
        attention_mask = [e["text_attention_mask"] for e in encoded_texts]
        labels = [e["text_input_ids"] for e in encoded_labels]
        decoder_attention_mask = [e["text_attention_mask"] for e in encoded_labels]

        loss, _ = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.log("train_loss:", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encoded_texts = [self.encode(t) for t in batch["text"]]
        encoded_labels = [self.encode(l, set_zero_to=-100) for l in batch["labels"]]

        loss, _ = self(
            input_ids=encoded_texts["text_input_ids"],
            attention_mask=encoded_texts["text_attention_mask"],
            labels=encoded_labels["text_input_ids"],
            decoder_attention_mask=encoded_labels["text_attention_mask"],
        )
        self.log("val_loss:", loss, prog_bar=True, logger=True)
        self.log("sample prediction", {"text": encoded_texts["text_input_ids"]})

    def test_step(self, batch, batch_idx):
        encoded_texts = [self.encode(t) for t in batch["text"]]
        encoded_labels = [self.encode(l, set_zero_to=-100) for l in batch["labels"]]
        loss, _ = self(
            input_ids=encoded_texts["text_input_ids"],
            attention_mask=encoded_texts["text_attention_mask"],
            labels=encoded_labels["text_input_ids"],
            decoder_attention_mask=encoded_labels["text_attention_mask"],
        )
        self.log("test_loss:", loss, prog_bar=True, logger=True)
