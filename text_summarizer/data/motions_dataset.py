"""Dataset class for Swedish Parliament Motions."""

import argparse
from typing import Any, Dict

from transformers import MT5Tokenizer

from .util import BaseDataset

MAX_TEXT_TOKENS = 512
MAX_SUMMARY_TOKENS = 64
TOKENIZER = MT5Tokenizer.from_pretrained("google/mt5-small")

class T5EncodingsDataset(BaseDataset):
    """Extends base class to return text encodings."""

class SwedishParliamentMotionsDataset(BaseDataset):
    """Extends base class to return tokenized texts."""

    def __init__(self, data, targets, args: argparse.Namespace = None) -> None:
        self.args = vars(args) if args is not None else {}
        super().__init__(
            data,
            targets,
            self.args.get("data_transform", None),
            self.args.get("target_transform", None),
        )
        self.tokenizer = self.args.get("tokenizer", TOKENIZER)
        self.max_text_tokens = self.args.get("max_text_tokens", MAX_TEXT_TOKENS)
        self.max_summary_tokens = self.args.get(
            "max_summary_tokens", MAX_SUMMARY_TOKENS
        )

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--max_text_tokens",
            type=int,
            default=MAX_TEXT_TOKENS,
            help="Maximum number of tokens to use from text",
        )
        parser.add_argument(
            "--max_summary_tokens",
            type=int,
            default=MAX_SUMMARY_TOKENS,
            help="Maximum number of tokens to generate for summary",
        )
        parser.add_argument(
            "--tokenizer",
            type=int,
            default=TOKENIZER,
            help="Tokenizer to use.",
        )
        return parser

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """Return text and summary with their encodings and attention masks."""
        text, summary = super().__getitem__(index)

        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.max_summary_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=summary,
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten(),
        )


if __name__ == "__main__":
    dummy_data = ["This is my first sentence", "And here's my 2nd one"]
    dummy_targets = ["The first", "The second"]

    ds = SwedishParliamentMotionsDataset(data=dummy_data, targets=dummy_targets)
