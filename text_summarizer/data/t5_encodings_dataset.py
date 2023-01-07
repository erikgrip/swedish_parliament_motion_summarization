import argparse
from typing import Any, Dict

from transformers.models.mt5 import MT5Tokenizer

from .base_dataset import BaseDataset

MAX_TEXT_TOKENS = 512
MAX_TITLE_TOKENS = 64
MT5_VERSION = "small"


class T5EncodingsDataset(BaseDataset):
    """Extends base class to return text encodings."""

    def __init__(self, data, targets, args: argparse.Namespace = None) -> None:
        self.args = vars(args) if args is not None else {}
        super().__init__(
            data,
            targets,
            self.args.get("data_transform", None),
            self.args.get("target_transform", None),
        )
        model_version = self.args.get("model_version", MT5_VERSION)
        model = f"google/mt5-{model_version}"
        self.tokenizer = MT5Tokenizer.from_pretrained(model)
        self.max_text_tokens = self.args.get("max_text_tokens", MAX_TEXT_TOKENS)
        self.max_title_tokens = self.args.get("max_title_tokens", MAX_TITLE_TOKENS)

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--model_version",
            type=int,
            default=MT5_VERSION,
            help="Version of MT5 model to load tokenizer from.",
        )
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
            max_length=self.max_title_tokens,
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

    ds = T5EncodingsDataset(data=dummy_data, targets=dummy_targets)
