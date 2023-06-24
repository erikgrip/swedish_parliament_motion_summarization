import argparse
from typing import Any, Dict

from transformers.models.mt5 import MT5Tokenizer

from motion_title_generator.data.base_dataset import BaseDataset
from utils.encode_decode import encode

MAX_TEXT_TOKENS = 512
MAX_TITLE_TOKENS = 64
MT5_VERSION = "small"


class MT5EncodingsDataset(BaseDataset):
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
        """Return text and title with their encodings and attention masks."""
        items = super().__getitem__(index)
        text = items["datum"]
        title = items["target"]

        text_encoding = encode(text, self.tokenizer, self.max_text_tokens)
        title_encoding = encode(title, self.tokenizer, self.max_title_tokens)

        title_mod_ids = title_encoding["input_ids"]
        title_mod_ids[title_mod_ids == 0] = -100

        return {
            "text": text,
            "title": title,
            "input_ids": text_encoding["input_ids"].flatten(),
            "attention_mask": text_encoding["attention_mask"].flatten(),
            "title_mod_ids": title_mod_ids.flatten(),
            "title_attention_mask": title_encoding["attention_mask"].flatten(),
        }
