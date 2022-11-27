"""Dataset class for Swedish Parliament Motions."""

import argparse
from typing import Any, Tuple

from transformers import MT5TokenizerFast

from .util import BaseDataset



class SwedishParliamentMotionsDataset(BaseDataset):
    """Extends base class to return tokenized texts"""
    def __init__(self, data, targets, args: argparse.Namespace=None) -> None:
        self.args = vars(args) if args is not None else {}
        self.data_transform = self.args.get("data_transform", None)
        self.target_transform = self.args.get("target_transform", None)
"""     def __init__(
        self,
        data,
        targets,
        transform=None,
        target_transform=None,
        tokenizer=MT5TokenizerFast,
        data_max_token_length=256,
        target_max_token_length=32,
    ) -> None: """
        super().__init__(data, targets, transform, target_transform)
        self.tokenizer = tokenizer
        self.data_max_token_length = data_max_token_length
        self.target_max_token_length = target_max_token_length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        text, summary = super().__getitem__(index)

        text_encoding = self.tokenizer(
            text,
            max_length=self.data_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.target_max_token_length,
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
