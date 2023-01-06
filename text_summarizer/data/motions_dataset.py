"""Dataset class for Swedish Parliament Motions."""

import argparse
from typing import Any, Dict

from transformers import MT5Tokenizer

from .util import BaseDataset


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

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """Return text and summary with their encodings and attention masks."""
        text, label = super().__getitem__(index)
        return dict(text=text, label=label)


if __name__ == "__main__":
    dummy_data = ["This is my first sentence", "And here's my 2nd one"]
    dummy_targets = ["The first", "The second"]

    ds = SwedishParliamentMotionsDataset(data=dummy_data, targets=dummy_targets)
