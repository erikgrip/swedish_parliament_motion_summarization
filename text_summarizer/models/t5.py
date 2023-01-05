from typing import Any, Dict
import argparse

from torch import nn
from transformers import MT5ForConditionalGeneration

MT5_VERSION = "small"


class MT5(nn.Module):
    """Model class for MT5 models for conditional generation."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        model_version = self.args.get("model", MT5_VERSION)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            f"google/mt5-{model_version}", return_dict=True
        )
        self.model.eval()

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--mt5_version",
            type=str,
            default=MT5_VERSION,
            help="MT5 model version (small, base, large, xl, xxl)",
        )
        return parser

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """Forward pass through self.model."""
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits
