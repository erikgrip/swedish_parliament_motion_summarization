from typing import Any, Dict
import argparse

from torch import nn
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers.models.mt5 import MT5Tokenizer

MT5_VERSION = "small"
MAX_TEXT_TOKENS = 512
MAX_TITLE_TOKENS = 64


class MT5(nn.Module):
    """Model class for MT5 models for conditional generation."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        model_version = self.args.get("mt5_version", MT5_VERSION)
        model = f"google/mt5-{model_version}"
        self.model = MT5ForConditionalGeneration.from_pretrained(
            model, return_dict=True
        )
        self.tokenizer = MT5Tokenizer.from_pretrained(model)
        self.max_text_tokens = self.args.get("max_text_tokens", MAX_TEXT_TOKENS)
        self.max_title_tokens = self.args.get("max_title_tokens", MAX_TITLE_TOKENS)

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--mt5_version",
            type=str,
            default=MT5_VERSION,
            help="MT5 model version (small, base, large, xl, xxl)",
        )
        parser.add_argument(
            "--max_text_tokens",
            type=int,
            default=MAX_TEXT_TOKENS,
            help="Maximum number of tokens to use from text",
        )
        parser.add_argument(
            "--max_title_tokens",
            type=int,
            default=MAX_TITLE_TOKENS,
            help="Maximum number of tokens to generate for motion title",
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
