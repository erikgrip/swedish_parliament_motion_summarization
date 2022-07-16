from typing import Any, Dict
import argparse

from torch import nn
from transformers import T5ForConditionalGeneration


class t5(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-small", return_dict=True
        )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=123)
        return parser

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits
