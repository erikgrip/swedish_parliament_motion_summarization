import argparse

from torch.utils.data import random_split, DataLoader
from transformers import T5TokenizerFast
import pandas as pd
import torch

from text_summarizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_summarizer.data.motions_dataset import SwedishParliamentMotionsDataset
from training_dataset_downloader import get_training_dataset


DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

BATCH_SIZE = 8
MAX_TEXT_TOKENS = 512
MAX_SUMMARY_TOKENS = 32
TOKENIZER = T5TokenizerFast


class SweParliamentMotionsDataModule(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = vars(args) if args is not None else {}
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.data_file = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"
        self.tokenizer = self.args.get("tokenizer", TOKENIZER)
        self.max_text_tokens = self.args.get("max_text_tokens", MAX_TEXT_TOKENS)
        self.max_summary_tokens = self.args.get(
            "max_summary_tokens", MAX_SUMMARY_TOKENS
        )

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        # parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--max_text_tokens", type=int, default=MAX_TEXT_TOKENS)
        parser.add_argument(
            "--max_summary_tokens", type=int, default=MAX_SUMMARY_TOKENS
        )
        return parser

    def prepare_data(self):
        # Define steps that should be done on only one GPU, like getting data.

        # Download and concatenate data
        # TODO: Make path not hardcoded here but passed in get_training_dataset call
        get_training_dataset()

    def setup(self, stage: str = None) -> None:
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        df = pd.read_feather(self.data_file)
        # TODO: Split data more elegantly
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        test_size = len(df) - train_size - val_size
        data_train, data_val, data_test = random_split(
            df,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.data_train = SwedishParliamentMotionsDataset(
            data_train.dataset["text"].tolist(),
            data_train.dataset["title"].tolist(),
        )
        self.data_val = SwedishParliamentMotionsDataset(
            data_val.dataset["text"].tolist(),
            data_val.dataset["title"].tolist(),
        )
        self.data_test = SwedishParliamentMotionsDataset(
            data_test.dataset["text"].tolist(),
            data_test.dataset["title"].tolist(),
        )

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        return DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        return DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


if __name__ == "__main__":
    load_and_print_info(SweParliamentMotionsDataModule)
