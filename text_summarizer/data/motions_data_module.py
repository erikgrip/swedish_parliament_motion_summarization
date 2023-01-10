import argparse

from torch.utils.data import random_split, DataLoader
import pandas as pd
import torch

from text_summarizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_summarizer.data.motions_dataset import SwedishParliamentMotionsDataset
from training_dataset_downloader import get_training_dataset


DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"
TEST_DATA_DIRNAME = BaseDataModule.data_dirname() / "test"

BATCH_SIZE = 8


class SweParliamentMotionsDataModule(BaseDataModule):
    """Pytorch lightning DataModule class for the motion data."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = vars(args) if args is not None else {}
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.data_file = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"
        self.seed = 2

    def prepare_data(self, *args, **kwargs):
        """Define steps that should be done on only one GPU, like getting data."""

        # Download and concatenate data
        # TODO: Make path not hardcoded here but passed in get_training_dataset call
        get_training_dataset()

    def setup(self, stage: str = None) -> None:
        """Define steps that should be done on every GPU, like splitting data,
        applying transform etc."""

        if self.args.get("overfit_batches", 0) == 1:
            data = pd.read_csv(TEST_DATA_DIRNAME / "test_data.csv")
        else:
            data = pd.read_feather(self.data_file)
        # TODO: Split data more elegantly
        train_size = int(len(data) * 0.70)
        val_size = int(len(data) * 0.15)
        test_size = len(data) - train_size - val_size
        data_train, data_val, data_test = random_split(
            data,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.data_train = SwedishParliamentMotionsDataset(
            data=data.iloc[list(data_train.indices)]["text"].tolist(),
            targets=data.iloc[list(data_train.indices)]["title"].tolist(),
        )
        self.data_val = SwedishParliamentMotionsDataset(
            data=data.iloc[list(data_val.indices)]["text"].tolist(),
            targets=data.iloc[list(data_val.indices)]["title"].tolist(),
        )
        self.data_test = SwedishParliamentMotionsDataset(
            data=data.iloc[list(data_test.indices)]["text"].tolist(),
            targets=data.iloc[list(data_test.indices)]["title"].tolist(),
        )

    def train_dataloader(self):
        """Return DataLoader for Training Data."""
        return DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        """Return DataLoader for Validation Data."""
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        """Return DataLoader for Test Data."""
        return DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    load_and_print_info(SweParliamentMotionsDataModule)
