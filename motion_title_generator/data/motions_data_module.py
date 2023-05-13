# type: ignore
import argparse

import pandas as pd
from torch.utils.data import DataLoader, random_split

from motion_title_generator.data.base_data_module import BaseDataModule, load_and_print_info
from motion_title_generator.data.t5_encodings_dataset import MT5EncodingsDataset
from motion_title_generator.data.util import split_data
from training_dataset_downloader import get_training_dataset

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"
TEST_DATA_DIRNAME = BaseDataModule.data_dirname() / "test"

DATA_FRACTION = 1.0  # Allows scaling data down for faster trining
TRAIN_FRAC = 0.75
VAL_FRAC = 0.15


class MotionsDataModule(BaseDataModule):
    """Pytorch lightning DataModule class for the motion data."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = vars(args) if args is not None else {}
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.data_file = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"
        self.data_fraction = float(self.args.get("data_fraction", DATA_FRACTION))
        self.seed = 2

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--data_fraction",
            type=float,
            default=DATA_FRACTION,
            help="Share of total training examples to use.",
        )
        return parser

    def prepare_data(self, *args, **kwargs):
        """Define steps that should be done on only one GPU, like getting data."""
        # Don't load/prep data if overfitting to 1 batch, test data is loaded in setup()
        if self.args.get("overfit_batches") == 1:
            return
        # Avoid loading data again in trainer.test() call.
        elif self.trainer and self.trainer.testing:
            return
        else:
            get_training_dataset()

    def setup(self, stage: str = None) -> None:
        """Define steps that should be done on every GPU, like splitting data,
        applying transform etc."""
        if stage and stage != "fit":
            # Setup for all stages is done in trainer.fit()
            return

        if self.args.get("overfit_batches", 0) == 1:
            data = pd.read_csv(TEST_DATA_DIRNAME / "test_data.csv")
        else:
            data = pd.read_feather(self.data_file)

        total_rows = len(data)
        data = data.sample(frac=self.data_fraction, random_state=self.seed)
        print(f"Using {len(data)} of {total_rows} examples.")

        data_train, data_val, data_test = split_data(
            data, TRAIN_FRAC, VAL_FRAC, self.seed
        )

        self.data_train = MT5EncodingsDataset(
            data=data.iloc[list(data_train.indices)]["text"].tolist(),
            targets=data.iloc[list(data_train.indices)]["title"].tolist(),
        )
        self.data_val = MT5EncodingsDataset(
            data=data.iloc[list(data_val.indices)]["text"].tolist(),
            targets=data.iloc[list(data_val.indices)]["title"].tolist(),
        )
        self.data_test = MT5EncodingsDataset(
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
    load_and_print_info(MotionsDataModule)
