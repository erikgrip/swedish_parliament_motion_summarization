from pathlib import Path
from typing import Dict, Optional, Union

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from motion_title_generator.data.base_dataset import BaseDataset

BATCH_SIZE = 8
NUM_WORKERS = 0


class BaseDataModule(L.LightningDataModule):
    """
    Base DataModule. Learn more at
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: Optional[Dict] = None) -> None:
        super().__init__()
        self.args = args if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        """Return Path relative to where this script is stored."""
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        return parser

    def config(self):
        """Return settings of the dataset, to be passed to instantiate models.

        For example:
            return {
                "input_dims": self.dims,
                "output_dims": self.output_dims,
                "mapping": self.mapping,
            }
        """
        return {}

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk,
        or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.

        Should assign `torch Dataset` objects to self.data_train,
        self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        """Return a DataLoader for training."""
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        """Return a DataLoader for validation."""
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        """Return a DataLoader for test data."""
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
