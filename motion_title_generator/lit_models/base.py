import argparse

import lightning as L
import torch


OPTIMIZER = "Adam"
LR = 1e-3
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(L.LightningModule):  # pylint: disable=too-many-ancestors
    """Generic PyTorch-Lightning class that must be initialized
    with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.save_hyperparameters(self.args)
        self.model = model

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)  # pylint: disable=invalid-name

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

    @staticmethod
    def add_to_argparse(parser):  # pylint: disable=missing-function-docstring
        parser.add_argument(
            "--optimizer",
            type=str,
            default=OPTIMIZER,
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument(
            "--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS
        )
        return parser

    def configure_optimizers(self):
        """Initialize optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # pylint: disable=arguments-differ,missing-function-docstring
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        raise NotImplementedError

    # pylint: disable=arguments-differ, unused-argument, missing-function-docstring
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    # pylint: disable=arguments-differ, unused-argument, missing-function-docstring
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    # pylint: disable=arguments-differ, unused-argument, missing-function-docstring
    def test_step(self, batch, batch_idx):
        raise NotImplementedError
