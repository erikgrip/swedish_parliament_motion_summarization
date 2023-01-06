import argparse

from torchmetrics import Accuracy
import pytorch_lightning as pl
import torch


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """ Generic PyTorch-Lightning class that must be initialized
    with a PyTorch module. """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)  # pylint: disable=invalid-name

        loss = self.args.get("loss", LOSS)
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

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
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser

    def configure_optimizers(self):
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

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch  # pylint: disable=invalid-name
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch  # pylint: disable=invalid-name
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch  # pylint: disable=invalid-name
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
