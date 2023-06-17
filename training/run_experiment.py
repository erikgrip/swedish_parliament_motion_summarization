# type: ignore
"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import pytorch_lightning as pl
import torch

from motion_title_generator import lit_models

DEFAULT_DATA_CLASS = "MotionsDataModule"
DEFAULT_MODEL_CLASS = "MT5"
DEFAULT_EARLY_STOPPING = 10

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'motion_title_generator.models.t5'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add pl.Trainer args to use
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument(
        "--accelerator", default="auto", help="Lightning Trainer accelerator"
    )
    trainer_group.add_argument("--devices", default="auto", help="Number of GPUs")
    trainer_group.add_argument("--max_epochs", type=int, default=-1)
    trainer_group.add_argument("--fast_dev_run", type=bool, default=False)
    trainer_group.add_argument("--overfit_batches", type=float, default=0.0)

    # Basic arguments
    parser.add_argument("--data_class", type=str, default=DEFAULT_DATA_CLASS)
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"motion_title_generator.data.{temp_args.data_class}")
    model_class = _import_class(
        f"motion_title_generator.models.{temp_args.model_class}"
    )

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py \
        --max_epochs=3 \
        --devices=0 \
        --num_workers=20
        --model_class=MT5 \
        --data_class=MotionsDataModule
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"motion_title_generator.data.{args.data_class}")
    model_class = _import_class(f"motion_title_generator.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.model_class == "MT5":
        lit_model_class = lit_models.MT5LitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    # There's no available val_loss when overfitting to batches
    if args.overfit_batches:
        loss_to_log = "train_loss"
        enable_checkpointing = False
    else:
        loss_to_log = "val_loss"
        enable_checkpointing = True

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=loss_to_log, mode="min", patience=args.early_stopping
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.2f}",
        monitor=loss_to_log,
        mode="min",
    )
    callbacks = (
        [early_stopping_callback, model_checkpoint_callback]
        if not args.overfit_batches
        else [early_stopping_callback]
    )

    trainer = Trainer(
        precision="bf16-mixed",
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        callbacks=callbacks,  # type: ignore
        logger=logger,
        enable_checkpointing=enable_checkpointing,
    )
    # pylint: disable=no-member
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)


if __name__ == "__main__":
    main()
