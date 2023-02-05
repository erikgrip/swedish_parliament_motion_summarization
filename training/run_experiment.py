# type: ignore
"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from text_summarizer import lit_models


DEFAULT_DATA_CLASS = "MotionsDataModule"
DEFAULT_MODEL_CLASS = "MT5"
DEFAULT_EARLY_STOPPING = 10


# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'motion_summarizer.models.t5'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[  # pylint: disable=protected-access
        1
    ].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default=DEFAULT_DATA_CLASS)
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_summarizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_summarizer.models.{temp_args.model_class}")

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
        --gpus=0 \
        --num_workers=20
        --model_class=MT5 \
        --data_class=MotionsDataModule
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_summarizer.data.{args.data_class}")
    model_class = _import_class(f"text_summarizer.models.{args.model_class}")
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
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

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
        dirpath="training/logs",
        filename="{epoch:03d}-{val_loss:.2f}",
        monitor=loss_to_log,
        mode="min",
    )
    callbacks = (
        [early_stopping_callback, model_checkpoint_callback]
        if not args.overfit_batches
        else [early_stopping_callback]
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
    )

    # pylint: disable=no-member
    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")


if __name__ == "__main__":
    main()
