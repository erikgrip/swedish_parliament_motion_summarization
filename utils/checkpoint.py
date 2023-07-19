import argparse
import glob
import logging
import os

import yaml

from motion_title_generator.lit_models import MT5LitModel
from motion_title_generator.models import t5

BASE_PATH = "training/logs/lightning_logs/"


def get_local_file_paths(version):
    """Return file paths of checkpoint and config files in Lightning logs dir."""
    version_path = BASE_PATH + f"version_{version}"
    checkpoint_filenames = list(glob.glob(version_path + "/checkpoints/*.ckpt"))
    cfg_path = version_path + "/hparams.yaml"

    num_chkpts = len(checkpoint_filenames)
    if num_chkpts != 1:
        raise ValueError(
            f"""Found {num_chkpts} checkpoints in {version_path} + "/checkpoints/"""
        )

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No model config found at {cfg_path}")
    return checkpoint_filenames[0], cfg_path


def load_litmodel(checkpoint_path, cfg_path):
    """Load Pytorch Lightning model from checkpoint and saved config file."""
    logging.info("Loading checkpoint config %s ...", cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as hparams_file:
        lightning_config = vars(argparse.Namespace(**yaml.safe_load(hparams_file)))

    model = t5.MT5(data_config={}, args=lightning_config)

    logging.info("Loading Lightning Model from checkpoint %s ...", checkpoint_path)
    lightning_model = MT5LitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
    )
    logging.info("Done!")
    return lightning_model


def add_version_arg(parser: argparse.ArgumentParser):
    """Add version argument to parser."""
    parser.add_argument(
        "--version",
        type=int,
        help="the version in the directory name in yor lightning_logs directory",
    )
    return parser
