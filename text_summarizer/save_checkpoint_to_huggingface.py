import argparse
import glob
import logging
import os
import shutil
import yaml

from huggingface_hub import HfApi
from transformers.models.mt5 import MT5Tokenizer

from text_summarizer.models import t5
from text_summarizer.lit_models import MT5LitModel

logging.basicConfig(level=logging.INFO)
logging.getLogger("torch").setLevel(logging.WARNING)


BASE_PATH = "training/logs/lightning_logs/"
TMP_SAVE_DIR = "tmp/hf_upload"


def get_local_file_paths(version):
    """Return file paths of checkpoint and config files in Lightning logs dir."""
    checkpoint_filenames = list(
        glob.glob(BASE_PATH + f"version_{version}/checkpoints/*.ckpt")
    )
    cfg_path = BASE_PATH + f"version_{version}/hparams.yaml"

    if len(checkpoint_filenames) != 1:
        raise Exception(
            f"Found {len(checkpoint_filenames)} found in {BASE_PATH + f'version_{version}/checkpoints/'}"
        )
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No model config found at {cfg_path}")
    return checkpoint_filenames[0], cfg_path


def load_litmodel_from_checkpoint(chkpt_path, cfg_path):
    """Load Pytorch Lightning model from checkpoint and saved config file"""

    logging.info(f"Loading checkpoint config {cfg_path} ...")
    with open(cfg_path, "r", encoding="utf-8") as hparams_file:
        lightning_config = argparse.Namespace(
            **yaml.load(hparams_file, Loader=yaml.Loader)
        )

    model = t5.MT5(data_config={}, args=lightning_config)

    logging.info(f"Loading Lightning Model from checkpoint {chkpt_path} ...")
    lit_model = MT5LitModel.load_from_checkpoint(
        checkpoint_path=chkpt_path,
        model=model,
    )
    logging.info("Done!")
    return lit_model


def load_tokenizer(model):
    """Return tokenizer object for a given MT5 model"""
    return MT5Tokenizer.from_pretrained(model.model_name)


def save_local_model(lit_model, tokenizer):
    "Save tokenizer and Pytorch Lightning model to a local directory."
    tokenizer.save_pretrained(TMP_SAVE_DIR)
    lit_model.model.model.save_pretrained(TMP_SAVE_DIR)


def hf_login(user):
    """Login to Huggingface as given user via terminal prompt."""
    current_user = os.popen("huggingface-cli whoami").read().strip()
    if not current_user == user:
        os.system("huggingface-cli logout")
        os.system("huggingface-cli login")


def push_model_dir_to_hf(hf_model):
    """Upload local model direktory to Huggingface"""
    logging.info(f"Pushing model files to {hf_model}...")
    api = HfApi()
    api.upload_folder(
        folder_path=TMP_SAVE_DIR,
        path_in_repo=".",
        repo_id=hf_model,
        repo_type="model",
    )
    logging.info("Done!")


def delete_tmp_dir():
    """Delete the local temporary model directory"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=int,
        help="the version in the directory name in training/logs/lightning_logs/",
    )
    parser.add_argument(
        "--hf_user",
        type=str,
        help="the huggingface user to login as",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        help="the user's huggingface repo to push the model to, e.g. 'someuser/a-model",
    )
    args = parser.parse_args()

    hf_login(args.hf_user)
    chkpt_path, config_path = get_local_file_paths(args.version)
    lit_model = load_litmodel_from_checkpoint(chkpt_path, config_path)
    tokenizer = load_tokenizer(lit_model.model)
    save_local_model(lit_model, tokenizer)
    push_model_dir_to_hf(args.hf_model)
    shutil.rmtree(TMP_SAVE_DIR)
