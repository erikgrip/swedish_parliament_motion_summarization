import argparse
import logging
import os
import shutil

from huggingface_hub import hf_api

from utils.checkpoint import (
    get_local_file_paths,
    load_litmodel_from_checkpoint,
    load_tokenizer,
)

TMP_SAVE_DIR = "tmp/hf_upload"

logging.basicConfig(level=logging.INFO)
logging.getLogger("torch").setLevel(logging.WARNING)


def hf_login(user):
    """Login to Huggingface as given user via terminal prompt."""
    current_user = os.popen("huggingface-cli whoami").read().strip()
    if not current_user == user:
        os.system("huggingface-cli logout")
        os.system("huggingface-cli login")


def push_model_dir_to_hf(hf_model):
    """Upload local model direktory to Huggingface."""
    logging.info("Pushing model files to %s...", hf_model)
    api = hf_api.HfApi()
    api.upload_folder(
        folder_path=TMP_SAVE_DIR,
        path_in_repo=".",
        repo_id=hf_model,
        repo_type="model",
    )
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=int,
        help="the version in the directory name in yor lightning_logs directory",
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
    tokenizer = load_tokenizer(lit_model.model.model_name)

    lit_model.model.model.save_pretrained(TMP_SAVE_DIR)
    tokenizer.save_pretrained(TMP_SAVE_DIR)
    push_model_dir_to_hf(args.hf_model)
    shutil.rmtree(TMP_SAVE_DIR)
