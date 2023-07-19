import argparse
from utils.checkpoint import (
    get_local_file_paths,
    load_litmodel_from_checkpoint,
    load_tokenizer,
)

ARTIFACTS_PATH = "motion_title_generator/artifacts"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=int,
        help="the version in the directory name in yor lightning_logs directory",
    )
    args = parser.parse_args()

    chkpt_path, config_path = get_local_file_paths(args.version)
    lit_model = load_litmodel_from_checkpoint(chkpt_path, config_path)
    tokenizer = load_tokenizer(lit_model.model.model_name)

    # Save model and tokenizer to local artifact directory
    chkpt_abbr = (
        chkpt_path.split("/")[-1]
        .replace(".ckpt", "")
        .replace("=", "")
        .replace("-", "_")
    )
    artifact_dir = f"{ARTIFACTS_PATH}/version{args.version}_{chkpt_abbr}"
    lit_model.model.model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
