import argparse

from transformers.models.mt5 import MT5Tokenizer

from utils.checkpoint import add_version_arg, get_local_file_paths, load_litmodel

ARTIFACTS_PATH = "motion_title_generator/artifacts"


if __name__ == "__main__":
    parser = add_version_arg(argparse.ArgumentParser())
    args = parser.parse_args()

    chkpt_path, config_path = get_local_file_paths(args.version)
    lit_model = load_litmodel(chkpt_path, config_path)
    tokenizer = MT5Tokenizer.from_pretrained(lit_model.model.model_name)

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
