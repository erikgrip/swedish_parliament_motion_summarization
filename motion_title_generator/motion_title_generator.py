from pathlib import Path
import argparse
import json
import io

import torch
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM


from motion_title_generator.data.motions_data_module import MotionsDataModule
from motion_title_generator.lit_models.t5 import MT5LitModel
from motion_title_generator.models.t5 import MT5


LOCALE_ENCODING = getattr(io, "LOCALE_ENCODING", "utf-8")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "motion_title_generator"


class MotionTitleGenerator:
    """Class to generate a title for a motion text."""

    def __init__(self):
        with open(ARTIFACT_DIR / "config.json", "r", encoding=LOCALE_ENCODING) as f:
            args = argparse.Namespace(**json.load(f))
        data = MotionsDataModule(args)
        model = MT5(data_config=data.config(), args=args)
        model.model = AutoModelForSeq2SeqLM.from_pretrained(
            "erikgrip2/mt5-finetuned-for-motion-title"
        )
        self.lit_model = MT5LitModel(model, args)

    @torch.no_grad()
    def predict(self, text: str) -> str:
        """Generate a title for an input text."""
        return self.lit_model.predict(text)


def main():
    """
    Run the paragraph text recognizer. Example runs:
    ```
    python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
    python text_recognizer/paragraph_text_recognizer.py
           https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    """
    parser = argparse.ArgumentParser(
        description="Generate a title for a Swedish Parliament Motion."
    )
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    text_recognizer = MotionTitleGenerator()
    pred_str = text_recognizer.predict(args.text)
    print(pred_str)


if __name__ == "__main__":
    main()
