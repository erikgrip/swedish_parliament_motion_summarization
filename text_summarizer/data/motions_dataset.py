from typing import Any, Tuple

from transformers import T5Tokenizer

from .util import BaseDataset


DEFAULT_TOKENIZER = T5Tokenizer.from_pretrained("t5-base")


class SwedishParliamentMotionsDataset(BaseDataset):
    def __init__(
        self,
        data,
        targets,
        transform=None,
        target_transform=None,
        tokenizer=DEFAULT_TOKENIZER,
        data_max_token_length=256,
        target_max_token_length=32,
    ) -> None:
        super().__init__(data, targets, transform, target_transform)
        self.tokenizer = tokenizer
        self.data_max_token_length = data_max_token_length
        self.target_max_token_length = target_max_token_length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        text, summary = super().__getitem__(index)

        text_encoding = self.tokenizer(
            text,
            max_length=self.data_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.target_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=summary,
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten(),
        )


if __name__ == "__main__":
    data = ["This is my first sentence", "And here's my 2nd one"]
    targets = ["The first", "The second"]

    ds = SwedishParliamentMotionsDataset(data=data, targets=targets)
