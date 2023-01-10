"""General utility stuff."""

from urllib.request import urlretrieve

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py."""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1
    ) as t:  # pylint: disable=C0103
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def tokenize(tokenizer, text, text_max_num_tokens):
    """Get encoding of an input text."""
    return tokenizer(
        text,
        max_length=text_max_num_tokens,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )


def summarize(model, tokenizer, text_encoding, summary_max_num_tokens):
    """Generate summary for an input text encoding."""
    model.eval()
    generated_ids = model.model.generate(
        input_ids=text_encoding["text_input_ids"],
        attention_mask=text_encoding["text_attention_mask"],
        max_length=summary_max_num_tokens,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )

    preds = [
        tokenizer.decode(
            gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for gen_id in generated_ids
    ]
    model.train()
    return "".join(preds)
