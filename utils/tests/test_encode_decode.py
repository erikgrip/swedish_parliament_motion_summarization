import pytest
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from utils.encode_decode import encode, generate


@pytest.fixture(scope="module", name="model")
def setup_model():
    """Fixture to load model."""
    yield MT5ForConditionalGeneration.from_pretrained("google/mt5-small")


@pytest.fixture(scope="module", name="tokenizer")
def setup_tokenizer():
    """Fixture to load tokenizer."""
    yield MT5Tokenizer.from_pretrained("google/mt5-small")


@pytest.mark.parametrize(
    "text, max_tokens",
    [
        ("This is a sample text.", 10),
        ("This is the first sentence. This is the second sentence.", 15),
    ],
)
def test_encode(tokenizer, text, max_tokens):
    """Test that encode returns tensors of correct shape."""
    encoding = encode(text, tokenizer, max_tokens)
    assert isinstance(encoding["input_ids"], torch.Tensor)
    assert isinstance(encoding["attention_mask"], torch.Tensor)
    assert encoding["input_ids"].shape == encoding["attention_mask"].shape
    assert encoding["input_ids"].shape[-1] == max_tokens


def test_generate_with_input(model, tokenizer):
    """Test that generate returns a non-empty string."""
    text_encoding = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    title = generate(model, tokenizer, text_encoding, max_title_tokens=10)
    assert isinstance(title, str)
    assert len(title) > 0


def test_generate_without_input(model, tokenizer):
    """Test that generate returns an empty string when input is empty."""
    text_encoding = {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
    }
    title = generate(model, tokenizer, text_encoding, max_title_tokens=10)
    assert title == ""
