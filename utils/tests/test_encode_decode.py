import pytest
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers.testing_utils import slow


from utils.encode_decode import encode, generate

@pytest.fixture(scope="module")
def tokenizer_model():
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
    yield tokenizer, model

@slow
@pytest.mark.parametrize(
    "text, max_tokens",
    [
        ("This is a sample text.", 10),
        ("This is the first sentence. This is the second sentence.", 15),
    ]
)
def test_encode(tokenizer_model, text, max_tokens):
    tokenizer, _ = tokenizer_model
    encoding = encode(text, tokenizer, max_tokens)
    assert isinstance(encoding["input_ids"], torch.Tensor)
    assert isinstance(encoding["attention_mask"], torch.Tensor)
    assert encoding["input_ids"].shape == encoding["attention_mask"].shape
    assert encoding["input_ids"].shape[-1] == max_tokens

@slow
def test_generate_with_input(tokenizer_model):
    tokenizer, model = tokenizer_model
    text_encoding = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    title = generate(model, tokenizer, text_encoding, max_title_tokens=10)
    assert isinstance(title, str)
    assert len(title) > 0


@slow
def test_generate_without_input(tokenizer_model):
    tokenizer, model = tokenizer_model
    text_encoding = {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
    }
    title = generate(model, tokenizer, text_encoding, max_title_tokens=10)
    assert title == ""
