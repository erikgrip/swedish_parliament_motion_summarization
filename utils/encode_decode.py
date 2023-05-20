def encode(text, tokenizer, max_tokens):
    """Use tokenizer to encode text."""
    return tokenizer(
        text,
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )


def generate(model, tokenizer, text_encoding, max_title_tokens):
    """Generate title for single text."""
    generated_ids = model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=max_title_tokens,
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
    return "".join(preds)
