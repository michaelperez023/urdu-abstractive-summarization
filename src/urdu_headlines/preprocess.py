"""Tokenization for the summary -> title task, and decoding of generated ids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def token_lengths(tokenizer, texts: Sequence[str], batch_size: int = 1000) -> list[int]:
    """Number of tokens in each text, including special tokens such as </s>."""
    texts = list(texts)
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size])
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def length_stats(lengths: Sequence[int]) -> dict[str, int]:
    values = np.asarray(lengths)
    stats = {f"p{q}": int(np.percentile(values, q)) for q in (50, 95, 99)}
    stats["max"] = int(values.max())
    return stats


def tokenize_batch(batch, tokenizer, max_source_length: int, max_target_length: int):
    """Summaries become the model input and titles the labels, each truncated in tokens."""
    features = tokenizer(batch["summary"], max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=batch["title"], max_length=max_target_length, truncation=True)
    features["labels"] = labels["input_ids"]
    return features


def decode(tokenizer, token_ids) -> list[str]:
    """Decode rows of generated token ids to text.

    The Trainer pads predictions from different batches with -100, which the tokenizer
    cannot decode, so negative ids are replaced by the pad token first.
    """
    pad = tokenizer.pad_token_id
    rows = [[pad if t < 0 else int(t) for t in row] for row in token_ids]
    return [text.strip() for text in tokenizer.batch_decode(rows, skip_special_tokens=True)]
