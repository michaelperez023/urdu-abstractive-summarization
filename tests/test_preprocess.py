import numpy as np

from urdu_headlines.preprocess import decode, length_stats, token_lengths, tokenize_batch

SUMMARY = "پاکستان میں بارش سے سیلاب حکومت نے امداد کا اعلان کیا"  # 11 words
TITLE = "پانچ کروڑ کہاں گئے؟"  # 4 words


def test_token_lengths_include_eos(tiny_tokenizer):
    assert token_lengths(tiny_tokenizer, [SUMMARY, TITLE], batch_size=1) == [12, 5]


def test_tokenize_batch_truncates_in_tokens_and_keeps_eos(tiny_tokenizer):
    batch = {"summary": [SUMMARY], "title": [TITLE]}
    features = tokenize_batch(batch, tiny_tokenizer, max_source_length=6, max_target_length=3)
    eos = tiny_tokenizer.eos_token_id
    assert len(features["input_ids"][0]) == 6
    assert features["input_ids"][0][-1] == eos
    assert len(features["labels"][0]) == 3
    assert features["labels"][0][-1] == eos
    assert decode(tiny_tokenizer, features["labels"]) == ["پانچ کروڑ"]


def test_limits_measured_on_the_data_do_not_truncate(tiny_tokenizer):
    batch = {"summary": [SUMMARY, "بارش"], "title": [TITLE, "کہاں گئے؟"]}
    max_source = max(token_lengths(tiny_tokenizer, batch["summary"]))
    max_target = max(token_lengths(tiny_tokenizer, batch["title"]))
    features = tokenize_batch(batch, tiny_tokenizer, max_source, max_target)
    assert decode(tiny_tokenizer, features["input_ids"]) == batch["summary"]
    assert decode(tiny_tokenizer, features["labels"]) == batch["title"]


def test_decode_ignores_padding_from_the_trainer(tiny_tokenizer):
    ids = tiny_tokenizer([TITLE, "بارش"])["input_ids"]
    width = max(len(row) for row in ids) + 2
    padded = np.array([row + [-100] * (width - len(row)) for row in ids])
    assert decode(tiny_tokenizer, padded) == [TITLE, "بارش"]


def test_length_stats():
    stats = length_stats(list(range(1, 101)))
    assert stats == {"p50": 50, "p95": 95, "p99": 99, "max": 100}
