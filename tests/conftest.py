"""Shared fixtures. Nothing here downloads a model or the dataset."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

WORDS = "پانچ کروڑ کہاں گئے؟ پاکستان میں بارش سے سیلاب حکومت نے امداد کا اعلان کیا".split()


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Word-level tokenizer built in memory that appends </s> to every sequence, like mT5's."""
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
