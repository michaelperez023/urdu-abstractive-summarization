"""The saved config must say that mT5's output layer has its own weights.

These tests need torch 2.6 or newer and build a tiny random mT5 on the CPU; nothing is
downloaded.
"""

import json

import pytest

torch = pytest.importorskip("torch", minversion="2.6")

from transformers import AutoModelForSeq2SeqLM, MT5Config, MT5ForConditionalGeneration  # noqa: E402

from urdu_headlines.train import load_model  # noqa: E402

INPUTS = {"input_ids": [[5, 6, 7, 1]], "decoder_input_ids": [[0, 5, 6]]}


def save_tiny_mt5(path):
    """A tiny random mT5 saved like google/mt5-small: `lm_head` has its own weights and
    config.json says tie_word_embeddings is false. Returns the `lm_head` weights."""
    config = MT5Config(vocab_size=40, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=2)
    torch.manual_seed(0)
    model = MT5ForConditionalGeneration(config)
    model.lm_head.weight = torch.nn.Parameter(torch.randn(config.vocab_size, config.d_model))
    model.save_pretrained(path)
    config_file = path / "config.json"
    saved = json.loads(config_file.read_text())
    saved["tie_word_embeddings"] = False
    config_file.write_text(json.dumps(saved))
    return model.lm_head.weight.detach().clone()


def check_embeddings(model):
    """Encoder and decoder share the input embeddings; `lm_head` is a separate tensor."""
    shared = model.shared.weight
    assert model.encoder.embed_tokens.weight is shared
    assert model.decoder.embed_tokens.weight is shared
    assert model.lm_head.weight.data_ptr() != shared.data_ptr()


def logits(model):
    with torch.no_grad():
        return model.eval()(**{k: torch.tensor(v) for k, v in INPUTS.items()}).logits


def test_load_model_keeps_the_output_layer_untied(tmp_path):
    lm_head = save_tiny_mt5(tmp_path / "base")
    model = load_model(str(tmp_path / "base"))
    assert model.config.tie_word_embeddings is False
    assert model.dtype == torch.float32
    check_embeddings(model)
    assert torch.equal(model.lm_head.weight, lm_head)


def test_saved_model_reloads_with_its_own_output_layer(tmp_path):
    save_tiny_mt5(tmp_path / "base")
    model = load_model(str(tmp_path / "base"))
    model.save_pretrained(tmp_path / "saved")

    config = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config["tie_word_embeddings"] is False
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(tmp_path / "saved")
    check_embeddings(reloaded)
    assert torch.equal(reloaded.lm_head.weight, model.lm_head.weight)
    assert torch.equal(logits(reloaded), logits(model))
