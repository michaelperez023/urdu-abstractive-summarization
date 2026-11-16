import pytest
from rouge_score import rouge_scorer, tokenizers

from urdu_headlines.metrics import format_scores, rouge_scores, tokenize

REFERENCE = "پانچ کروڑ کہاں گئے؟"
PREDICTION = "پاچ کروڑ کہاں گئے؟"  # first word misspelled
SCORE_KEYS = [
    f"{r}_{p}" for r in ("rouge1", "rouge2", "rougeL") for p in ("precision", "recall", "f1")
]


def test_default_rouge_tokenizer_drops_urdu_text():
    # rouge-score (used by Hugging Face evaluate) keeps only [a-z0-9] by default.
    assert tokenizers.DefaultTokenizer().tokenize(REFERENCE) == []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    assert scorer.score(REFERENCE, REFERENCE)["rouge1"].fmeasure == 0.0
    assert scorer.score(REFERENCE, PREDICTION)["rougeL"].fmeasure == 0.0


def test_whitespace_tokenizer_keeps_urdu_words():
    assert tokenize(REFERENCE) == ["پانچ", "کروڑ", "کہاں", "گئے؟"]


def test_one_wrong_word():
    scores = rouge_scores([PREDICTION], [REFERENCE])
    assert scores["rouge1_f1"] == pytest.approx(75.0)
    assert scores["rouge2_f1"] == pytest.approx(200 / 3, abs=1e-3)
    assert scores["rougeL_f1"] == pytest.approx(75.0)


def test_identical_strings_score_100():
    scores = rouge_scores([REFERENCE], [REFERENCE])
    assert all(scores[key] == pytest.approx(100.0) for key in SCORE_KEYS)


def test_short_prediction_has_full_precision_and_half_recall():
    scores = rouge_scores(["پانچ کروڑ"], [REFERENCE])
    assert scores["rouge1_precision"] == pytest.approx(100.0)
    assert scores["rouge1_recall"] == pytest.approx(50.0)
    assert scores["rouge1_f1"] == pytest.approx(200 / 3, abs=1e-3)


def test_recall_rewards_long_outputs_but_f1_does_not():
    long_prediction = REFERENCE + " " + " ".join(["خبر"] * 12)  # 16 words, 4 of them correct
    scores = rouge_scores([long_prediction], [REFERENCE])
    assert scores["rouge1_recall"] == pytest.approx(100.0)
    assert scores["rouge1_precision"] == pytest.approx(25.0)
    assert scores["rouge1_f1"] == pytest.approx(40.0)


def test_empty_predictions_score_zero_and_are_counted():
    scores = rouge_scores(["", "   "], [REFERENCE, REFERENCE])
    assert scores["empty_predictions"] == 2
    assert all(scores[key] == 0.0 for key in SCORE_KEYS)


def test_scores_are_averaged_over_examples():
    scores = rouge_scores([REFERENCE, ""], [REFERENCE, REFERENCE])
    assert scores["rouge1_f1"] == pytest.approx(50.0)
    assert scores["empty_predictions"] == 1
    assert scores["pred_words"] == 2.0
    assert scores["ref_words"] == 4.0


def test_bad_input_raises():
    with pytest.raises(ValueError):
        rouge_scores([PREDICTION], [REFERENCE, REFERENCE])
    with pytest.raises(ValueError):
        rouge_scores([], [])


def test_format_scores():
    line = format_scores(rouge_scores([PREDICTION], [REFERENCE]))
    assert line.startswith("ROUGE-1 F1 75.00 (P 75.00 R 75.00) | ROUGE-2 F1 66.67")
