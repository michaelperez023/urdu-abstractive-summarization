import pytest
from datasets import Dataset

from urdu_headlines.baseline import baseline_titles, evaluate_baseline, lead_sentences

FOUR_SENTENCES = "پہلا جملہ۔ دوسرا جملہ۔ تیسرا جملہ۔ چوتھا جملہ۔"
SUMMARY = "حکومت نے سیلاب زدگان کے لیے امداد کا اعلان کیا ہے۔"  # 11 words
TITLE = "سیلاب زدگان کے لیے امداد"  # 5 words, all in the summary


def test_lead_sentences_split_on_the_urdu_full_stop():
    assert lead_sentences(FOUR_SENTENCES) == "پہلا جملہ دوسرا جملہ تیسرا جملہ"
    assert lead_sentences(FOUR_SENTENCES, n=1) == "پہلا جملہ"


def test_single_sentence_summary_is_kept_whole():
    assert lead_sentences(SUMMARY) == SUMMARY.removesuffix("۔")


def test_text_without_sentences():
    assert lead_sentences("بغیر فل اسٹاپ") == "بغیر فل اسٹاپ"
    assert lead_sentences("") == ""
    assert lead_sentences("۔ ۔") == ""


def test_baseline_titles():
    assert baseline_titles([FOUR_SENTENCES, SUMMARY], n=1) == ["پہلا جملہ", SUMMARY[:-1]]


def test_summary_as_title_has_full_recall_but_low_precision():
    scores = evaluate_baseline(Dataset.from_dict({"summary": [SUMMARY], "title": [TITLE]}))
    assert scores["rouge1_recall"] == pytest.approx(100.0)
    assert scores["rouge1_precision"] == pytest.approx(500 / 11, abs=1e-3)
    assert scores["rouge1_f1"] == pytest.approx(62.5)
