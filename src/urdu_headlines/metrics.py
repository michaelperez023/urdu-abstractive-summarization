"""ROUGE for Urdu text.

`rouge-score`, which Hugging Face `evaluate`'s "rouge" metric also uses, tokenizes by
lowercasing and replacing every character outside [a-z0-9] with a space. Urdu words are
dropped and only Latin letters and digits are compared, so a title without them scores 0,
even against itself. Here the same scorer is given a tokenizer that splits on whitespace
and keeps every character.

Scores are precision, recall and F1 for ROUGE-1, ROUGE-2 and ROUGE-L, averaged over
examples and scaled to 0-100. F1 is the main number: recall alone rewards long outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

from rouge_score import rouge_scorer

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
PARTS = ("precision", "recall", "f1")
NAMES = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization; Urdu characters and attached punctuation are kept."""
    return text.split()


class WhitespaceTokenizer:
    """Tokenizer object for `rouge_scorer.RougeScorer` (it only needs `tokenize`)."""

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Mean ROUGE-1/2/L precision, recall and F1 (0-100) over aligned predictions/references.

    An empty prediction scores 0 on every metric; `empty_predictions` counts them.
    `pred_words` and `ref_words` are the mean lengths in words.
    """
    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but {len(references)} references")
    if not predictions:
        raise ValueError("no examples to score")

    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), tokenizer=WhitespaceTokenizer())
    totals = {f"{rouge}_{part}": 0.0 for rouge in ROUGE_TYPES for part in PARTS}
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(reference, prediction)
        for rouge in ROUGE_TYPES:
            score = scores[rouge]
            totals[f"{rouge}_precision"] += score.precision
            totals[f"{rouge}_recall"] += score.recall
            totals[f"{rouge}_f1"] += score.fmeasure

    n = len(predictions)
    result = {name: round(100 * total / n, 4) for name, total in totals.items()}
    result["empty_predictions"] = sum(not tokenize(p) for p in predictions)
    result["pred_words"] = round(sum(len(tokenize(p)) for p in predictions) / n, 2)
    result["ref_words"] = round(sum(len(tokenize(r)) for r in references) / n, 2)
    return result


def format_scores(scores: dict[str, float], prefix: str = "") -> str:
    """One line such as 'ROUGE-1 F1 30.12 (P 31.0 R 29.9) | ROUGE-2 ...'."""
    parts = []
    for rouge in ROUGE_TYPES:
        f1, p, r = (scores[f"{prefix}{rouge}_{part}"] for part in ("f1", "precision", "recall"))
        parts.append(f"{NAMES[rouge]} F1 {f1:.2f} (P {p:.2f} R {r:.2f})")
    return " | ".join(parts)
