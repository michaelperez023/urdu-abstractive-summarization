"""Baseline from the original notebook: use the summary itself as the title.

The notebook took the first three sentences of the summary (split on the Urdu full stop
"۔"). Most XL-Sum summaries are a single sentence, so this is usually the whole summary.

    python -m urdu_headlines.baseline --data-dir data
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from .data import DEFAULT_DATA_DIR, download, load_splits
from .metrics import format_scores, rouge_scores

URDU_FULL_STOP = "۔"


def lead_sentences(text: str, n: int = 3) -> str:
    """The first `n` sentences of `text`, split on the Urdu full stop (which is dropped)."""
    sentences = [s.strip() for s in text.split(URDU_FULL_STOP)]
    return " ".join([s for s in sentences if s][:n])


def baseline_titles(summaries: Iterable[str], n: int = 3) -> list[str]:
    return [lead_sentences(summary, n) for summary in summaries]


def evaluate_baseline(dataset, n: int = 3) -> dict[str, float]:
    """ROUGE of the lead-sentences baseline against the reference titles of `dataset`."""
    return rouge_scores(baseline_titles(dataset["summary"], n), dataset["title"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the summary-as-title baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--output", type=Path, help="optional JSON file for the scores")
    args = parser.parse_args(argv)

    download(args.data_dir)
    splits = load_splits(args.data_dir)
    results = {}
    for split in args.splits:
        results[split] = evaluate_baseline(splits[split])
        print(f"{split:<10} n={len(splits[split])}  {format_scores(results[split])}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
