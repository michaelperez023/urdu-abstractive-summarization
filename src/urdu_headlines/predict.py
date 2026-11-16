"""Generate titles with a model fine-tuned by `urdu_headlines.train`.

    python -m urdu_headlines.predict --model outputs/model --n 5
    python -m urdu_headlines.predict --model outputs/model --summary "..."

Without --summary, random examples from the chosen split are shown next to their
reference titles.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import DEFAULT_DATA_DIR, download, load_splits
from .metrics import rouge_scores
from .preprocess import decode


def generate_titles(
    model, tokenizer, summaries: Sequence[str], batch_size: int = 16, **generate_kwargs
) -> list[str]:
    """One title per summary. Generation settings default to the model's generation config,
    which `urdu_headlines.train` saves with the model."""
    model.eval()
    summaries = list(summaries)
    titles: list[str] = []
    for start in range(0, len(summaries), batch_size):
        inputs = tokenizer(
            summaries[start : start + batch_size],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        titles.extend(decode(tokenizer, output.cpu().numpy()))
    return titles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate titles with a fine-tuned model.")
    parser.add_argument("--model", default="outputs/model", help="directory saved by training")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=5, help="number of random examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--summary", action="append", help="generate a title for this text (repeatable)"
    )
    parser.add_argument("--num-beams", type=int, help="override the saved beam size")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    if args.summary:
        summaries = args.summary
        references: list[str | None] = [None] * len(summaries)
    else:
        download(args.data_dir)
        split = load_splits(args.data_dir)[args.split]
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(len(split), size=min(args.n, len(split)), replace=False)
        rows = [split[int(i)] for i in picks]
        summaries = [row["summary"] for row in rows]
        references = [row["title"] for row in rows]

    generate_kwargs = {"num_beams": args.num_beams} if args.num_beams else {}
    titles = generate_titles(model, tokenizer, summaries, **generate_kwargs)
    rows = zip(summaries, references, titles, strict=True)
    for i, (summary, reference, title) in enumerate(rows, 1):
        print(f"[{i}] summary:   {summary}")
        if reference is not None:
            f1 = rouge_scores([title], [reference])["rougeL_f1"]
            print(f"    reference: {reference}")
            print(f"    generated: {title}  (ROUGE-L F1 {f1:.1f})")
        else:
            print(f"    generated: {title}")
        print()


if __name__ == "__main__":
    main()
