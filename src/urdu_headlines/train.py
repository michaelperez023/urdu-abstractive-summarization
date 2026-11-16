"""Fine-tune mT5 to write the title of a BBC Urdu article from its summary.

    python -m urdu_headlines.train --epochs 4

Training runs in fp32 (mT5 is known to overflow in fp16). After every epoch the model
generates titles for the validation split; the checkpoint with the best ROUGE-L F1 is
kept, scored on the test split and saved to OUTPUT_DIR/model. Scores, settings and the
per-epoch history are written to OUTPUT_DIR/metrics.json and test predictions to
OUTPUT_DIR/test_predictions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback, TrainerCallback

from .baseline import evaluate_baseline
from .data import (
    ARCHIVE_PATH,
    DEFAULT_DATA_DIR,
    MIN_TITLE_WORDS,
    REPO_ID,
    REVISION,
    download,
    load_splits,
)
from .metrics import format_scores, rouge_scores
from .preprocess import decode, length_stats, token_lengths, tokenize_batch

TASK = "XL-Sum Urdu: generate the article title from its summary"
BEST_METRIC = "rougeL_f1"
# History fields that are not validation scores.
NOT_SCORES = {"epoch", "step", "train_loss", "runtime", "samples_per_second", "steps_per_second"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mT5 on XL-Sum Urdu (summary -> title).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="google/mt5-small", help="model name or local path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="stop after this many steps (overrides --epochs)"
    )
    parser.add_argument("--learning-rate", type=float, default=5.6e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-source-length",
        type=int,
        help="summary limit in tokens; by default the longest training summary",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        help="title limit in tokens; by default the longest training title",
    )
    parser.add_argument("--num-beams", type=int, default=1, help="1 means greedy decoding")
    parser.add_argument("--max-train-samples", type=int, help="random training subset")
    parser.add_argument("--max-eval-samples", type=int, help="random validation subset")
    parser.add_argument("--max-test-samples", type=int, help="random test subset")
    parser.add_argument("--logging-steps", type=int, default=200)
    parser.add_argument(
        "--num-examples", type=int, default=5, help="test predictions copied into metrics.json"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from the last checkpoint in OUTPUT_DIR"
    )
    parser.add_argument("--cpu", action="store_true", help="use the CPU even if a GPU is found")
    return parser.parse_args(argv)


def duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class ProgressLog(TrainerCallback):
    """Prints one line per logging step and the validation scores after each evaluation.

    Throughput is measured between training logs; the clock restarts after each
    evaluation and checkpoint save so that those pauses are not counted.
    """

    def __init__(self) -> None:
        self.timed_steps = 0
        self.timed_seconds = 0.0

    def _restart(self, state) -> None:
        self.window_start = time.perf_counter()
        self.window_step = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()
        self._restart(state)

    def on_save(self, args, state, control, **kwargs):
        self._restart(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"epoch {state.epoch:.2f} ended at step {state.global_step}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        now = time.perf_counter()
        if "loss" in logs:
            steps = state.global_step - self.window_step
            seconds = now - self.window_start
            self.timed_steps += steps
            self.timed_seconds += seconds
            rate = steps / seconds if seconds > 0 else float("nan")
            eta = (state.max_steps - state.global_step) / rate if rate > 0 else float("nan")
            print(
                f"step {state.global_step}/{state.max_steps} epoch {state.epoch:.2f} "
                f"loss {logs['loss']:.4f} lr {logs.get('learning_rate', float('nan')):.2e} "
                f"{rate:.2f} it/s elapsed {duration(now - self.start)} "
                f"training left {duration(eta)}",
                flush=True,
            )
            self._restart(state)
        elif "eval_loss" in logs:
            scores = {k.removeprefix("eval_"): v for k, v in logs.items() if k.startswith("eval_")}
            print(
                f"validation after epoch {state.epoch:.2f}: loss {scores['loss']:.4f} "
                f"{format_scores(scores)} ({duration(scores.get('runtime', float('nan')))})",
                flush=True,
            )
            self._restart(state)

    def steps_per_second(self) -> float | None:
        if not self.timed_seconds:
            return None
        return round(self.timed_steps / self.timed_seconds, 3)


def load_model(name_or_path: str):
    """The pre-trained model in fp32, with `tie_word_embeddings` set to match its weights.

    mT5's output layer (`lm_head`) has its own weights; only the encoder and decoder share the
    input embeddings. transformers 5 sets `tie_word_embeddings=True` in every mT5 config and
    leaves `lm_head` untied only because the checkpoint contains it, so the saved config would
    claim the weights are tied. transformers 4 follows the saved value: it ties `lm_head` to the
    input embeddings and rescales the decoder output, which ruins the predictions. transformers 5
    reads the value only when it builds the model and ties its weights, so setting it afterwards
    does not change training; transformers 4 already reads False from mT5's config.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, dtype=torch.float32)
    model.config.tie_word_embeddings = False
    return model


def subsample(dataset, n: int | None, seed: int):
    if n is None or n >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(n))


def make_compute_metrics(tokenizer, references: list[str]):
    """ROUGE against the raw reference titles (not the tokenized, possibly truncated labels)."""

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return rouge_scores(decode(tokenizer, predictions), references)

    return compute_metrics


def epoch_history(log_history: list[dict]) -> list[dict]:
    """One row per evaluation: mean training loss since the previous one, then eval metrics."""
    rows, losses = [], []
    for entry in log_history:
        if "loss" in entry:
            losses.append(entry["loss"])
        elif "eval_loss" in entry:
            row = {
                "epoch": round(entry["epoch"], 2),
                "step": entry["step"],
                "train_loss": round(float(np.mean(losses)), 4) if losses else None,
            }
            row.update(
                {k.removeprefix("eval_"): v for k, v in entry.items() if k.startswith("eval_")}
            )
            rows.append(row)
            losses = []
    return rows


def checkpoint_step(path: str | None) -> int | None:
    return int(Path(path).name.rsplit("-", 1)[-1]) if path else None


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    download(args.data_dir)
    splits = load_splits(args.data_dir)
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_eval_samples,
        "test": args.max_test_samples,
    }
    for split, n in limits.items():
        splits[split] = subsample(splits[split], n, args.seed)
    sizes = {split: len(dataset) for split, dataset in splits.items()}
    print(f"examples: {sizes}", flush=True)

    # Length limits in mT5 tokens, measured on the training split.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    summary_lengths = token_lengths(tokenizer, splits["train"]["summary"])
    title_lengths = token_lengths(tokenizer, splits["train"]["title"])
    max_source_length = args.max_source_length or max(summary_lengths)
    max_target_length = args.max_target_length or max(title_lengths)
    # Generated sequences start with the decoder start token, hence one extra position.
    generation_max_length = max_target_length + 1
    print(
        f"training tokens: summary {length_stats(summary_lengths)}, "
        f"title {length_stats(title_lengths)}; max_source_length {max_source_length}, "
        f"max_target_length {max_target_length}, generation_max_length {generation_max_length}",
        flush=True,
    )

    baseline = {split: evaluate_baseline(splits[split]) for split in ("validation", "test")}
    for split, scores in baseline.items():
        print(f"baseline ({split}): {format_scores(scores)}", flush=True)

    tokenized = splits.map(
        tokenize_batch,
        batched=True,
        remove_columns=splits["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
        },
    )

    model = load_model(args.model)
    model.generation_config.max_length = generation_max_length
    model.generation_config.num_beams = args.num_beams

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=args.num_beams,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        fp16=False,
        bf16=False,
        report_to="none",
        disable_tqdm=True,
        use_cpu=args.cpu,
    )
    progress = ProgressLog()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer, list(splits["validation"]["title"])),
        callbacks=[progress],
    )
    trainer.remove_callback(PrinterCallback)

    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)

    # load_best_model_at_end has restored the checkpoint with the best validation ROUGE-L F1.
    tokenizer.model_max_length = max_source_length
    trainer.save_model(str(output_dir / "model"))
    (output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )
    history = epoch_history(trainer.state.log_history)
    best_step = checkpoint_step(trainer.state.best_model_checkpoint)
    best = next((row for row in history if row["step"] == best_step), history[-1])
    validation = {k: v for k, v in best.items() if k not in NOT_SCORES}

    print("generating titles for the test split", flush=True)
    trainer.compute_metrics = None
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_split = splits["test"]
    predictions = decode(tokenizer, test_output.predictions)
    test = rouge_scores(predictions, test_split["title"])
    test["loss"] = round(test_output.metrics["test_loss"], 4)

    with open(output_dir / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for example, prediction in zip(test_split, predictions, strict=True):
            record = {
                "id": example["id"],
                "summary": example["summary"],
                "title": example["title"],
                "prediction": prediction,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(test_split), size=min(args.num_examples, len(test_split)), replace=False)
    examples = [
        {
            "id": test_split[int(i)]["id"],
            "summary": test_split[int(i)]["summary"],
            "title": test_split[int(i)]["title"],
            "prediction": predictions[int(i)],
        }
        for i in sorted(picks)
    ]

    cuda = training_args.device.type == "cuda"
    results = {
        "task": TASK,
        "model": args.model,
        "data": {
            "source": f"{REPO_ID}/{ARCHIVE_PATH}",
            "revision": REVISION,
            "min_title_words": MIN_TITLE_WORDS,
            "examples": sizes,
        },
        "metric": "ROUGE with whitespace tokenization (rouge-score), mean over examples, 0-100",
        "config": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "precision": "fp32",
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "generation_max_length": generation_max_length,
            "num_beams": args.num_beams,
            "best_model_metric": f"validation {BEST_METRIC}",
        },
        "train_token_lengths": {
            "summary": length_stats(summary_lengths),
            "title": length_stats(title_lengths),
        },
        "baseline": baseline,
        "best_checkpoint": {"epoch": best["epoch"], "step": best["step"]},
        "validation": validation,
        "test": test,
        "history": history,
        "runtime": {
            "device": torch.cuda.get_device_name() if cuda else training_args.device.type,
            "train_seconds": round(train_result.metrics["train_runtime"], 1),
            "train_steps": trainer.state.global_step,
            "train_steps_per_second": progress.steps_per_second(),
            "test_seconds": round(test_output.metrics["test_runtime"], 1),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if cuda
            else None,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
        "examples": examples,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"best checkpoint: epoch {best['epoch']} (step {best['step']})")
    print(f"validation:      {format_scores(validation)}")
    print(f"test, baseline:  {format_scores(baseline['test'])}")
    print(f"test, model:     {format_scores(test)}")
    print(f"wrote {metrics_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
