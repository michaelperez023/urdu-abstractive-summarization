"""Chart of a training run from its metrics.json: validation ROUGE F1 and loss after each epoch.

The notebook draws it inline, and

    python -m urdu_headlines.plots results/metrics.json docs/figures/training_history.png

writes the copy shown in the README. Needs matplotlib (the notebook extra).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Chart colors: light surface, gridline and axis, text inks, two series colors.
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"

SCORES = {"rouge1_f1": "ROUGE-1", "rouge2_f1": "ROUGE-2", "rougeL_f1": "ROUGE-L"}
LINE = {
    "linewidth": 2,
    "solid_capstyle": "round",
    "solid_joinstyle": "round",
    "marker": "o",
    "markersize": 7,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 1.5,
}


def _style(ax, epochs: list[float]) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.3, max(epochs) + 0.8)
    ax.set_xlabel("epoch", color=INK_2)


def _end_label(ax, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK_2)


def training_history(metrics: dict):
    """One panel per validation ROUGE F1 score, each with the summary-as-title baseline as
    a gray line, and a panel with the training and validation loss. Returns the figure."""
    history = metrics["history"]
    # Whole epochs as integers, so that the ticks read 1, 2, 3 rather than 1.0, 2.0, 3.0.
    epochs = [int(e) if float(e).is_integer() else e for e in (row["epoch"] for row in history)]
    baseline = metrics["baseline"]["validation"]
    top = 1.15 * max(max(row[key], baseline[key]) for row in history for key in SCORES)

    fig, (*score_axes, ax_loss) = plt.subplots(1, 4, figsize=(10, 3.2), facecolor=SURFACE)
    for ax, (key, name) in zip(score_axes, SCORES.items(), strict=True):
        _style(ax, epochs)
        values = [row[key] for row in history]
        ax.axhline(baseline[key], color=MUTED, linewidth=1)
        ax.annotate(
            "summary as title",
            (epochs[0], baseline[key]),
            xytext=(0, -4),
            textcoords="offset points",
            va="top",
            color=INK_2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
        )
        ax.plot(epochs, values, color=BLUE, **LINE)
        _end_label(ax, epochs[-1], values[-1], f"{values[-1]:.2f}")
        ax.set_ylim(0, top)
        ax.set_title(f"Validation {name} F1", loc="left", color=INK)

    _style(ax_loss, epochs)
    for key, label, color in (("loss", "validation", BLUE), ("train_loss", "training", ORANGE)):
        # train_loss is None if no training loss was logged during the epoch.
        xs = [e for e, row in zip(epochs, history, strict=True) if row[key] is not None]
        ys = [row[key] for row in history if row[key] is not None]
        if not xs:
            continue
        ax_loss.plot(xs, ys, color=color, label=label, **LINE)
        _end_label(ax_loss, xs[-1], ys[-1], label)
    ax_loss.legend(frameon=False, labelcolor=INK_2, loc="upper right")
    ax_loss.set_title("Loss", loc="left", color=INK)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the training history in a metrics.json.")
    parser.add_argument("metrics", type=Path, nargs="?", default=Path("results/metrics.json"))
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("docs/figures/training_history.png")
    )
    args = parser.parse_args(argv)

    plt.switch_backend("agg")
    fig = training_history(json.loads(args.metrics.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
