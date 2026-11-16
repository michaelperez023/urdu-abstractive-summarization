"""Download and load the Urdu part of XL-Sum.

XL-Sum (Hasan et al., 2021) pairs BBC news articles with their titles and short summaries,
usually a single sentence. The Urdu data is released under CC BY-NC-SA 4.0, so it is
downloaded at run time and is not stored in this repository.

    python -m urdu_headlines.data --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from datasets import DatasetDict, load_dataset

REPO_ID = "csebuetnlp/xlsum"
ARCHIVE_PATH = "data/urdu_XLSum_v2.0.tar.bz2"
# Pinned dataset revision and the SHA-256 of the archive (from the Hub's LFS metadata).
REVISION = "30fece425f9a3866e04321773ca7a80056d55ca6"
ARCHIVE_SHA256 = "c15fc605d523c7987422941227e73b6ea20edd95d143bc5dcd4581bac7898f20"

SPLIT_FILES = {
    "train": "urdu_train.jsonl",
    "validation": "urdu_val.jsonl",
    "test": "urdu_test.jsonl",
}
DEFAULT_DATA_DIR = Path("data")
# Like the original notebook, keep only examples whose title has more than two words.
MIN_TITLE_WORDS = 3


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_splits(archive: str | os.PathLike, data_dir: str | os.PathLike) -> None:
    """Copy the three JSONL split files out of the archive into `data_dir`.

    Only regular files with the expected names are read, and each is written under its
    fixed name, so paths stored in the archive are never used as output paths.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = set(SPLIT_FILES.values())
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            name = PurePosixPath(member.name).name
            if not member.isfile() or name not in missing:
                continue
            source = tar.extractfile(member)
            partial = data_dir / f"{name}.part"
            with source, open(partial, "wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(data_dir / name)
            missing.discard(name)
    if missing:
        raise RuntimeError(f"{archive} does not contain {sorted(missing)}")


def download(data_dir: str | os.PathLike = DEFAULT_DATA_DIR) -> Path:
    """Download XL-Sum Urdu from the Hugging Face Hub and extract it, unless already present."""
    data_dir = Path(data_dir)
    if all((data_dir / name).is_file() for name in SPLIT_FILES.values()):
        return data_dir

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(REPO_ID, ARCHIVE_PATH, repo_type="dataset", revision=REVISION)
    checksum = sha256_file(archive)
    if checksum != ARCHIVE_SHA256:
        raise RuntimeError(f"Unexpected SHA-256 {checksum} for {archive}")
    extract_splits(archive, data_dir)
    return data_dir


def has_min_words(title: str, min_words: int = MIN_TITLE_WORDS) -> bool:
    return len(title.split()) >= min_words


def filter_short_titles(dataset: DatasetDict, min_words: int = MIN_TITLE_WORDS) -> DatasetDict:
    """Drop examples whose title has fewer than `min_words` words."""
    return dataset.filter(has_min_words, input_columns="title", fn_kwargs={"min_words": min_words})


def load_splits(
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    min_title_words: int = MIN_TITLE_WORDS,
    keep_text: bool = False,
    cache_dir: str | os.PathLike | None = None,
) -> DatasetDict:
    """Load train/validation/test from the JSONL files and drop examples with short titles.

    Each example has `id`, `url`, `title` and `summary`; the article body (`text`) is not
    used by this project and is dropped unless `keep_text` is set.
    """
    data_dir = Path(data_dir)
    files = {split: str(data_dir / name) for split, name in SPLIT_FILES.items()}
    dataset = load_dataset(
        "json", data_files=files, cache_dir=str(cache_dir) if cache_dir else None
    )
    if not keep_text:
        dataset = dataset.remove_columns("text")
    return filter_short_titles(dataset, min_title_words)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download XL-Sum Urdu and print the split sizes.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    download(args.data_dir)
    everything = load_splits(args.data_dir, min_title_words=0)
    kept = filter_short_titles(everything)
    for split in SPLIT_FILES:
        print(
            f"{split:<10} {len(everything[split]):>6} examples, "
            f"{len(kept[split]):>6} with titles of {MIN_TITLE_WORDS}+ words"
        )


if __name__ == "__main__":
    main()
