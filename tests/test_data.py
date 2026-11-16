import json
import tarfile

import huggingface_hub
import pytest

from urdu_headlines import data
from urdu_headlines.data import SPLIT_FILES, extract_splits, has_min_words, load_splits

ALL_FILES = tuple(SPLIT_FILES.values())


def example(id_, title):
    return {"id": id_, "url": "", "title": title, "summary": "خلاصہ۔", "text": ""}


def write_archive(tmp_path, names=ALL_FILES, prefix="./"):
    """A small stand-in for the XL-Sum archive: one long and one short title per split."""
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    archive = tmp_path / "urdu.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for name in names:
            rows = [example(f"{name}-1", "ایک دو تین"), example(f"{name}-2", "ایک دو")]
            path = source / name
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            tar.add(path, arcname=prefix + name)
    return archive


def test_has_min_words():
    assert has_min_words("ایک دو تین")
    assert not has_min_words("ایک دو")


def test_extract_and_load_splits(tmp_path):
    data_dir = tmp_path / "data"
    extract_splits(write_archive(tmp_path), data_dir)
    assert sorted(p.name for p in data_dir.iterdir()) == sorted(SPLIT_FILES.values())

    splits = load_splits(data_dir, cache_dir=tmp_path / "cache")
    assert set(splits) == {"train", "validation", "test"}
    for split in splits.values():
        assert list(split["title"]) == ["ایک دو تین"]  # the two-word title is filtered out
        assert "text" not in split.column_names


def test_extract_writes_only_under_data_dir(tmp_path):
    archive = write_archive(tmp_path, prefix="../")
    data_dir = tmp_path / "out" / "data"
    extract_splits(archive, data_dir)
    assert sorted(p.name for p in data_dir.iterdir()) == sorted(SPLIT_FILES.values())
    assert not (tmp_path / "out" / "urdu_train.jsonl").exists()


def test_extract_reports_missing_files(tmp_path):
    archive = write_archive(tmp_path, names=("urdu_train.jsonl",))
    with pytest.raises(RuntimeError, match="urdu_test.jsonl"):
        extract_splits(archive, tmp_path / "data")


def test_download_skips_existing_files(tmp_path, monkeypatch):
    for name in SPLIT_FILES.values():
        (tmp_path / name).write_text("")

    def fail(*args, **kwargs):
        raise AssertionError("download attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    assert data.download(tmp_path) == tmp_path


def test_download_rejects_a_wrong_checksum(tmp_path, monkeypatch):
    archive = write_archive(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(archive))
    with pytest.raises(RuntimeError, match="SHA-256"):
        data.download(tmp_path / "data")


def test_download_extracts_a_verified_archive(tmp_path, monkeypatch):
    archive = write_archive(tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(archive))
    monkeypatch.setattr(data, "ARCHIVE_SHA256", data.sha256_file(archive))
    data_dir = data.download(tmp_path / "data")
    assert all((data_dir / name).is_file() for name in SPLIT_FILES.values())
