"""Crash-safe readers and writers for the CSV / JSONL data files."""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from autopilot.dataset.schemas import Candidate

T = TypeVar("T", bound=BaseModel)

CANDIDATE_COLUMNS = ("candidate_id", "prompt")


class DataFileError(ValueError):
    """A data file exists but cannot be parsed."""


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers never see a half-written file.

    We write a temp file, flush it to disk, then rename it over the target. A rename is
    atomic, so a Ctrl+C or crash leaves either the old file or the new one, never a mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    """Atomically write one JSON object per line."""
    atomic_write_text(path, "".join(r.model_dump_json() + "\n" for r in records))


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    """Read and validate a JSONL file; a missing file is an empty list."""
    if not path.exists():
        return []
    records: list[T] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as error:
                raise DataFileError(f"{path}:{line_no}: invalid {model.__name__}: {error}") from error
    return records


def write_candidates_csv(path: Path, candidates: Iterable[Candidate]) -> None:
    """Write ``candidate_id,prompt`` rows. Multi-line prompts are quoted, so they round-trip safely."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CANDIDATE_COLUMNS)
    for candidate in candidates:
        writer.writerow((candidate.candidate_id, candidate.prompt))
    atomic_write_text(path, buffer.getvalue())


def read_candidates_csv(path: Path) -> list[Candidate]:
    """Read ``data/candidates.csv``; refuses files with unexpected columns."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_COLUMNS:
            raise DataFileError(f"{path}: expected columns {CANDIDATE_COLUMNS}, got {reader.fieldnames}")
        return [Candidate.model_validate(row) for row in reader]
