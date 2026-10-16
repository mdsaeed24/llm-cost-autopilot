"""Embedding features: caching, batching, errors, sklearn transformer. Ollama is always faked."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import numpy as np
import openai
import pytest

import autopilot.classifier.embeddings as embeddings_module
from autopilot.classifier.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    EmbeddingFeatures,
    OllamaEmbedder,
    set_embedder,
)
from autopilot.settings import EmbeddingSettings


class FakeEmbeddingsAPI:
    """Stands in for ``OpenAI().embeddings``: deterministic 4-d vectors derived from the text."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def create(self, *, model: str, input: list[str]) -> Any:  # noqa: A002 - mirrors the SDK
        self.calls.append(list(input))
        if self.fail:
            raise openai.APIConnectionError(request=httpx2.Request("POST", "http://x/v1/embeddings"))
        data = [SimpleNamespace(index=i, embedding=[len(t), t.count("?") + 1, 1.0, float(len(t) % 7)])
                for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


def make_embedder(tmp_path: Path | None = None, *, batch_size: int = 2, fail: bool = False) -> tuple[OllamaEmbedder, FakeEmbeddingsAPI]:
    api = FakeEmbeddingsAPI(fail=fail)
    cache = EmbeddingCache(tmp_path / "cache.sqlite") if tmp_path else None
    embedder = OllamaEmbedder(
        EmbeddingSettings(batch_size=batch_size), cache=cache, client=SimpleNamespace(embeddings=api)  # type: ignore[arg-type]
    )
    return embedder, api


@pytest.fixture(autouse=True)
def _restore_embedder() -> Iterator[None]:
    previous = embeddings_module._embedder
    yield
    set_embedder(previous)


def test_embeddings_are_batched_ordered_and_unit_length() -> None:
    embedder, api = make_embedder(batch_size=2)
    texts = ["a", "bb?", "ccc", "dddd", "e"]
    matrix = embedder.embed(texts, model="m")
    assert matrix.shape == (5, 4)
    assert [len(c) for c in api.calls] == [2, 2, 1]
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1)
    # row order follows the input even though the fake API answers in reverse
    for row, text in zip(matrix, texts, strict=True):
        expected = np.array([len(text), text.count("?") + 1, 1.0, float(len(text) % 7)])
        assert np.allclose(row, expected / np.linalg.norm(expected))


def test_cache_avoids_repeat_calls_and_survives_restart(tmp_path: Path) -> None:
    embedder, api = make_embedder(tmp_path)
    first = embedder.embed(["x", "y"], model="m")
    embedder.embed(["x", "y", "z"], model="m")
    assert api.calls == [["x", "y"], ["z"]]

    restarted, api2 = make_embedder(tmp_path)
    assert np.allclose(restarted.embed(["x", "y"], model="m"), first)
    assert api2.calls == []


def test_cache_is_per_model(tmp_path: Path) -> None:
    embedder, api = make_embedder(tmp_path)
    embedder.embed(["x"], model="m1")
    embedder.embed(["x"], model="m2")
    assert len(api.calls) == 2


def test_duplicate_texts_are_embedded_once() -> None:
    embedder, api = make_embedder(batch_size=10)
    assert embedder.embed(["same", "same", "other"], model="m").shape == (3, 4)
    assert api.calls == [["same", "other"]]


def test_failure_gives_actionable_error() -> None:
    embedder, _ = make_embedder(fail=True)
    with pytest.raises(EmbeddingError, match="ollama pull m"):
        embedder.embed(["x"], model="m")


def test_transformer_uses_prefix_and_part() -> None:
    embedder, api = make_embedder(batch_size=10)
    set_embedder(embedder)
    features = EmbeddingFeatures(model="m", prefix="classification: ", part="instruction")
    out = features.fit(["Summarise.\n\nContext:\nLong text"]).transform(["Why?\n\nContext:\nIgnored passage"])
    assert out.shape == (1, 4)
    assert api.calls[-1] == ["classification: Why?"]
    assert len(features.get_feature_names_out()) == 4


def test_transformer_params_are_sklearn_compatible() -> None:
    features = EmbeddingFeatures(model="m", part="full")
    assert features.get_params() == {"model": "m", "prefix": "classification: ", "part": "full"}
