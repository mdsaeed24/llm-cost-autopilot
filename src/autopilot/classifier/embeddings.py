"""Sentence embeddings from a local Ollama model, cached in SQLite.

An *embedding* turns a sentence into a list of numbers (768 for nomic-embed-text) that
captures its meaning: "What causes rain?" and "Why does it rain?" land close together even
though they share few words. That is what TF-IDF cannot do, and why it helps with ~200 labels.

The trained classifier stores only the embedding *model name* — never a URL — so the same
model file works on the host (localhost) and in Docker (host.docker.internal).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import openai
from numpy.typing import NDArray
from openai import OpenAI
from sklearn.base import BaseEstimator, TransformerMixin

from autopilot.settings import PROJECT_ROOT, EmbeddingSettings, load_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
# nomic-embed-text was trained with task prefixes; "classification: " is the one for our use.
DEFAULT_PREFIX = "classification: "


class EmbeddingError(RuntimeError):
    """The embedding model could not be reached or returned something unusable."""


class Embedder(Protocol):
    """Turns texts into a (len(texts), dim) array."""

    def embed(self, texts: Sequence[str], *, model: str) -> NDArray[np.float64]: ...


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class EmbeddingCache:
    """SQLite key → vector store. Safe to share across threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self._lock = threading.Lock()

    def get_many(self, keys: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        if not keys:
            return {}
        found: dict[str, NDArray[np.float64]] = {}
        with self._lock:
            for start in range(0, len(keys), 500):  # stay under SQLite's variable limit
                chunk = list(keys[start : start + 500])
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", chunk)
                found.update({k: np.frombuffer(v, dtype=np.float64) for k, v in rows})
        return found

    def put_many(self, items: dict[str, NDArray[np.float64]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(k, np.asarray(v, dtype=np.float64).tobytes()) for k, v in items.items()],
            )

    def close(self) -> None:
        self._conn.close()


class OllamaEmbedder:
    """Calls Ollama's OpenAI-compatible ``/v1/embeddings`` endpoint, in batches, through the cache."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        cache: EmbeddingCache | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self._client = client or OpenAI(
            api_key="ollama", base_url=settings.base_url, timeout=settings.timeout_s, max_retries=2
        )

    def embed(self, texts: Sequence[str], *, model: str) -> NDArray[np.float64]:
        keys = [cache_key(model, t) for t in texts]
        vectors: dict[str, NDArray[np.float64]] = self.cache.get_many(keys) if self.cache else {}
        missing = [(k, t) for k, t in dict(zip(keys, texts, strict=True)).items() if k not in vectors]

        for start in range(0, len(missing), self.settings.batch_size):
            batch = missing[start : start + self.settings.batch_size]
            try:
                response = self._client.embeddings.create(model=model, input=[t for _, t in batch])
            except openai.OpenAIError as error:
                raise EmbeddingError(
                    f"Embedding model {model!r} at {self.settings.base_url} failed ({type(error).__name__}: {error}). "
                    f"Is Ollama running, and has `ollama pull {model}` been done?"
                ) from error
            ordered = sorted(response.data, key=lambda d: d.index)
            if len(ordered) != len(batch):
                raise EmbeddingError(f"Asked for {len(batch)} embeddings, got {len(ordered)}")
            fresh = {k: np.asarray(d.embedding, dtype=np.float64) for (k, _), d in zip(batch, ordered, strict=True)}
            vectors.update(fresh)
            if self.cache:
                self.cache.put_many(fresh)

        matrix = np.vstack([vectors[k] for k in keys])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)  # unit length, so only direction (meaning) counts


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Process-wide embedder built from ``config/settings.yaml`` (tests replace it via set_embedder)."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            settings = load_settings().embeddings
            cache = EmbeddingCache(PROJECT_ROOT / settings.cache_path) if settings.cache_path else None
            _embedder = OllamaEmbedder(settings, cache=cache)
        return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Override (or reset with None) the process-wide embedder."""
    global _embedder
    with _embedder_lock:
        _embedder = embedder


class EmbeddingFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """scikit-learn transformer: prompt → embedding vector.

    Args:
        model: embedding model name (part of the trained artefact; changing it needs retraining).
        prefix: task prefix prepended to every text.
        part: embed the whole prompt, or only the instruction (context passage removed).
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        prefix: str = DEFAULT_PREFIX,
        part: Literal["full", "instruction"] = "full",
    ) -> None:
        self.model = model
        self.prefix = prefix
        self.part = part

    def _texts(self, texts: Sequence[str]) -> list[str]:
        from autopilot.classifier.features import split_prompt  # avoid an import cycle

        if self.part == "instruction":
            texts = [split_prompt(t)[0] for t in texts]
        return [self.prefix + t for t in texts]

    def fit(self, texts: Sequence[str], y: object = None) -> EmbeddingFeatures:
        self.dim_ = get_embedder().embed(self._texts(list(texts)[:1]), model=self.model).shape[1]
        return self

    def transform(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return get_embedder().embed(self._texts(list(texts)), model=self.model)

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.str_]:
        return np.array([f"emb_{i}" for i in range(self.dim_)])
