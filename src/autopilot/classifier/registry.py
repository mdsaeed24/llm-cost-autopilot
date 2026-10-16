"""Versioned classifier storage: ``models/classifier_vN.joblib`` + ``classifier_vN.json``.

``models/registry.json`` records which version is *current* (used for routing) and the
history of promotions, so a retrained model only replaces the current one when promoted.

Security note: joblib files are pickles, and loading a pickle can run arbitrary code.
Only load model files this project produced.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from autopilot.classifier.predict import TierClassifier
from autopilot.classifier.training import TrainedModels
from autopilot.dataset.io import atomic_write_text
from autopilot.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
_VERSION_FILE = re.compile(r"^classifier_v(\d+)\.joblib$")


class RegistryError(RuntimeError):
    """No usable model (nothing trained yet, unknown version, ...)."""


class ClassifierMetadata(BaseModel):
    """Saved next to each model: when, on what, with which library versions, how well."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    version: str
    created_at: datetime
    model_kind: str
    training_size: int
    test_size: int
    label_files: dict[str, str] = Field(description="Path → SHA-256, to tell exactly which labels were used.")
    test_split_file: str
    sklearn_version: str
    python_version: str
    metrics: dict[str, Any]


class PromotionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    promoted_at: datetime
    reason: str


class Registry(BaseModel):
    """Contents of ``models/registry.json``."""

    model_config = ConfigDict(extra="forbid")

    current: str | None = None
    history: list[PromotionRecord] = Field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def list_versions(models_dir: Path) -> list[int]:
    """Version numbers present on disk, ascending."""
    if not models_dir.exists():
        return []
    return sorted(int(m.group(1)) for p in models_dir.iterdir() if (m := _VERSION_FILE.match(p.name)))


def read_registry(models_dir: Path) -> Registry:
    path = models_dir / "registry.json"
    return Registry.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else Registry()


def save_models(
    trained: TrainedModels,
    *,
    models_dir: Path,
    label_files: list[Path],
    test_split_file: Path,
) -> ClassifierMetadata:
    """Save as the next version (v1, v2, ...). Does NOT promote it."""
    models_dir.mkdir(parents=True, exist_ok=True)
    version = f"v{(list_versions(models_dir) or [0])[-1] + 1}"
    report = trained.report
    metadata = ClassifierMetadata(
        version=version,
        created_at=datetime.now(UTC),
        model_kind=report.selected.name,
        training_size=report.training_size,
        test_size=report.test_size,
        label_files={_relative(p): _sha256(p) for p in label_files if p.exists()},
        test_split_file=_relative(test_split_file),
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        metrics=report.model_dump(mode="json"),
    )
    joblib.dump(
        {"version": version, "tier_pipeline": trained.tier_pipeline, "task_pipeline": trained.task_pipeline},
        models_dir / f"classifier_{version}.joblib",
    )
    atomic_write_text(models_dir / f"classifier_{version}.json", metadata.model_dump_json(indent=2))
    return metadata


def promote(models_dir: Path, version: str, reason: str) -> Registry:
    """Make ``version`` the current model and log why."""
    if not (models_dir / f"classifier_{version}.joblib").exists():
        raise RegistryError(f"Cannot promote {version}: model file not found in {models_dir}")
    registry = read_registry(models_dir)
    registry.current = version
    registry.history.append(PromotionRecord(version=version, promoted_at=datetime.now(UTC), reason=reason))
    atomic_write_text(models_dir / "registry.json", registry.model_dump_json(indent=2))
    return registry


def load_metadata(models_dir: Path, version: str) -> ClassifierMetadata:
    return ClassifierMetadata.model_validate_json((models_dir / f"classifier_{version}.json").read_text(encoding="utf-8"))


def load_classifier(models_dir: Path = DEFAULT_MODELS_DIR, version: str = "current") -> TierClassifier:
    """Load a classifier by version ("v1", ...) or the promoted ``"current"`` one."""
    if version == "current":
        current = read_registry(models_dir).current
        if current is None:
            raise RegistryError(f"No promoted classifier in {models_dir} — run scripts/train_classifier.py")
        version = current
    path = models_dir / f"classifier_{version}.joblib"
    if not path.exists():
        raise RegistryError(f"Classifier {version} not found at {path}")
    metadata = load_metadata(models_dir, version)
    if metadata.sklearn_version != sklearn.__version__:
        logger.warning(
            "Classifier %s was saved with scikit-learn %s but %s is installed; predictions may differ",
            version, metadata.sklearn_version, sklearn.__version__,
        )
    bundle = joblib.load(path)
    return TierClassifier(bundle["tier_pipeline"], bundle["task_pipeline"], version=bundle["version"])
