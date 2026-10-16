"""Train → save → compare with the current model → promote if not worse. Shared by both scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from autopilot.classifier.metrics import EvalMetrics
from autopilot.classifier.promotion import PromotionDecision, decide_promotion, score_on
from autopilot.classifier.registry import ClassifierMetadata, load_classifier, promote, read_registry, save_models
from autopilot.classifier.training import DEFAULT_SPECS, ModelSpec, TrainedModels, load_labeled, train_models

PromoteMode = Literal["auto", "force", "never"]


@dataclass
class TrainingOutcome:
    trained: TrainedModels
    metadata: ClassifierMetadata | None  # None when not saved (dry run)
    current_version: str | None
    current_metrics: EvalMetrics | None
    decision: PromotionDecision
    promoted: bool
    promotion_reason: str


def train_compare_promote(
    label_files: Sequence[Path],
    *,
    split: Path,
    models_dir: Path,
    specs: Sequence[ModelSpec] = DEFAULT_SPECS,
    folds: int = 5,
    seed: int = 42,
    test_size: float = 0.2,
    min_accuracy: float = 0.7,
    promote_mode: PromoteMode = "auto",
    save: bool = True,
    tolerance: float = 0.0,
) -> TrainingOutcome:
    """Train a candidate and apply the promotion rule against the current model.

    Both models are scored on the same frozen test prompts, with the labels from ``label_files``.
    """
    data = load_labeled(*label_files)
    trained = train_models(
        data, split, test_size=test_size, folds=folds, seed=seed, min_accuracy=min_accuracy, specs=specs
    )
    current_version = read_registry(models_dir).current
    current_metrics = None
    if current_version is not None:
        current_metrics = score_on(
            load_classifier(models_dir, current_version), trained.test_data.texts, trained.test_data.tiers.tolist()
        )
    decision = decide_promotion(trained.report.test, current_metrics, tolerance=tolerance)

    if not save:
        return TrainingOutcome(trained, None, current_version, current_metrics, decision, False, "dry run: nothing saved")

    metadata = save_models(trained, models_dir=models_dir, label_files=list(label_files), test_split_file=split)
    if promote_mode == "never":
        promoted, reason = False, f"--no-promote (rule said: {decision.reason})"
    elif promote_mode == "force" and not decision.promote:
        promoted, reason = True, f"forced via --promote (rule said: {decision.reason})"
    else:
        promoted, reason = decision.promote, decision.reason
    if promoted:
        promote(models_dir, metadata.version, reason)
    return TrainingOutcome(trained, metadata, current_version, current_metrics, decision, promoted, reason)


class RetrainLogEntry(BaseModel):
    """One line of models/retrain_log.jsonl."""

    model_config = ConfigDict(extra="forbid")

    at: datetime
    dry_run: bool
    candidate_version: str | None
    current_version: str | None
    human_examples: int
    failure_examples: int
    new_failures: int
    candidate: dict[str, float]
    current: dict[str, float] | None
    decision: str
    promoted: bool
