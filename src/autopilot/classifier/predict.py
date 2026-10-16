"""Run a trained classifier on one prompt."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from autopilot.dataset.schemas import TaskType, Tier


class Prediction(BaseModel):
    """The classifier's view of one prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Tier
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted tier.")
    probabilities: dict[int, float]
    task_type: TaskType
    task_confidence: float = Field(ge=0, le=1)


class TierPredictor(Protocol):
    """Anything that can classify a prompt (the real classifier, or a stub in tests)."""

    version: str

    def predict(self, prompt: str) -> Prediction: ...


class TierClassifier:
    """Wraps the fitted tier and task-type pipelines loaded from the registry."""

    def __init__(self, tier_pipeline: Pipeline, task_pipeline: Pipeline, version: str) -> None:
        self.tier_pipeline = tier_pipeline
        self.task_pipeline = task_pipeline
        self.version = version

    def predict(self, prompt: str) -> Prediction:
        texts = np.array([prompt], dtype=object)
        tier_proba = self.tier_pipeline.predict_proba(texts)[0]
        tier_classes = self.tier_pipeline.classes_
        best = int(np.argmax(tier_proba))
        task_proba = self.task_pipeline.predict_proba(texts)[0]
        task_best = int(np.argmax(task_proba))
        return Prediction(
            tier=int(tier_classes[best]),  # type: ignore[arg-type]
            confidence=float(tier_proba[best]),
            probabilities={int(c): round(float(p), 4) for c, p in zip(tier_classes, tier_proba, strict=True)},
            task_type=str(self.task_pipeline.classes_[task_best]),  # type: ignore[arg-type]
            task_confidence=float(task_proba[task_best]),
        )
