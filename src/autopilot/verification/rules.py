"""Per-task-type quality rules (``config/quality.yaml``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autopilot.dataset.schemas import TASK_TYPES, TaskType
from autopilot.llm.catalog import ModelCatalog
from autopilot.settings import ConfigError, config_dir, load_yaml


class QualityRule(BaseModel):
    """What the judge checks, and the agreement score (1-5) needed to pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: str = Field(min_length=10)
    pass_threshold: int = Field(ge=1, le=5)


class QualityConfig(BaseModel):
    """All rules plus which model judges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_model: str
    min_task_confidence: float = Field(ge=0, le=1)
    rules: dict[TaskType, QualityRule]
    default: QualityRule

    @field_validator("rules")
    @classmethod
    def _every_task_type(cls, rules: dict[TaskType, QualityRule]) -> dict[TaskType, QualityRule]:
        missing = set(TASK_TYPES) - set(rules)
        if missing:
            raise ValueError(f"quality rules missing for task types: {sorted(missing)}")
        return rules

    def select(self, task_type: str | None, task_confidence: float | None) -> tuple[str, QualityRule]:
        """Rule for this request: the task type's rule if the classifier is confident, else ``default``."""
        if (
            task_type in self.rules
            and task_confidence is not None
            and task_confidence >= self.min_task_confidence
        ):
            return task_type, self.rules[task_type]  # type: ignore[index]
        return "default", self.default


def load_quality_config(catalog: ModelCatalog, path: Path | None = None) -> QualityConfig:
    """Load ``config/quality.yaml`` and check the judge model exists."""
    config = QualityConfig.model_validate(load_yaml(path or config_dir() / "quality.yaml"))
    if config.judge_model not in catalog.models:
        raise ConfigError(f"judge_model {config.judge_model!r} is not in config/models.yaml")
    return config
