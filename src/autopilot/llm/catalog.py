"""Load the model catalogue (``config/models.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from autopilot.llm.schemas import ModelConfig, PriceSchedule
from autopilot.settings import config_dir, load_yaml

_QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2}


class ModelCatalog(BaseModel):
    """All configured models plus the provider's peak/off-peak schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    price_schedule: PriceSchedule
    models: dict[str, ModelConfig]

    @model_validator(mode="before")
    @classmethod
    def _inject_names(cls, data: Any) -> Any:
        """Let YAML keys double as model names so they're never written twice."""
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            data = dict(data)
            data["models"] = {
                name: {"name": name, **spec} if isinstance(spec, dict) else spec
                for name, spec in data["models"].items()
            }
        return data

    def get(self, name: str) -> ModelConfig:
        """Return the model called ``name`` or raise ``KeyError`` listing valid names."""
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(f"Unknown model {name!r}; known: {sorted(self.models)}") from None

    def by_quality(self) -> list[ModelConfig]:
        """Models ordered cheapest/weakest first."""
        return sorted(self.models.values(), key=lambda m: _QUALITY_ORDER[m.quality_tier])


def load_catalog(path: Path | None = None) -> ModelCatalog:
    """Load and validate ``config/models.yaml`` (or ``path``)."""
    return ModelCatalog.model_validate(load_yaml(path or config_dir() / "models.yaml"))
