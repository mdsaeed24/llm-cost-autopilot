"""``config/routing.yaml``: which model serves each tier, and the low-confidence rule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autopilot.dataset.schemas import TIERS, Tier
from autopilot.llm.catalog import ModelCatalog
from autopilot.settings import ConfigError, config_dir, load_yaml

logger = logging.getLogger(__name__)

_QUALITY_RANK = {"low": 0, "medium": 1, "high": 2}


class ConfidenceRule(BaseModel):
    """If the classifier's top probability is below ``threshold``, route one tier up."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    threshold: float = Field(ge=0, le=1)


class RoutingConfig(BaseModel):
    """Tier → model mapping plus routing rules. Editable without code changes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tiers: dict[Tier, str]
    confidence: ConfidenceRule
    fallback_tier: Tier = Field(default=3, description="Tier used when the classifier fails (e.g. Ollama down).")
    classifier_version: str = Field(default="current", description='"current" or a pinned version like "v1".')

    @field_validator("tiers", mode="before")
    @classmethod
    def _numeric_keys(cls, tiers: object) -> object:
        """JSON object keys are always strings ("1"); tiers are ints."""
        if isinstance(tiers, dict):
            return {int(k) if isinstance(k, str) and k.strip().isdigit() else k: v for k, v in tiers.items()}
        return tiers

    @field_validator("tiers")
    @classmethod
    def _all_tiers_present(cls, tiers: dict[Tier, str]) -> dict[Tier, str]:
        missing = set(TIERS) - set(tiers)
        if missing:
            raise ValueError(f"tiers must map every tier 1-3 to a model; missing {sorted(missing)}")
        return tiers

    def check_models_exist(self, catalog: ModelCatalog) -> None:
        """Raise ConfigError if a tier points at a model not in ``config/models.yaml``."""
        unknown = {tier: name for tier, name in self.tiers.items() if name not in catalog.models}
        if unknown:
            raise ConfigError(f"routing tiers reference unknown models {unknown}; known: {sorted(catalog.models)}")

    def check_quality_order(self, catalog: ModelCatalog) -> None:
        """Higher tiers must not use a weaker model — Tier 3 is also the verification reference."""
        ranks = [_QUALITY_RANK[catalog.get(self.tiers[t]).quality_tier] for t in sorted(self.tiers)]
        if ranks != sorted(ranks):
            mapping = {t: f"{self.tiers[t]} ({catalog.get(self.tiers[t]).quality_tier})" for t in sorted(self.tiers)}
            raise ConfigError(f"tier models must not get weaker as the tier rises: {mapping}")

    def validate_against(self, catalog: ModelCatalog) -> None:
        self.check_models_exist(catalog)
        self.check_quality_order(catalog)


def load_routing_config(catalog: ModelCatalog, path: Path | None = None) -> RoutingConfig:
    """Load ``config/routing.yaml`` and validate it against the model catalogue."""
    config = RoutingConfig.model_validate(load_yaml(path or config_dir() / "routing.yaml"))
    config.validate_against(catalog)
    return config


class ConfigStore(Protocol):
    def latest_config(self) -> str | None: ...


def effective_routing_config(base: RoutingConfig, store: ConfigStore, catalog: ModelCatalog) -> RoutingConfig:
    """The latest runtime change (PUT /v1/routing-config) if there is one, else ``routing.yaml``.

    A stored config that no longer validates (e.g. its model was removed from models.yaml)
    is ignored with a warning rather than taking the service down.
    """
    stored = store.latest_config()
    if stored is None:
        return base
    try:
        config = RoutingConfig.model_validate_json(stored)
        config.validate_against(catalog)
    except (ValueError, ConfigError) as error:
        logger.warning("Ignoring stored routing config (%s); using routing.yaml", error)
        return base
    return config
