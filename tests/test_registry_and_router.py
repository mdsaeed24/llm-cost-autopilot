"""Versioned model registry and the router (with stub and real classifiers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from autopilot.classifier.predict import Prediction
from autopilot.classifier.registry import (
    RegistryError,
    list_versions,
    load_classifier,
    promote,
    read_registry,
    save_models,
)
from autopilot.classifier.training import TrainedModels, load_labeled, train_models
from autopilot.llm.catalog import ModelCatalog, load_catalog
from autopilot.routing.config import ConfidenceRule, RoutingConfig, load_routing_config
from autopilot.routing.router import Router
from autopilot.settings import ConfigError
from tests.factories import write_synthetic_labels


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[TrainedModels, Path, Path]:
    tmp = tmp_path_factory.mktemp("reg")
    labels = write_synthetic_labels(tmp / "labels.jsonl")
    split = tmp / "split.json"
    return train_models(load_labeled(labels), split, folds=3), labels, split


@pytest.fixture
def catalog() -> ModelCatalog:
    return load_catalog()


# ---------- registry ----------

def test_versions_increment_and_promotion(trained: tuple[TrainedModels, Path, Path], tmp_path: Path) -> None:
    models, labels, split = trained
    v1 = save_models(models, models_dir=tmp_path, label_files=[labels], test_split_file=split)
    v2 = save_models(models, models_dir=tmp_path, label_files=[labels], test_split_file=split)
    assert (v1.version, v2.version) == ("v1", "v2")
    assert list_versions(tmp_path) == [1, 2]
    assert len(v1.label_files) == 1 and len(next(iter(v1.label_files.values()))) == 64  # sha256
    assert v1.metrics["selected"]["kind"] in v1.model_kind

    with pytest.raises(RegistryError, match="No promoted"):
        load_classifier(tmp_path)
    promote(tmp_path, "v1", "first")
    promote(tmp_path, "v2", "better")
    registry = read_registry(tmp_path)
    assert registry.current == "v2"
    assert [h.reason for h in registry.history] == ["first", "better"]
    assert load_classifier(tmp_path).version == "v2"
    assert load_classifier(tmp_path, "v1").version == "v1"


def test_registry_errors(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        promote(tmp_path, "v9", "nope")
    with pytest.raises(RegistryError, match="not found"):
        load_classifier(tmp_path, "v9")


def test_loaded_classifier_predicts(trained: tuple[TrainedModels, Path, Path], tmp_path: Path) -> None:
    models, labels, split = trained
    save_models(models, models_dir=tmp_path, label_files=[labels], test_split_file=split)
    prediction = load_classifier(tmp_path, "v1").predict("Why did the Roman Empire matter? Explain step by step.")
    assert prediction.tier == 3
    assert sum(prediction.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert prediction.confidence == pytest.approx(max(prediction.probabilities.values()), abs=1e-3)


def test_sklearn_version_mismatch_warns(
    trained: tuple[TrainedModels, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    models, labels, split = trained
    save_models(models, models_dir=tmp_path, label_files=[labels], test_split_file=split)
    meta_path = tmp_path / "classifier_v1.json"
    meta = json.loads(meta_path.read_text())
    meta["sklearn_version"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    with caplog.at_level(logging.WARNING):
        load_classifier(tmp_path, "v1")
    assert "0.0.1" in caplog.text


# ---------- routing config ----------

def test_repo_routing_config_is_valid(catalog: ModelCatalog) -> None:
    config = load_routing_config(catalog)
    assert config.tiers == {1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"}
    assert 0 < config.confidence.threshold < 1


def test_routing_config_rejects_missing_tier_and_unknown_model(catalog: ModelCatalog) -> None:
    with pytest.raises(ValidationError, match="missing"):
        RoutingConfig(tiers={1: "local-llama", 2: "deepseek-flash"}, confidence=ConfidenceRule(threshold=0.5))
    bad = RoutingConfig(tiers={1: "local-llama", 2: "deepseek-flash", 3: "gpt-9"}, confidence=ConfidenceRule(threshold=0.5))
    with pytest.raises(ConfigError, match="gpt-9"):
        bad.check_models_exist(catalog)


# ---------- router ----------

class StubClassifier:
    version = "stub"

    def __init__(self, tier: int, confidence: float) -> None:
        self.tier, self.confidence = tier, confidence

    def predict(self, prompt: str) -> Prediction:
        rest = (1 - self.confidence) / 2
        probs = {t: (self.confidence if t == self.tier else rest) for t in (1, 2, 3)}
        return Prediction(tier=self.tier, confidence=self.confidence, probabilities=probs,  # type: ignore[arg-type]
                          task_type="qa_classification", task_confidence=0.9)


def router(catalog: ModelCatalog, tier: int, confidence: float, *, threshold: float = 0.6, enabled: bool = True) -> Router:
    config = RoutingConfig(
        tiers={1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"},
        confidence=ConfidenceRule(enabled=enabled, threshold=threshold),
    )
    return Router(StubClassifier(tier, confidence), config, catalog)


def test_confident_prediction_is_routed_as_is(catalog: ModelCatalog) -> None:
    decision = router(catalog, 1, 0.9).decide("hi")
    assert (decision.predicted_tier, decision.routed_tier, decision.model_name) == (1, 1, "local-llama")
    assert not decision.routed_up
    assert decision.classifier_version == "stub"
    assert "p=0.90" in decision.reason


@pytest.mark.parametrize(("tier", "expected_tier", "expected_model"), [(1, 2, "deepseek-flash"), (2, 3, "deepseek-v4-pro")])
def test_low_confidence_routes_one_tier_up(catalog: ModelCatalog, tier: int, expected_tier: int, expected_model: str) -> None:
    decision = router(catalog, tier, 0.45).decide("hi")
    assert decision.routed_tier == expected_tier and decision.model_name == expected_model
    assert decision.routed_up
    assert "below confidence threshold 0.60" in decision.reason


def test_low_confidence_top_tier_stays(catalog: ModelCatalog) -> None:
    decision = router(catalog, 3, 0.4).decide("hi")
    assert decision.routed_tier == 3 and not decision.routed_up
    assert "already top tier" in decision.reason


def test_rule_can_be_disabled(catalog: ModelCatalog) -> None:
    assert router(catalog, 1, 0.4, enabled=False).decide("hi").routed_tier == 1


def test_model_for_returns_catalog_entry(catalog: ModelCatalog) -> None:
    r = router(catalog, 2, 0.9)
    assert r.model_for(r.decide("hi")).model_id == "deepseek-flash"


class BrokenClassifier:
    version = "v-broken"

    def predict(self, prompt: str) -> Prediction:
        raise RuntimeError("embedding model unreachable")


def test_classifier_failure_routes_to_fallback_tier(catalog: ModelCatalog) -> None:
    config = RoutingConfig(
        tiers={1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"},
        confidence=ConfidenceRule(threshold=0.5),
        fallback_tier=3,
    )
    decision = Router(BrokenClassifier(), config, catalog).decide("anything")
    assert decision.routed_tier == 3 and decision.model_name == "deepseek-v4-pro"
    assert decision.predicted_tier is None and decision.confidence is None
    assert decision.classifier_error == "RuntimeError: embedding model unreachable"
    assert "fallback tier 3" in decision.reason


def test_embedding_model_classifier_round_trip(trained: tuple[TrainedModels, Path, Path], tmp_path: Path) -> None:
    """A saved model whose features include embeddings loads and predicts (fake embedder in tests)."""
    models, labels, split = trained
    save_models(models, models_dir=tmp_path, label_files=[labels], test_split_file=split)
    assert load_classifier(tmp_path, "v1").predict("Write a poem about volcanoes").tier in (1, 2, 3)
