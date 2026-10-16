"""Training protocol: frozen split, CV, model selection, end-to-end training on synthetic data."""

from __future__ import annotations

from pathlib import Path

import pytest

from autopilot.classifier.training import (
    DEFAULT_SPECS,
    CVSummary,
    LabeledData,
    TrainedModels,
    load_labeled,
    load_or_create_split,
    missing_test_ids,
    select_model,
    train_models,
)
from autopilot.dataset.io import write_jsonl
from autopilot.dataset.schemas import LabeledPrompt
from tests.factories import synthetic_labels, write_synthetic_labels


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TrainedModels:
    tmp = tmp_path_factory.mktemp("train")
    data = load_labeled(write_synthetic_labels(tmp / "labels.jsonl"))
    return train_models(data, tmp / "split.json", folds=3)


def _data(tmp_path: Path, per_tier: int = 30) -> LabeledData:
    return load_labeled(write_synthetic_labels(tmp_path / "labels.jsonl", per_tier))


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    records = synthetic_labels(2)
    write_jsonl(tmp_path / "a.jsonl", records)
    write_jsonl(tmp_path / "b.jsonl", records[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        load_labeled(tmp_path / "a.jsonl", tmp_path / "b.jsonl")


def test_split_is_stratified_and_frozen(tmp_path: Path) -> None:
    data = _data(tmp_path)
    split_path = tmp_path / "split.json"
    train_idx, test_idx, split, created = load_or_create_split(data, split_path, test_size=0.2, seed=1)
    assert created and split_path.exists()
    assert len(test_idx) == 18 and not set(train_idx) & set(test_idx)
    assert sorted(data.tiers[test_idx].tolist()) == [1] * 6 + [2] * 6 + [3] * 6  # stratified

    # Different seed, more data: the frozen split still wins and new examples go to training.
    extra = LabeledPrompt(candidate_id="new1", prompt="Why?", tier=3, task_type="reasoning_creative", source="routing_failure")
    write_jsonl(tmp_path / "extra.jsonl", [extra])
    bigger = load_labeled(tmp_path / "labels.jsonl", tmp_path / "extra.jsonl")
    train2, test2, split2, created2 = load_or_create_split(bigger, split_path, test_size=0.5, seed=99)
    assert not created2
    assert {bigger.ids[i] for i in test2} == set(split.test_ids)
    assert "new1" in {bigger.ids[i] for i in train2}


def test_routing_failures_never_enter_a_new_test_split(tmp_path: Path) -> None:
    records = synthetic_labels(10) + [
        LabeledPrompt(candidate_id=f"f{i}", prompt=f"Why {i}?", tier=3, task_type="reasoning_creative", source="routing_failure")
        for i in range(10)
    ]
    write_jsonl(tmp_path / "labels.jsonl", records)
    data = load_labeled(tmp_path / "labels.jsonl")
    _, test_idx, _, _ = load_or_create_split(data, tmp_path / "split.json", test_size=0.3, seed=0)
    assert all(data.sources[i] == "human" for i in test_idx)


def test_split_drops_missing_ids_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = _data(tmp_path)
    _, test_idx, split, _ = load_or_create_split(data, tmp_path / "split.json", test_size=0.2, seed=1)
    skipped = split.test_ids[0]
    keep = [i for i, cid in enumerate(data.ids) if cid != skipped]
    partial = data.subset(keep)
    with caplog.at_level("WARNING"):
        _, test_idx2, _, _ = load_or_create_split(partial, tmp_path / "split.json", test_size=0.2, seed=1)
    assert len(test_idx2) == len(test_idx) - 1
    assert skipped in caplog.text
    assert missing_test_ids(partial, split) == [skipped]


def test_split_with_no_matching_ids_fails(tmp_path: Path) -> None:
    load_or_create_split(_data(tmp_path), tmp_path / "split.json", test_size=0.2, seed=1)
    other = LabeledPrompt(candidate_id="zzz", prompt="Why?", tier=3, task_type="reasoning_creative")
    write_jsonl(tmp_path / "other.jsonl", [other])
    with pytest.raises(ValueError, match="None of the frozen"):
        load_or_create_split(load_labeled(tmp_path / "other.jsonl"), tmp_path / "split.json", test_size=0.2, seed=1)


def cv(acc: float, t3: float) -> CVSummary:
    return CVSummary(folds=5, accuracy_mean=acc, accuracy_std=0, macro_f1_mean=acc, tier3_recall_mean=t3,
                     tier3_recall_std=0, under_routing_mean=0, over_routing_mean=0)


def test_select_model_prefers_tier3_recall_above_accuracy_floor() -> None:
    kind, _ = select_model({"logistic_regression": cv(0.80, 0.70), "random_forest": cv(0.78, 0.85)}, min_accuracy=0.7)
    assert kind == "random_forest"


def test_select_model_rejects_degenerate_high_recall() -> None:
    kind, _ = select_model({"logistic_regression": cv(0.80, 0.70), "random_forest": cv(0.35, 1.0)}, min_accuracy=0.7)
    assert kind == "logistic_regression"


def test_select_model_ties_go_to_accuracy_and_fallback() -> None:
    assert select_model({"logistic_regression": cv(0.75, 0.8), "random_forest": cv(0.82, 0.8)}, min_accuracy=0.7)[0] == "random_forest"
    kind, reason = select_model({"logistic_regression": cv(0.5, 0.9), "random_forest": cv(0.6, 0.2)}, min_accuracy=0.7)
    assert kind == "random_forest" and "no model" in reason


def test_end_to_end_training_report(trained: TrainedModels) -> None:
    report = trained.report
    assert report.selected in DEFAULT_SPECS
    assert set(report.cv) == {spec.name for spec in DEFAULT_SPECS}
    assert report.training_size + report.test_size == 90
    assert report.test.accuracy >= 0.8  # the synthetic signal is easy
    assert len(report.test_errors) == round(report.test.n * (1 - report.test.accuracy))
    assert set(report.top_features) == {1, 2, 3}
    assert len(report.threshold_table) >= 2
    assert 0 <= report.task_type_test_accuracy <= 1
    assert len(report.test_other_models) == len(DEFAULT_SPECS) - 1
    assert trained.test_data.ids and len(trained.test_data) == report.test_size


def test_trained_pipelines_predict_probabilities(trained: TrainedModels) -> None:
    proba = trained.tier_pipeline.predict_proba(["Write a poem about the sea with exactly 6 lines."])
    assert proba.shape == (1, 3) and proba.sum() == pytest.approx(1)
    assert set(trained.task_pipeline.classes_) <= {
        "extraction_reformatting", "qa_classification", "summarization_analysis", "reasoning_creative"
    }
