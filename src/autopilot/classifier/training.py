"""Train and compare tier classifiers on the hand-labeled prompts.

Protocol (the order matters, and is how you avoid fooling yourself):
1. Freeze a stratified held-out **test set** (20%) once, and save its ids. Every later model —
   including retrained ones in Phase 3 — is scored on exactly these prompts.
2. On the remaining **training set**, compare models with stratified k-fold cross-validation.
3. **Select** the model using cross-validation results only — never the test set.
4. Fit the selected model on the full training set and score it on the test set, once.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from autopilot.classifier.features import FeatureSet, build_feature_union, feature_names
from autopilot.classifier.metrics import EvalMetrics, ThresholdRow, evaluate, threshold_analysis
from autopilot.dataset.io import atomic_write_text, read_jsonl
from autopilot.dataset.schemas import LabeledPrompt

logger = logging.getLogger(__name__)

ModelKind = Literal["logistic_regression", "random_forest"]
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)


@dataclass(frozen=True)
class LabeledData:
    """Parallel arrays of labeled examples."""

    ids: list[str]
    texts: list[str]
    tiers: NDArray[np.int_]
    task_types: list[str]
    sources: list[str]

    def subset(self, index: Sequence[int]) -> LabeledData:
        idx = list(index)
        return LabeledData(
            ids=[self.ids[i] for i in idx],
            texts=[self.texts[i] for i in idx],
            tiers=self.tiers[idx],
            task_types=[self.task_types[i] for i in idx],
            sources=[self.sources[i] for i in idx],
        )

    def __len__(self) -> int:
        return len(self.ids)


def load_labeled(*paths: Path) -> LabeledData:
    """Load and concatenate labeled JSONL files (human labels, later routing failures)."""
    records: list[LabeledPrompt] = []
    for path in paths:
        records.extend(read_jsonl(path, LabeledPrompt))
    ids = [r.candidate_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate ids across label files")
    return LabeledData(
        ids=ids,
        texts=[r.prompt for r in records],
        tiers=np.array([r.tier for r in records], dtype=int),
        task_types=[r.task_type for r in records],
        sources=[r.source for r in records],
    )


# ---------- the frozen test split ----------

class TestSplit(BaseModel):
    """Ids of the held-out test prompts. Created once; never changes afterwards."""

    __test__: ClassVar[bool] = False  # not a pytest test class, despite the name

    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    seed: int
    test_size: float
    test_ids: list[str]


def load_or_create_split(
    data: LabeledData, path: Path, *, test_size: float, seed: int
) -> tuple[list[int], list[int], TestSplit, bool]:
    """Return (train_index, test_index, split, created).

    If ``path`` exists, its test ids are reused, and any example not in it (new labels,
    routing failures) goes to training. Test ids absent from ``data`` (e.g. skipped when
    re-labeling) are dropped with a warning; see :func:`missing_test_ids`.
    Otherwise a stratified split is created and saved.
    """
    if path.exists():
        split = TestSplit.model_validate_json(path.read_text(encoding="utf-8"))
        test_ids = set(split.test_ids)
        missing = test_ids - set(data.ids)
        if missing == test_ids:
            raise ValueError("None of the frozen test ids are in the labels — wrong labels file?")
        if missing:
            logger.warning("%d frozen test ids are not in the labels and are left out: %s", len(missing), sorted(missing))
        test_index = [i for i, cid in enumerate(data.ids) if cid in test_ids]
        train_index = [i for i, cid in enumerate(data.ids) if cid not in test_ids]
        return train_index, test_index, split, False

    human = [i for i, s in enumerate(data.sources) if s == "human"]
    train_h, test_h = train_test_split(
        human, test_size=test_size, stratify=data.tiers[human], random_state=seed
    )
    split = TestSplit(
        created_at=datetime.now(UTC), seed=seed, test_size=test_size, test_ids=sorted(data.ids[i] for i in test_h)
    )
    atomic_write_text(path, split.model_dump_json(indent=2))
    test_set = set(test_h)
    train_index = [i for i in range(len(data)) if i not in test_set]
    return train_index, sorted(test_h), split, True


def missing_test_ids(data: LabeledData, split: TestSplit) -> list[str]:
    """Frozen test ids with no label in ``data`` (reported so the test set never shrinks silently)."""
    return sorted(set(split.test_ids) - set(data.ids))


# ---------- models ----------

class ModelSpec(BaseModel):
    """One candidate: a classifier type plus the feature set it reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ModelKind
    features: FeatureSet

    @property
    def name(self) -> str:
        return f"{self.kind}+{self.features}"


# Candidates compared by cross-validation. Random forest is skipped on the largest feature
# set: hundreds of dense embedding columns plus TF-IDF suit a linear model better.
DEFAULT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(kind="logistic_regression", features="tfidf_hc"),
    ModelSpec(kind="random_forest", features="tfidf_hc"),
    ModelSpec(kind="logistic_regression", features="embed_hc"),
    ModelSpec(kind="random_forest", features="embed_hc"),
    ModelSpec(kind="logistic_regression", features="embed_hc_tfidf"),
)
INTERPRETABLE_SPEC = ModelSpec(kind="logistic_regression", features="tfidf_hc")


def make_tier_pipeline(spec: ModelSpec, seed: int) -> Pipeline:
    """Features → classifier. ``class_weight="balanced"`` makes mistakes on smaller tiers count more."""
    if spec.kind == "logistic_regression":
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    elif spec.kind == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1
        )
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown model kind {spec.kind!r}")
    return Pipeline([("features", build_feature_union(spec.features)), ("classifier", classifier)])


def make_task_type_pipeline(features: FeatureSet) -> Pipeline:
    """Same features, predicting the task-type group (used to pick a quality rule in Phase 3)."""
    return Pipeline(
        [
            ("features", build_feature_union(features)),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        ]
    )


class CVSummary(BaseModel):
    """Mean and standard deviation of each metric across cross-validation folds."""

    model_config = ConfigDict(extra="forbid")

    folds: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    tier3_recall_mean: float
    tier3_recall_std: float
    under_routing_mean: float
    over_routing_mean: float


def cross_validate(spec: ModelSpec, data: LabeledData, *, folds: int, seed: int) -> CVSummary:
    """Stratified k-fold CV: train on k-1 folds, score on the held-out fold, repeat k times."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    texts = np.array(data.texts, dtype=object)
    fold_metrics: list[EvalMetrics] = []
    for train_idx, val_idx in splitter.split(texts, data.tiers):
        model = make_tier_pipeline(spec, seed).fit(texts[train_idx], data.tiers[train_idx])
        fold_metrics.append(evaluate(data.tiers[val_idx], model.predict(texts[val_idx])))

    def stat(values: list[float]) -> tuple[float, float]:
        return float(np.mean(values)), float(np.std(values))

    acc_mean, acc_std = stat([m.accuracy for m in fold_metrics])
    t3_mean, t3_std = stat([m.tier3_recall for m in fold_metrics])
    return CVSummary(
        folds=folds,
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        macro_f1_mean=stat([m.macro_f1 for m in fold_metrics])[0],
        tier3_recall_mean=t3_mean,
        tier3_recall_std=t3_std,
        under_routing_mean=stat([m.under_routing_rate for m in fold_metrics])[0],
        over_routing_mean=stat([m.over_routing_rate for m in fold_metrics])[0],
    )


def select_model(cv: dict[str, CVSummary], *, min_accuracy: float) -> tuple[str, str]:
    """Pick the candidate with the best CV Tier-3 recall among those with acceptable CV accuracy.

    The accuracy floor stops a degenerate "always say Tier 3" model (recall 1.0) from winning.
    Ties on Tier-3 recall go to higher accuracy.
    """
    eligible = {k: s for k, s in cv.items() if s.accuracy_mean >= min_accuracy}
    if not eligible:
        best = max(cv, key=lambda k: cv[k].accuracy_mean)
        return best, f"no model reached CV accuracy {min_accuracy:.0%}; chose highest accuracy"
    best = max(eligible, key=lambda k: (round(eligible[k].tier3_recall_mean, 3), eligible[k].accuracy_mean))
    return best, f"highest CV Tier-3 recall among models with CV accuracy ≥ {min_accuracy:.0%}"


def top_tfidf_features(model: Pipeline, n: int = 8) -> dict[int, list[tuple[str, float]]]:
    """For logistic regression: features pushing hardest toward each tier (largest coefficients)."""
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        return {}
    names = feature_names(model.named_steps["features"])
    top: dict[int, list[tuple[str, float]]] = {}
    for row, tier in enumerate(classifier.classes_):
        order = np.argsort(classifier.coef_[row])[::-1][:n]
        top[int(tier)] = [(names[i], round(float(classifier.coef_[row][i]), 3)) for i in order]
    return top


# ---------- orchestration ----------

class TestError(BaseModel):
    """One misclassified held-out prompt, for error analysis."""

    __test__: ClassVar[bool] = False  # not a pytest test class, despite the name

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    true_tier: int
    predicted_tier: int
    confidence: float
    prompt_preview: str


class TrainingReport(BaseModel):
    """Everything produced by one training run (also stored as model metadata)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    selected: ModelSpec
    selection_reason: str
    cv: dict[str, CVSummary]
    test: EvalMetrics
    test_errors: list[TestError]
    test_other_models: dict[str, EvalMetrics]
    threshold_table: list[ThresholdRow]
    task_type_test_accuracy: float
    top_features: dict[int, list[tuple[str, float]]]
    training_size: int
    test_size: int
    tier_counts_train: dict[int, int]
    split_created: bool
    test_ids_missing: list[str]


@dataclass
class TrainedModels:
    """Fitted pipelines plus the report describing them."""

    tier_pipeline: Pipeline
    task_pipeline: Pipeline
    report: TrainingReport
    test_data: LabeledData


def train_models(
    data: LabeledData,
    split_path: Path,
    *,
    test_size: float = 0.2,
    folds: int = 5,
    seed: int = 42,
    min_accuracy: float = 0.7,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    specs: Sequence[ModelSpec] = DEFAULT_SPECS,
) -> TrainedModels:
    """Run the full protocol described in the module docstring."""
    train_idx, test_idx, split, created = load_or_create_split(data, split_path, test_size=test_size, seed=seed)
    train, test = data.subset(train_idx), data.subset(test_idx)
    train_x = np.array(train.texts, dtype=object)
    test_x = np.array(test.texts, dtype=object)

    by_name = {spec.name: spec for spec in specs}
    cv = {spec.name: cross_validate(spec, train, folds=folds, seed=seed) for spec in specs}
    selected_name, reason = select_model(cv, min_accuracy=min_accuracy)
    selected = by_name[selected_name]

    # Out-of-fold probabilities on the training set: every prompt is scored by a model that
    # never saw it, so the threshold table is not flattered by memorisation.
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        make_tier_pipeline(selected, seed), train_x, train.tiers, cv=splitter, method="predict_proba"
    )
    thresholds_table = threshold_analysis(train.tiers, oof_proba, sorted(set(train.tiers.tolist())), thresholds)

    tier_pipeline = make_tier_pipeline(selected, seed).fit(train_x, train.tiers)
    test_proba = tier_pipeline.predict_proba(test_x)
    test_pred = tier_pipeline.classes_[test_proba.argmax(axis=1)]
    test_metrics = evaluate(test.tiers, test_pred)
    test_errors = [
        TestError(
            candidate_id=test.ids[i],
            true_tier=int(test.tiers[i]),
            predicted_tier=int(test_pred[i]),
            confidence=round(float(test_proba[i].max()), 3),
            prompt_preview=" ".join(test.texts[i].split())[:120],
        )
        for i in range(len(test))
        if test_pred[i] != test.tiers[i]
    ]

    # For the report only: how the non-selected candidates would have done on the test set.
    other: dict[str, EvalMetrics] = {}
    for spec in specs:
        if spec != selected:
            other_model = make_tier_pipeline(spec, seed).fit(train_x, train.tiers)
            other[spec.name] = evaluate(test.tiers, other_model.predict(test_x))

    task_pipeline = clone(make_task_type_pipeline(selected.features)).fit(train_x, np.array(train.task_types))
    task_accuracy = float(np.mean(task_pipeline.predict(test_x) == np.array(test.task_types)))

    # Word-level logistic-regression weights are readable ("this word pushes toward Tier 3");
    # embedding dimensions are not. So we always explain with the interpretable model.
    explain_model = tier_pipeline if selected == INTERPRETABLE_SPEC else make_tier_pipeline(
        INTERPRETABLE_SPEC, seed
    ).fit(train_x, train.tiers)

    report = TrainingReport(
        selected=selected,
        selection_reason=reason,
        cv={k: v for k, v in cv.items()},
        test=test_metrics,
        test_errors=test_errors,
        test_other_models=other,
        threshold_table=thresholds_table,
        task_type_test_accuracy=task_accuracy,
        top_features=top_tfidf_features(explain_model),
        training_size=len(train),
        test_size=len(test),
        tier_counts_train={int(t): int(c) for t, c in zip(*np.unique(train.tiers, return_counts=True), strict=True)},
        split_created=created,
        test_ids_missing=missing_test_ids(data, split),
    )
    logger.info("Trained %s: test accuracy %.3f", selected.name, test_metrics.accuracy)
    return TrainedModels(tier_pipeline=tier_pipeline, task_pipeline=task_pipeline, report=report, test_data=test)

