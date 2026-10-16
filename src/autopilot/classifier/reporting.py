"""Console output for training runs (shared by train_classifier.py and retrain.py)."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from autopilot.classifier.metrics import EvalMetrics
from autopilot.classifier.training import TrainingReport
from autopilot.classifier.workflow import TrainingOutcome

V1_ACCURACY_TARGET = 0.80


def pct(value: float) -> str:
    return f"{value:.0%}"


def print_cv(report: TrainingReport, console: Console) -> None:
    table = Table(title=f"Cross-validation on the training set ({next(iter(report.cv.values())).folds} folds)")
    for col in ("Model", "Accuracy", "Macro F1", "Tier-3 recall", "Under-routing", "Over-routing"):
        table.add_column(col, justify="left" if col == "Model" else "right")
    for kind, cv in report.cv.items():
        marker = " ✔" if kind == report.selected.name else ""
        table.add_row(
            kind + marker,
            f"{pct(cv.accuracy_mean)} ± {pct(cv.accuracy_std)}",
            pct(cv.macro_f1_mean),
            f"{pct(cv.tier3_recall_mean)} ± {pct(cv.tier3_recall_std)}",
            pct(cv.under_routing_mean),
            pct(cv.over_routing_mean),
        )
    console.print(table)
    console.print(f"Selected [bold]{report.selected.name}[/bold]: {report.selection_reason}")


def print_test(report: TrainingReport, console: Console) -> None:
    test: EvalMetrics = report.test
    table = Table(title=f"Held-out test set (n={test.n}) — {report.selected.name}")
    for col in ("Tier", "Precision", "Recall", "F1", "Support"):
        table.add_column(col, justify="right")
    for tier, scores in test.per_tier.items():
        table.add_row(str(tier), pct(scores.precision), pct(scores.recall), pct(scores.f1), str(scores.support))
    console.print(table)

    matrix = Table(title="Confusion matrix (rows = true tier, columns = predicted)")
    matrix.add_column("true ↓ / pred →")
    for tier in (1, 2, 3):
        matrix.add_column(str(tier), justify="right")
    for tier, row in zip((1, 2, 3), test.confusion_matrix, strict=True):
        cells = [f"[bold]{v}[/bold]" if i + 1 == tier else (f"[red]{v}[/red]" if i + 1 < tier and v else str(v))
                 for i, v in enumerate(row)]
        matrix.add_row(str(tier), *cells)
    console.print(matrix)
    console.print("[dim]Red = under-routed (below the diagonal): hard prompt sent to a weaker model.[/dim]")

    ok = "✓" if test.accuracy >= V1_ACCURACY_TARGET else "✗"
    console.print(
        f"Accuracy [bold]{pct(test.accuracy)}[/bold] ({ok} target ≥{pct(V1_ACCURACY_TARGET)}) · "
        f"Tier-3 recall [bold]{pct(test.tier3_recall)}[/bold] · under-routing {pct(test.under_routing_rate)} "
        f"(severe T3→T1: {test.severe_under_routing}) · over-routing {pct(test.over_routing_rate)}"
    )
    for kind, other in report.test_other_models.items():
        console.print(
            f"[dim]For reference only (not used to choose): {kind} test accuracy {pct(other.accuracy)}, "
            f"Tier-3 recall {pct(other.tier3_recall)}[/dim]"
        )


def print_errors(report: TrainingReport, console: Console) -> None:
    if not report.test_errors:
        return
    table = Table(title="Misclassified test prompts")
    for col in ("id", "true", "pred", "conf", "prompt"):
        table.add_column(col, overflow="fold" if col == "prompt" else "ellipsis")
    for e in sorted(report.test_errors, key=lambda e: e.predicted_tier - e.true_tier):
        style = "red" if e.predicted_tier < e.true_tier else ""
        table.add_row(e.candidate_id, str(e.true_tier), str(e.predicted_tier), f"{e.confidence:.2f}", e.prompt_preview, style=style)
    console.print(table)


def print_thresholds(report: TrainingReport, console: Console) -> None:
    table = Table(title="Route-up threshold (out-of-fold predictions on the training set)")
    for col in ("Threshold", "Routed up", "Accuracy", "Under-routing", "Over-routing", "Tier-3 recall", "Traffic T1/T2/T3"):
        table.add_column(col, justify="right")
    for row in report.threshold_table:
        share = "/".join(pct(row.tier_share[t]) for t in (1, 2, 3))
        label = "off" if row.threshold == 0 else f"{row.threshold:.2f}"
        table.add_row(label, pct(row.routed_up_rate), pct(row.accuracy), pct(row.under_routing_rate),
                      pct(row.over_routing_rate), pct(row.tier3_recall), share)
    console.print(table)


def print_features(report: TrainingReport, console: Console) -> None:
    table = Table(title="Strongest signals per tier (logistic-regression weights)")
    table.add_column("Tier")
    table.add_column("Top features")
    for tier, features in report.top_features.items():
        table.add_row(str(tier), ", ".join(name for name, _ in features))
    console.print(table)


def print_report(report: TrainingReport, console: Console, *, split_name: str) -> None:
    """Everything about one training run: data, CV comparison, test results, errors, thresholds."""
    split_note = "created and frozen" if report.split_created else "reused (frozen)"
    console.print(
        f"Train {report.training_size} · test {report.test_size} — test split {split_note}: {split_name}. "
        f"Train tier counts: {report.tier_counts_train}"
    )
    if report.test_ids_missing:
        console.print(
            f"[yellow]{len(report.test_ids_missing)} frozen test prompts have no label in this file and are "
            f"left out of the test set: {', '.join(report.test_ids_missing)}[/yellow]"
        )
    print_cv(report, console)
    print_test(report, console)
    print_errors(report, console)
    print_thresholds(report, console)
    print_features(report, console)
    console.print(f"Task-type classifier (4 groups) test accuracy: {pct(report.task_type_test_accuracy)}")


def print_outcome(outcome: TrainingOutcome, console: Console, *, models_dir_name: str) -> None:
    """Saved version, comparison with the current model, and the promotion result."""
    if outcome.metadata is not None:
        version = outcome.metadata.version
        console.print(f"[green]✓ Saved {version} → {models_dir_name}/classifier_{version}.joblib (+ .json)[/green]")
    if outcome.current_metrics is not None:
        m = outcome.current_metrics
        console.print(
            f"Current model {outcome.current_version} on this test set: accuracy {pct(m.accuracy)}, "
            f"Tier-3 recall {pct(m.tier3_recall)}, under-routing {pct(m.under_routing_rate)}"
        )
    if outcome.metadata is None:
        console.print(f"[yellow]Dry run — nothing saved. Promotion rule would say: {outcome.decision.reason}[/yellow]")
    elif outcome.promoted:
        console.print(f"[green]✓ {outcome.metadata.version} is now the current model ({outcome.promotion_reason})[/green]")
    else:
        console.print(
            f"[yellow]{outcome.metadata.version} saved but NOT promoted: {outcome.promotion_reason}. "
            f"Current stays {outcome.current_version}.[/yellow]"
        )
