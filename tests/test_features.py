"""Feature extraction: instruction/context split, cue counting, the sklearn pipeline."""

from __future__ import annotations

import io

import joblib
import numpy as np

from autopilot.classifier.features import (
    HANDCRAFTED_FEATURE_NAMES,
    HandcraftedFeatures,
    build_feature_union,
    constraint_count,
    feature_names,
    format_complexity,
    handcrafted_features,
    split_prompt,
)

LONG = " ".join(["word"] * 50)


def features_of(text: str) -> dict[str, float]:
    return dict(zip(HANDCRAFTED_FEATURE_NAMES, handcrafted_features(text), strict=True))


def test_split_uses_context_marker() -> None:
    assert split_prompt("Summarise this.\n\nContext:\nSome text.") == ("Summarise this.", "Some text.")


def test_split_heuristic_finds_long_paragraph_either_side() -> None:
    assert split_prompt(f"Summarise this:\n\n{LONG}") == ("Summarise this:", LONG)
    assert split_prompt(f"{LONG}\n\nWhat is the main idea?") == ("What is the main idea?", LONG)


def test_split_without_context() -> None:
    assert split_prompt("Why is the sky blue?") == ("Why is the sky blue?", "")
    assert split_prompt("Short one.\n\nShort two.") == ("Short one.\n\nShort two.", "")


def test_cues_are_counted_in_instruction_only() -> None:
    f = features_of("Who wrote it?\n\nContext:\nWhy explain compare analyse evaluate.")
    assert f["reasoning_verbs"] == 0
    assert f["lookup_verbs"] == 1  # "who"
    assert f["has_context"] == 1
    assert f["question_count"] == 1


def test_reasoning_and_creative_cues() -> None:
    f = features_of("Why should we compare these? Explain, then write a poem.")
    assert f["reasoning_verbs"] == 4  # why, should, compare, explain
    assert f["creative_verbs"] == 2   # write, poem
    assert f["has_context"] == 0


def test_constraints_and_formats() -> None:
    assert constraint_count("Write exactly 6 lines and do not use the word sea") == 3
    assert format_complexity("Return JSON in a markdown table") == 6
    assert format_complexity("Give a numbered list") == 2
    assert format_complexity("Tell me a joke") == 0


def test_long_easy_prompt_does_not_look_hard() -> None:
    easy = features_of(f"Extract the dates.\n\nContext:\n{LONG}")
    hard = features_of("Why did Rome fall?")
    assert easy["log_tokens_total"] > hard["log_tokens_total"]
    assert easy["reasoning_verbs"] < hard["reasoning_verbs"]


def test_handcrafted_transformer_shape() -> None:
    out = HandcraftedFeatures().fit(["a"]).transform(["Why?", "List three fruits"])
    assert out.shape == (2, len(HANDCRAFTED_FEATURE_NAMES))
    assert list(HandcraftedFeatures().get_feature_names_out()) == list(HANDCRAFTED_FEATURE_NAMES)


def test_feature_union_keeps_question_words_and_pickles() -> None:
    texts = np.array(["Why is it so?", "Why not?", "List the names", "List items\n\nContext:\nwhy why why"], dtype=object)
    union = build_feature_union()
    matrix = union.fit_transform(texts)
    names = feature_names(union)
    assert matrix.shape == (4, len(names))
    assert "tfidf:why" in names  # not removed as a stop word
    assert "tfidf:list" in names

    buffer = io.BytesIO()
    joblib.dump(union, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    assert (restored.transform(texts) != matrix).nnz == 0
