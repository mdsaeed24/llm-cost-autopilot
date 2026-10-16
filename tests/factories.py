"""Synthetic labeled prompts with a learnable tier signal (tests never touch the real labels)."""

from __future__ import annotations

import random
from pathlib import Path

from autopilot.dataset.io import write_jsonl
from autopilot.dataset.schemas import LabeledPrompt

TOPICS = [
    "the Roman Empire", "photosynthesis", "electric cars", "jazz music", "the stock market", "volcanoes",
    "remote work", "coffee farming", "the moon landing", "machine learning", "chess openings", "honeybees",
    "the French Revolution", "solar panels", "marathon training", "public libraries", "deep sea fish",
]

PASSAGE = (
    "{topic} has a long and well documented history. Researchers have described its origins, the people "
    "involved, the key dates and the places where it developed. Several sources agree on the main facts, "
    "although some details are still debated by historians and scientists today, and new evidence appears."
)

TEMPLATES: dict[int, list[tuple[str, str]]] = {
    1: [
        ("Extract the key dates mentioned in the text about {topic}.\n\nContext:\n" + PASSAGE, "extraction_reformatting"),
        ("Who is mentioned in this passage about {topic}?\n\nContext:\n" + PASSAGE, "qa_classification"),
        ("Rewrite this text about {topic} as a bulleted list.\n\nContext:\n" + PASSAGE, "extraction_reformatting"),
        ("When did {topic} start, according to the passage?\n\nContext:\n" + PASSAGE, "qa_classification"),
    ],
    2: [
        ("Summarize this text about {topic} in two sentences.\n\nContext:\n" + PASSAGE, "summarization_analysis"),
        ("Classify each of these as related to {topic} or not: apples, rivers, books", "qa_classification"),
        ("Summarise the main points of {topic}", "summarization_analysis"),
        ("Classify {topic} as science, history, or culture", "qa_classification"),
    ],
    3: [
        ("Why did {topic} matter so much? Explain your reasoning step by step.", "reasoning_creative"),
        ("Write a short poem about {topic} with exactly 6 lines.", "reasoning_creative"),
        ("Should a city invest in {topic}? Weigh the pros and cons and recommend.", "reasoning_creative"),
        ("Imagine a story where {topic} never existed. What would change and why?", "reasoning_creative"),
    ],
}


def synthetic_labels(per_tier: int = 30, seed: int = 0) -> list[LabeledPrompt]:
    """``per_tier`` prompts for each tier, cycling templates and topics."""
    rng = random.Random(seed)
    records: list[LabeledPrompt] = []
    for tier, templates in TEMPLATES.items():
        for i in range(per_tier):
            template, task_type = templates[i % len(templates)]
            topic = TOPICS[(i * 7 + tier) % len(TOPICS)]
            records.append(
                LabeledPrompt(
                    candidate_id=f"c{tier}{i:03d}",
                    prompt=template.format(topic=topic),
                    tier=tier,  # type: ignore[arg-type]
                    task_type=task_type,  # type: ignore[arg-type]
                )
            )
    rng.shuffle(records)
    return records


def write_synthetic_labels(path: Path, per_tier: int = 30) -> Path:
    write_jsonl(path, synthetic_labels(per_tier))
    return path
