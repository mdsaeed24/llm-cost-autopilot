"""LLM judge (blinding, parsing, scoring) and quality-rule selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from autopilot.llm.catalog import load_catalog
from autopilot.llm.schemas import ChatMessage, ModelConfig
from autopilot.settings import ConfigError
from autopilot.verification.judge import JudgeError, agreement_score, judge, parse_verdict
from autopilot.verification.rules import QualityRule, load_quality_config
from tests.conftest import FakeOpenAI, make_client, make_completion

RULE = QualityRule(criterion="The final answer is correct.", pass_threshold=4)
MESSAGES = [ChatMessage(role="user", content="What is 2+2?")]


@pytest.mark.parametrize(("routed", "reference", "expected"), [(5, 5, 5), (4, 5, 4), (2, 5, 2), (1, 5, 1), (5, 3, 5), (1, 2, 4)])
def test_agreement_score(routed: int, reference: int, expected: int) -> None:
    assert agreement_score(routed, reference) == expected


def test_parse_verdict_tolerates_fences() -> None:
    verdict = parse_verdict('```json\n{"score_a": 4, "score_b": 2, "reason": "B is wrong"}\n```')
    assert (verdict.score_a, verdict.score_b) == (4, 2)
    with pytest.raises(ValueError):
        parse_verdict("I think A is better")
    with pytest.raises(ValidationError):
        parse_verdict('{"score_a": 9, "score_b": 2}')


def scoring_judge(routed_text: str, routed_score: int, reference_score: int) -> Any:
    """Fake judge that finds the routed answer in A or B and scores accordingly."""

    async def reply(kwargs: dict[str, Any]) -> Any:
        user = kwargs["messages"][1]["content"]
        a_block = user.split("<answer_a>")[1].split("</answer_a>")[0]
        routed_is_a = routed_text in a_block
        scores = (routed_score, reference_score) if routed_is_a else (reference_score, routed_score)
        return make_completion(json.dumps({"score_a": scores[0], "score_b": scores[1], "reason": "because"}))

    return reply


@pytest.mark.parametrize("seed", range(6))  # both A/B orders get exercised
async def test_judge_maps_blinded_scores_back(seed: int, flash_model: ModelConfig, api_key: str) -> None:
    fake = FakeOpenAI([scoring_judge("ALPHA-ANSWER", routed_score=2, reference_score=5)])
    client, _ = make_client(fake)
    result = await judge(client, flash_model, messages=MESSAGES, routed_answer="ALPHA-ANSWER", reference_answer="OMEGA-ANSWER",
                         rule=RULE, rng=random.Random(seed), max_tokens=300)
    assert (result.routed_score, result.reference_score, result.agreement) == (2, 5, 2)

    call = fake.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0
    user = call["messages"][1]["content"]
    assert not any(word in user.lower() for word in ("reference", "routed", "tier", "cheap"))  # blinded
    assert RULE.criterion in user


async def test_unparseable_judge_reply_keeps_cost(flash_model: ModelConfig, api_key: str) -> None:
    client, _ = make_client(FakeOpenAI([make_completion("no json here", prompt_tokens=1000, completion_tokens=10)]))
    with pytest.raises(JudgeError) as info:
        await judge(client, flash_model, messages=MESSAGES, routed_answer="a", reference_answer="b",
                    rule=RULE, rng=random.Random(0), max_tokens=300)
    assert info.value.response.cost_usd > 0


def test_repo_quality_config_and_rule_selection() -> None:
    config = load_quality_config(load_catalog())
    assert config.judge_model == "deepseek-v4-pro"
    name, rule = config.select("summarization_analysis", 0.9)
    assert name == "summarization_analysis" and rule.pass_threshold == 4
    assert config.select("summarization_analysis", 0.2)[0] == "default"  # unsure task type
    assert config.select(None, None)[0] == "default"


def test_quality_config_validation(tmp_path: Path) -> None:
    path = tmp_path / "quality.yaml"
    path.write_text(
        "judge_model: deepseek-v4-pro\nmin_task_confidence: 0.5\n"
        "rules:\n  qa_classification:\n    criterion: The answer is correct.\n    pass_threshold: 4\n"
        "default:\n  criterion: The answer is fine and useful.\n  pass_threshold: 4\n"
    )
    with pytest.raises(ValidationError, match="missing"):
        load_quality_config(load_catalog(), path)

    good = Path("config/quality.yaml").read_text().replace("judge_model: deepseek-v4-pro", "judge_model: gpt-9")
    path.write_text(good)
    with pytest.raises(ConfigError, match="gpt-9"):
        load_quality_config(load_catalog(), path)
