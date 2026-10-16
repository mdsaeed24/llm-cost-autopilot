"""HTTP API: contracts, validation, error shapes, runtime config, auth. LLMs are faked."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from autopilot.api.app import create_app
from autopilot.llm.catalog import load_catalog
from autopilot.routing.config import ConfidenceRule, RoutingConfig, effective_routing_config
from autopilot.settings import ApiSettings, AppSettings
from autopilot.verification.worker import Worker
from tests.test_verification_flow import World, build

ASK = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


class Env:
    def __init__(self, client: TestClient, world: World, worker: Worker) -> None:
        self.client, self.world, self.worker = client, world, worker
        self.service = client.app.state.service  # type: ignore[attr-defined]


@pytest.fixture
def env(tmp_path: Path, api_key: str) -> Iterator[Env]:
    yield from make_env(tmp_path)


def make_env(tmp_path: Path, *, settings: AppSettings | None = None, tier: int = 1) -> Iterator[Env]:
    world = World()
    service, worker, _ = build(tmp_path, load_catalog(), world, tier=tier)
    app = create_app(service=service, settings=settings or AppSettings(api=ApiSettings(max_total_chars=500)))
    with TestClient(app) as client:
        yield Env(client, world, worker)


def error_of(response: Any) -> dict[str, Any]:
    body = response.json()
    assert set(body) == {"error"}, body
    return body["error"]  # type: ignore[no-any-return]


# ---------- completions ----------

def test_completion_contract(env: Env) -> None:
    response = env.client.post("/v1/completions", json={**ASK, "model": "auto", "temperature": 0.1})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == "llama3.2:3b"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "CHEAP-ANSWER"}
    assert body["usage"] == {"prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500}
    meta = body["autopilot"]
    assert (meta["chosen_model"], meta["chosen_tier"], meta["predicted_tier"]) == ("local-llama", 1, 1)
    assert meta["classifier_confidence"] == 0.9 and meta["routed_up"] is False and meta["escalated"] is False
    assert meta["reason"].startswith("classifier predicted tier 1")
    assert meta["verification_status"] == "queued"
    assert response.headers["X-Autopilot-Request-Id"] == meta["request_id"]
    assert env.service.repo.get_request(meta["request_id"]) is not None


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({**ASK, "model": "gpt-4o"}, "model selection is automatic"),
        ({**ASK, "stream": True}, "streaming is not supported"),
        ({"messages": [{"role": "system", "content": "hi"}]}, "at least one 'user' message"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "user", "content": ""}]}, "messages.0.content"),
        ({"messages": [{"role": "robot", "content": "hi"}]}, "messages.0.role"),
        ({**ASK, "temperature": 5}, "temperature"),
        ({**ASK, "top_k": 3}, "top_k"),
        ({"messages": [{"role": "user", "content": "x" * 600}]}, "request too long"),
        ({**ASK, "max_tokens": 100_000}, "max_tokens must be"),
    ],
)
def test_invalid_requests_get_clear_422s(env: Env, payload: dict[str, Any], fragment: str) -> None:
    response = env.client.post("/v1/completions", json=payload)
    assert response.status_code == 422
    error = error_of(response)
    assert error["type"] == "invalid_request"
    assert fragment in error["message"] + str(error.get("details", ""))


def test_upstream_failure_is_502(env: Env) -> None:
    env.world.down |= {"llama3.2:3b", "deepseek-flash", "deepseek-v4-pro"}
    response = env.client.post("/v1/completions", json=ASK)
    assert response.status_code == 502
    assert error_of(response)["type"] == "upstream_error"


def test_missing_provider_key_is_503(tmp_path: Path) -> None:
    for e in make_env(tmp_path, tier=2):  # no api_key fixture → DEEPSEEK_API_KEY unset
        response = e.client.post("/v1/completions", json=ASK)
        assert response.status_code == 503
        assert error_of(response)["type"] == "provider_not_configured"


# ---------- models, stats, health ----------

def test_models(env: Env) -> None:
    body = env.client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["local-llama", "deepseek-flash", "deepseek-v4-pro"]
    pro = body["data"][2]
    assert pro["serves_tiers"] == [3] and pro["pricing_usd"]["output_per_1m"] == 3.96
    assert "PEAK" in body["pricing_basis"] and "DeepSeek pricing" in body["pricing_now"]


async def test_stats_pair_savings_with_quality(env: Env) -> None:
    env.client.post("/v1/completions", json=ASK)
    await env.worker.run(once=True)
    body = env.client.get("/v1/stats").json()
    summary = body["summary"]
    assert summary["quality"]["judged"] == 1 and summary["quality"]["parity"] == 1.0
    assert "quality parity" in body["headline"] and "net of verification" in body["headline"]
    for key in ("net_savings_pct", "gross_savings_pct", "verification_cost_usd", "tiers"):
        assert key in summary
    assert env.client.get("/v1/stats", params={"mode": "evaluation"}).json()["summary"]["requests"] == 0
    assert env.client.get("/v1/stats", params={"mode": "bogus"}).status_code == 422
    assert env.client.get("/v1/stats", params={"since_hours": 0}).status_code == 422
    future = env.client.get("/v1/stats", params={"since": "2999-01-01T00:00:00Z"}).json()
    assert future["summary"]["requests"] == 0
    past = env.client.get("/v1/stats", params={"since": "2000-01-01T00:00:00"}).json()  # naive → UTC
    assert past["summary"]["requests"] == 1


def test_health(env: Env) -> None:
    body = env.client.get("/health").json()
    assert body["status"] == "ok" and body["classifier_version"] == "stub"


# ---------- routing config ----------

NEW_TIERS = {"1": "deepseek-flash", "2": "deepseek-flash", "3": "deepseek-v4-pro"}


def test_routing_config_update_applies_logs_and_persists(env: Env) -> None:
    assert env.client.get("/v1/routing-config").json()["source"] == "routing.yaml"
    response = env.client.put("/v1/routing-config",
                              json={"tiers": NEW_TIERS, "confidence_threshold": 0.6, "reason": "llama under review"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "runtime" and body["config"]["tiers"]["1"] == "deepseek-flash"
    assert body["config"]["confidence_threshold"] == 0.6 and body["change_id"] == 1

    answer = env.client.post("/v1/completions", json=ASK).json()
    assert answer["autopilot"]["chosen_model"] == "deepseek-flash"  # applied immediately

    change = env.service.repo.list_config_changes()[0]
    assert change["reason"] == "llama under review" and '"local-llama"' in change["old_config"]
    base = RoutingConfig(tiers={1: "local-llama", 2: "deepseek-flash", 3: "deepseek-v4-pro"},
                         confidence=ConfidenceRule(threshold=0.5))
    restored = effective_routing_config(base, env.service.repo, load_catalog())  # what a restart / the worker sees
    assert restored.tiers[1] == "deepseek-flash" and restored.confidence.threshold == 0.6


@pytest.mark.parametrize(
    ("tiers", "fragment"),
    [
        ({"1": "local-llama", "2": "deepseek-flash", "3": "gpt-9"}, "unknown models"),
        ({"1": "deepseek-v4-pro", "2": "deepseek-flash", "3": "local-llama"}, "must not get weaker"),
        ({"1": "local-llama", "2": "deepseek-flash"}, "missing"),
    ],
)
def test_invalid_routing_config_rejected(env: Env, tiers: dict[str, str], fragment: str) -> None:
    response = env.client.put("/v1/routing-config", json={"tiers": tiers})
    assert response.status_code == 422
    error = error_of(response)
    assert error["type"] == "invalid_config" and fragment in error["message"]
    assert env.service.repo.list_config_changes() == []  # nothing logged, nothing applied
    assert env.service.router.config.tiers[1] == "local-llama"


# ---------- auth & docs ----------

def test_api_key_required_when_configured(tmp_path: Path, api_key: str) -> None:
    for e in make_env(tmp_path, settings=AppSettings(api=ApiSettings(api_key="s3cret"))):
        denied = e.client.get("/v1/models")
        assert denied.status_code == 401 and error_of(denied)["type"] == "unauthorized"
        assert e.client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert e.client.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert e.client.get("/health").status_code == 200  # liveness stays open


def test_openapi_is_documented(env: Env) -> None:
    spec = env.client.get("/openapi.json").json()
    for path, method in [("/v1/completions", "post"), ("/v1/models", "get"), ("/v1/stats", "get"),
                         ("/v1/routing-config", "put")]:
        operation = spec["paths"][path][method]
        assert operation["summary"] and operation["tags"]
        assert "422" in operation["responses"]
    assert spec["components"]["schemas"]["CompletionRequest"]["examples"]
    assert env.client.get("/docs").status_code == 200


def test_unknown_route_uses_error_shape(env: Env) -> None:
    response = env.client.get("/v1/nope")
    assert response.status_code == 404 and error_of(response)["type"] == "not_found"


async def test_worker_picks_up_runtime_routing_change(env: Env) -> None:
    catalog = load_catalog()
    base = env.service.router.config
    worker = Worker(env.service.repo, env.worker.verifier, env.worker.settings,
                    refresh_routing=lambda: effective_routing_config(base, env.service.repo, catalog))
    env.client.put("/v1/routing-config", json={"tiers": NEW_TIERS})
    await worker.run_batch()
    assert worker.verifier.routing.tiers[1] == "deepseek-flash"
