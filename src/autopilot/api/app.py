"""FastAPI application.

    uvicorn autopilot.api.app:app --port 8000      # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from autopilot import __version__
from autopilot.analytics.metrics import load_frames, summarise
from autopilot.api.errors import ApiError, register_error_handlers
from autopilot.api.schemas import (
    AssistantMessage,
    Choice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelPrices,
    ModelsResponse,
    RoutingConfigResponse,
    RoutingConfigUpdate,
    RoutingConfigView,
    RoutingMetadata,
    StatsResponse,
    Usage,
)
from autopilot.bootstrap import build_service
from autopilot.llm.pricing import peak_status
from autopilot.routing.config import ConfidenceRule, RoutingConfig
from autopilot.service import AutopilotService
from autopilot.settings import AppSettings, ConfigError, load_settings

DESCRIPTION = """
A routing layer in front of several LLMs. Each request is classified by complexity and sent to the
cheapest model that can handle it; a random sample is verified in the background against the top
tier, and confirmed routing failures become training data.

**Honest metrics:** `/v1/stats` never reports savings without quality parity, verification
overhead and net savings. All costs use DeepSeek *peak* prices (off-peak is 50% cheaper).
"""

TAGS = [
    {"name": "completions", "description": "Send a chat request; the autopilot chooses the model."},
    {"name": "models", "description": "Models behind the router, with prices."},
    {"name": "stats", "description": "Savings, quality parity and escalations — always reported together."},
    {"name": "config", "description": "Inspect and change the tier → model mapping at runtime."},
    {"name": "health", "description": "Liveness."},
]

ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing or wrong API key (only when one is configured)."},
    422: {"model": ErrorResponse, "description": "Invalid request."},
}

bearer = HTTPBearer(auto_error=False)


def get_service(request: Request) -> AutopilotService:
    return request.app.state.service  # type: ignore[no-any-return]


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings  # type: ignore[no-any-return]


def require_api_key(
    request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> None:
    """No-op unless ``api.api_key`` (AUTOPILOT_API_KEY) is set."""
    expected = get_settings(request).api.api_key
    if expected is None:
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise ApiError(401, "unauthorized", "Missing or invalid API key (send 'Authorization: Bearer <key>').")


Service = Annotated[AutopilotService, Depends(get_service)]
Settings = Annotated[AppSettings, Depends(get_settings)]
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)], responses=ERRORS)


# ---------- completions ----------


@v1.post(
    "/completions",
    response_model=CompletionResponse,
    tags=["completions"],
    summary="Create a chat completion (model chosen automatically)",
    responses={
        502: {"model": ErrorResponse, "description": "Every model tier failed to answer."},
        503: {"model": ErrorResponse, "description": "A provider API key is not configured."},
    },
)
async def create_completion(body: CompletionRequest, service: Service, settings: Settings, response: Response) -> CompletionResponse:
    """Classifies the request, routes it to the cheapest suitable model, logs it, and (for a random
    sample) queues a background quality check. The `autopilot` field explains the decision."""
    limits = settings.api
    if len(body.messages) > limits.max_messages:
        raise ApiError(422, "invalid_request", f"too many messages ({len(body.messages)} > {limits.max_messages})")
    total_chars = sum(len(m.content) for m in body.messages)
    if total_chars > limits.max_total_chars:
        raise ApiError(422, "invalid_request",
                       f"request too long ({total_chars} characters > {limits.max_total_chars})")
    if body.max_tokens is not None and body.max_tokens > limits.max_tokens_cap:
        raise ApiError(422, "invalid_request", f"max_tokens must be ≤ {limits.max_tokens_cap}")

    result = await service.complete(
        [m.model_dump() for m in body.messages], max_tokens=body.max_tokens, temperature=body.temperature
    )
    decision, answer = result.decision, result.response
    response.headers["X-Autopilot-Request-Id"] = result.request_id
    return CompletionResponse(
        id=f"chatcmpl-{result.request_id}",
        created=int(time.time()),
        model=answer.model_id,
        choices=[Choice(message=AssistantMessage(content=result.output_text), finish_reason=answer.finish_reason)],
        usage=Usage(prompt_tokens=answer.input_tokens, completion_tokens=answer.output_tokens,
                    total_tokens=answer.input_tokens + answer.output_tokens),
        autopilot=RoutingMetadata(
            request_id=result.request_id,
            chosen_model=result.served_model,
            chosen_tier=result.served_tier,
            routed_model=decision.model_name,
            predicted_tier=decision.predicted_tier,
            classifier_confidence=decision.confidence,
            routed_up=decision.routed_up,
            failover=result.failover,
            escalated=result.escalated,
            reason=decision.reason,
            task_type=decision.task_type,
            classifier_version=decision.classifier_version,
            cost_usd=answer.cost_usd,
            latency_ms=result.latency_ms,
            verification_status=result.verification_status,
        ),
    )


# ---------- models ----------


@v1.get("/models", response_model=ModelsResponse, tags=["models"], summary="List models and prices")
async def list_models(service: Service) -> ModelsResponse:
    tiers = service.router.config.tiers
    data = [
        ModelInfo(
            id=m.name,
            model_id=m.model_id,
            provider=m.provider,
            quality_tier=m.quality_tier,
            serves_tiers=sorted(t for t, name in tiers.items() if name == m.name),
            pricing_usd=ModelPrices(**m.pricing.model_dump()),
            typical_latency_ms=m.typical_latency_ms,
            thinking_enabled=m.thinking.enabled if m.thinking else None,
        )
        for m in service.catalog.by_quality()
    ]
    return ModelsResponse(
        data=data,
        pricing_basis="USD per 1M tokens, DeepSeek PEAK prices (conservative; off-peak is 50% cheaper)",
        pricing_now=peak_status(service.catalog.price_schedule).describe(),
    )


# ---------- stats ----------


@v1.get("/stats", response_model=StatsResponse, tags=["stats"], summary="Savings, quality parity and escalations")
async def get_stats(
    service: Service,
    mode: Annotated[Literal["normal", "evaluation"] | None, Query(description="Only this traffic mode.")] = None,
    since_hours: Annotated[int | None, Query(ge=1, le=24 * 365, description="Only the last N hours.")] = None,
    since: Annotated[datetime | None, Query(description="Only requests at/after this ISO-8601 time (e.g. a load-test start).")] = None,
) -> StatsResponse:
    """Net savings after verification costs, next to quality parity (random-sample verifications only,
    with a 95% confidence interval), exclusions, escalation rate, and per-tier truncation and latency."""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if since_hours:
        hours_ago = datetime.now(UTC) - timedelta(hours=since_hours)
        since = max(since, hours_ago) if since else hours_ago
    frames = load_frames(service.repo.db, mode=mode, since=since)
    tier3 = service.catalog.get(service.router.config.tiers[3])
    summary = summarise(frames, tier3_model=tier3.name, tier3_pricing=tier3.pricing)
    return StatsResponse(headline=summary.headline(), mode=mode, since_hours=since_hours, summary=summary)


# ---------- routing config ----------


def _view(config: RoutingConfig) -> RoutingConfigView:
    return RoutingConfigView(
        tiers=dict(config.tiers), confidence_enabled=config.confidence.enabled,
        confidence_threshold=config.confidence.threshold, fallback_tier=config.fallback_tier,
        classifier_version=config.classifier_version,
    )


@v1.get("/routing-config", response_model=RoutingConfigResponse, tags=["config"], summary="Current routing config")
async def get_routing_config(service: Service) -> RoutingConfigResponse:
    changes = service.repo.list_config_changes()
    return RoutingConfigResponse(
        config=_view(service.router.config),
        source="runtime" if changes else "routing.yaml",
        change_id=changes[-1]["id"] if changes else None,
        changes_logged=len(changes),
    )


@v1.put("/routing-config", response_model=RoutingConfigResponse, tags=["config"],
        summary="Update the tier → model mapping at runtime")
async def update_routing_config(body: RoutingConfigUpdate, service: Service, request: Request) -> RoutingConfigResponse:
    """Validated (known models, every tier mapped, quality never decreasing with tier), applied
    immediately, logged with the old and new config, and picked up by the worker. Survives restarts."""
    current = service.router.config
    try:
        new = RoutingConfig(
            tiers=body.tiers,  # type: ignore[arg-type]
            confidence=ConfidenceRule(
                enabled=current.confidence.enabled if body.confidence_enabled is None else body.confidence_enabled,
                threshold=current.confidence.threshold if body.confidence_threshold is None else body.confidence_threshold,
            ),
            fallback_tier=current.fallback_tier if body.fallback_tier is None else body.fallback_tier,
            classifier_version=current.classifier_version,
        )
        new.validate_against(service.catalog)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        raise ApiError(422, "invalid_config", problems) from error
    except ConfigError as error:
        raise ApiError(422, "invalid_config", str(error)) from error

    change_id = service.repo.record_config_change(
        current.model_dump_json(), new.model_dump_json(),
        actor=request.client.host if request.client else None, reason=body.reason,
    )
    service.router.config = new          # next request uses it
    service.verifier.routing = new       # sync verification uses the new Tier 3
    return RoutingConfigResponse(config=_view(new), source="runtime", change_id=change_id,
                                 changes_logged=len(service.repo.list_config_changes()))


# ---------- app ----------


def create_app(*, service: AutopilotService | None = None, settings: AppSettings | None = None) -> FastAPI:
    """Build the app. Tests pass a ready-made ``service``; production builds one from config."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.service = service or build_service(settings)
        try:
            yield
        finally:
            if service is None:
                await app.state.service.client.aclose()

    app = FastAPI(title="LLM Cost Autopilot", version=__version__, description=DESCRIPTION,
                  openapi_tags=TAGS, lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(v1)

    @app.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness check")
    async def health(request: Request) -> HealthResponse:
        svc: AutopilotService = request.app.state.service
        return HealthResponse(
            status="ok", classifier_version=svc.router.classifier.version, mode=svc.settings.mode,
            escalation_mode=svc.settings.escalation_mode, queue=svc.repo.queue_counts(),
        )

    return app


app = create_app()
