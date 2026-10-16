"""Request and response bodies for the HTTP API (these drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autopilot.analytics.metrics import HonestSummary

# ---------- errors ----------


class ErrorDetail(BaseModel):
    type: str = Field(description="Machine-readable error type.", examples=["invalid_request"])
    message: str = Field(description="Human-readable explanation.")
    details: list[dict[str, Any]] | None = Field(default=None, description="Per-field problems, when relevant.")


class ErrorResponse(BaseModel):
    """Every error the API returns has this shape."""

    error: ErrorDetail


# ---------- completions ----------


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class CompletionRequest(BaseModel):
    """OpenAI-style chat request. The caller does NOT choose the model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"messages": [{"role": "user", "content": "Summarise the plot of Hamlet in two sentences."}]},
                {
                    "model": "auto",
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": "Why is the sky blue?"},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2,
                },
            ]
        },
    )

    messages: list[MessageIn] = Field(min_length=1, description="Conversation; must contain a user message.")
    model: str | None = Field(default=None, description='Omit, or "auto". The autopilot picks the model.')
    max_tokens: int | None = Field(default=None, ge=1, description="Output token cap (bounded by the server).")
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = Field(default=False, description="Streaming is not supported; must be false.")

    @model_validator(mode="after")
    def _check(self) -> CompletionRequest:
        if self.model not in (None, "auto"):
            raise ValueError(f"model selection is automatic: omit 'model' or set it to \"auto\" (got {self.model!r})")
        if self.stream:
            raise ValueError("streaming is not supported: set 'stream' to false")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("messages must contain at least one 'user' message")
        return self


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutingMetadata(BaseModel):
    """How the autopilot produced this answer."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    chosen_model: str = Field(description="Model (name from /v1/models) whose answer is returned.")
    chosen_tier: int
    routed_model: str = Field(description="Model the router picked (differs after failover or sync escalation).")
    predicted_tier: int | None = Field(description="Classifier's tier; null if the classifier was unavailable.")
    classifier_confidence: float | None
    routed_up: bool = Field(description="True if low confidence moved the request one tier up.")
    failover: bool = Field(description="True if the routed model was down and a higher tier answered.")
    escalated: bool = Field(description="True if sync verification replaced the answer with Tier 3's.")
    reason: str
    task_type: str | None
    classifier_version: str
    cost_usd: float = Field(description="Cost of the returned answer at DeepSeek peak prices.")
    latency_ms: float
    verification_status: str


class CompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str = Field(description="Provider model id that produced the answer.")
    choices: list[Choice]
    usage: Usage
    autopilot: RoutingMetadata


# ---------- models ----------


class ModelPrices(BaseModel):
    input_cache_miss_per_1m: float
    input_cache_hit_per_1m: float
    output_per_1m: float


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    model_id: str
    provider: str
    quality_tier: str
    serves_tiers: list[int] = Field(description="Routing tiers currently mapped to this model.")
    pricing_usd: ModelPrices
    typical_latency_ms: int
    thinking_enabled: bool | None


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
    pricing_basis: str
    pricing_now: str


# ---------- stats ----------


class StatsResponse(BaseModel):
    """Savings are always returned together with quality parity and verification overhead."""

    headline: str
    mode: str | None
    since_hours: int | None
    summary: HonestSummary


# ---------- routing config ----------


class RoutingConfigUpdate(BaseModel):
    """New tier → model mapping. Omitted optional fields keep their current values."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tiers": {"1": "local-llama", "2": "deepseek-flash", "3": "deepseek-v4-pro"},
                    "confidence_threshold": 0.55,
                    "reason": "route more borderline prompts up while Tier 1 quality is under review",
                }
            ]
        },
    )

    tiers: dict[str, str] = Field(description='Every tier "1", "2", "3" mapped to a model name from /v1/models.')
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    confidence_enabled: bool | None = None
    fallback_tier: Literal[1, 2, 3] | None = None
    reason: str | None = Field(default=None, max_length=500, description="Why — stored in the change log.")


class RoutingConfigView(BaseModel):
    tiers: dict[int, str]
    confidence_enabled: bool
    confidence_threshold: float
    fallback_tier: int
    classifier_version: str


class RoutingConfigResponse(BaseModel):
    config: RoutingConfigView
    source: Literal["routing.yaml", "runtime"]
    change_id: int | None = None
    changes_logged: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    classifier_version: str
    mode: str
    escalation_mode: str
    queue: dict[str, int]
