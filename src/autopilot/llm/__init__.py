"""Unified model interface: one ``send_request`` for every provider and tier."""

from autopilot.llm.catalog import ModelCatalog, load_catalog
from autopilot.llm.client import LLMClient, LLMRequestError, send_request
from autopilot.llm.schemas import ChatMessage, LLMResponse, MissingAPIKeyError, ModelConfig

__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMRequestError",
    "LLMResponse",
    "MissingAPIKeyError",
    "ModelCatalog",
    "ModelConfig",
    "load_catalog",
    "send_request",
]
