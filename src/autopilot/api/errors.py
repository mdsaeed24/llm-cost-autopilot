"""One error format for everything: ``{"error": {"type", "message", "details"}}``."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from autopilot.llm.client import LLMRequestError
from autopilot.llm.schemas import MissingAPIKeyError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Raise from a route to return a typed error."""

    def __init__(self, status_code: int, error_type: str, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.details = details


def error_body(error_type: str, message: str, details: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"type": error_type, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "problem": err["msg"].removeprefix("Value error, ")}
        for err in exc.errors()
    ]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.error_type, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = _validation_details(exc)
        message = "; ".join(f"{d['field']}: {d['problem']}" if d['field'] else d['problem'] for d in details)
        return JSONResponse(error_body("invalid_request", message, details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(error_body(error_type, str(exc.detail)), status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(LLMRequestError)
    async def _upstream(_: Request, exc: LLMRequestError) -> JSONResponse:
        return JSONResponse(
            error_body("upstream_error", f"All model tiers failed to answer: {exc.cause.__class__.__name__}"),
            status_code=502,
        )

    @app.exception_handler(MissingAPIKeyError)
    async def _missing_key(_: Request, exc: MissingAPIKeyError) -> JSONResponse:
        return JSONResponse(error_body("provider_not_configured", str(exc)), status_code=503)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")  # full trace in the logs, never in the response
        return JSONResponse(error_body("internal_error", "Unexpected server error"), status_code=500)
