"""Build the service and worker from the config files (one place, shared by scripts and the API)."""

from __future__ import annotations

import random

from autopilot.classifier.registry import load_classifier
from autopilot.llm.catalog import load_catalog
from autopilot.llm.client import LLMClient
from autopilot.routing.config import effective_routing_config, load_routing_config
from autopilot.routing.router import Router
from autopilot.service import AutopilotService
from autopilot.settings import AppSettings, load_settings
from autopilot.storage.db import Database
from autopilot.storage.repository import Repository
from autopilot.verification.rules import load_quality_config
from autopilot.verification.verifier import Verifier
from autopilot.verification.worker import Worker


def build_service(
    settings: AppSettings | None = None, *, client: LLMClient | None = None, rng: random.Random | None = None
) -> AutopilotService:
    settings = settings or load_settings()
    catalog = load_catalog()
    repo = Repository(Database(settings.database.resolved_path()))
    routing = effective_routing_config(load_routing_config(catalog), repo, catalog)
    router = Router(load_classifier(version=routing.classifier_version), routing, catalog)
    client = client or LLMClient(settings.client)
    verifier = Verifier(client, catalog, routing, load_quality_config(catalog), settings.verification, rng=rng)
    return AutopilotService(
        router=router,
        client=client,
        repo=repo,
        verifier=verifier,
        settings=settings.verification,
        rng=rng,
    )


def build_worker(settings: AppSettings | None = None, *, client: LLMClient | None = None) -> Worker:
    settings = settings or load_settings()
    catalog = load_catalog()
    repo = Repository(Database(settings.database.resolved_path()))
    base = load_routing_config(catalog)
    client = client or LLMClient(settings.client)
    verifier = Verifier(
        client, catalog, effective_routing_config(base, repo, catalog), load_quality_config(catalog),
        settings.verification,
    )
    return Worker(repo, verifier, settings.verification,
                  refresh_routing=lambda: effective_routing_config(base, repo, catalog))
