from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .config import ROOT_DIR


logger = logging.getLogger(__name__)
_ENABLED = False


def _is_enabled(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _is_placeholder(value: str | None) -> bool:
    return value is None or value.startswith("<")


def enable_llm_observability() -> bool:
    """Enable Datadog LLM Observability once for local Streamlit runs."""
    global _ENABLED
    if _ENABLED:
        return True

    load_dotenv(ROOT_DIR / ".env")
    if not _is_enabled(os.getenv("DD_LLMOBS_ENABLED")):
        return False

    site = os.getenv("DD_SITE")
    api_key = os.getenv("DD_API_KEY")
    agentless_enabled = _is_enabled(os.getenv("DD_LLMOBS_AGENTLESS_ENABLED"))
    if agentless_enabled and (_is_placeholder(site) or _is_placeholder(api_key)):
        logger.warning("Datadog LLMObs is enabled but DD_SITE or DD_API_KEY is missing.")
        return False

    try:
        from ddtrace.llmobs import LLMObs
    except ImportError:
        logger.warning("Datadog LLMObs requested but ddtrace is not installed.")
        return False

    try:
        LLMObs.enable(
            agent_service=os.getenv("DD_LLMOBS_ML_APP") or os.getenv("DD_SERVICE"),
            integrations_enabled=False,
            agentless_enabled=agentless_enabled,
            site=site,
            api_key=api_key,
            env=os.getenv("DD_ENV"),
            service=os.getenv("DD_SERVICE"),
        )
    except Exception:
        logger.exception("Failed to enable Datadog LLM Observability.")
        return False

    _ENABLED = True
    return True
