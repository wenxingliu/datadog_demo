from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _capture_io_enabled() -> bool:
    import os

    return os.getenv("DD_LLMOBS_CAPTURE_IO", "").lower() in {"1", "true", "yes", "on"}


def extract_token_metrics(usage: Any) -> dict[str, int]:
    """Normalize token usage from Chat Completions or Responses API objects."""
    if usage is None:
        return {}

    def get_int(*names: str) -> int | None:
        for name in names:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if isinstance(value, int):
                return value
        return None

    metrics: dict[str, int] = {}
    input_tokens = get_int("input_tokens", "prompt_tokens")
    output_tokens = get_int("output_tokens", "completion_tokens")
    total_tokens = get_int("total_tokens")
    if input_tokens is not None:
        metrics["input_tokens"] = input_tokens
    if output_tokens is not None:
        metrics["output_tokens"] = output_tokens
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    return metrics


@contextmanager
def llm_span(
    *,
    model: str,
    provider: str,
    question: str,
    mode: str,
    retrieved_chunk_count: int,
    top_k: int,
    vector_store_enabled: bool,
    session_id: str | None,
) -> Iterator[Any | None]:
    try:
        from ddtrace.llmobs import LLMObs
    except ImportError:
        yield None
        return

    if not LLMObs.enabled:
        yield None
        return

    metadata = {
        "mode": mode,
        "retrieved_chunk_count": retrieved_chunk_count,
        "top_k": top_k,
        "vector_store_enabled": vector_store_enabled,
    }
    tags = {
        "component": "data_science_tutor",
        "tutor.mode": mode,
        "tutor.vector_store_enabled": str(vector_store_enabled).lower(),
        "tutor.retrieved_chunk_count": str(retrieved_chunk_count),
        "tutor.top_k": str(top_k),
    }

    with LLMObs.llm(
        name="tutor.answer_question",
        model_name=model,
        model_provider=provider,
        session_id=session_id,
    ) as span:
        LLMObs.annotate(
            span=span,
            input_data=question if _capture_io_enabled() else None,
            metadata=metadata,
            tags=tags,
        )
        yield span


def annotate_llm_output(span: Any | None, *, answer: str, token_metrics: dict[str, int]) -> None:
    if span is None:
        return
    from ddtrace.llmobs import LLMObs

    LLMObs.annotate(
        span=span,
        output_data=answer if _capture_io_enabled() else None,
        metrics=token_metrics or None,
    )
