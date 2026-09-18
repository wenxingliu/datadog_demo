from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import file_sha256
from .observability import annotate_llm_output, extract_token_metrics, llm_span
from .pdf_index import Chunk, format_context
from .prompts import system_prompt, user_prompt


def _make_client(api_key: str, base_url: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `pip install -r requirements.txt`.") from exc
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=True, indent=2)


def ensure_vector_store(api_key: str, pdf_path: Path, state_path: Path) -> str:
    pdf_hash = file_sha256(pdf_path)
    state = _load_state(state_path)
    if state.get("pdf_sha256") == pdf_hash and state.get("vector_store_id"):
        return str(state["vector_store_id"])

    client = _make_client(api_key)
    vector_store = client.vector_stores.create(name="Principles of Data Science Tutor")
    with pdf_path.open("rb") as pdf_file:
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[pdf_file],
        )

    if getattr(file_batch, "status", None) not in {None, "completed"}:
        for _ in range(30):
            time.sleep(1)
            file_batch = client.vector_stores.file_batches.retrieve(
                file_batch.id,
                vector_store_id=vector_store.id,
            )
            if getattr(file_batch, "status", None) == "completed":
                break
        else:
            raise RuntimeError(f"Vector store indexing did not complete: {file_batch.status}")

    _save_state(
        state_path,
        {
            "pdf_sha256": pdf_hash,
            "vector_store_id": vector_store.id,
            "file_batch_id": getattr(file_batch, "id", None),
        },
    )
    return vector_store.id


def answer_question(
    *,
    api_key: str,
    model: str,
    provider: str,
    base_url: str | None,
    question: str,
    mode: str,
    chunks: list[Chunk],
    vector_store_id: str | None,
    top_k: int,
    datadog_session_id: str | None = None,
) -> str:
    with llm_span(
        model=model,
        provider=provider,
        question=question,
        mode=mode,
        retrieved_chunk_count=len(chunks),
        top_k=top_k,
        vector_store_enabled=bool(vector_store_id),
        session_id=datadog_session_id,
    ) as span:
        client = _make_client(api_key, base_url)
        local_context = format_context(chunks)
        if provider == "kimi":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt(mode)},
                    {"role": "user", "content": user_prompt(question, local_context)},
                ],
            )
            content = response.choices[0].message.content
            answer = content or ""
            annotate_llm_output(
                span,
                answer=answer,
                token_metrics=extract_token_metrics(getattr(response, "usage", None)),
            )
            return answer

        tools = []
        if vector_store_id:
            tools.append({"type": "file_search", "vector_store_ids": [vector_store_id]})

        create_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt(mode),
            "input": user_prompt(question, local_context),
        }
        if tools:
            create_kwargs["tools"] = tools

        response = client.responses.create(**create_kwargs)
        answer = response.output_text
        annotate_llm_output(
            span,
            answer=answer,
            token_metrics=extract_token_metrics(getattr(response, "usage", None)),
        )
        return answer
