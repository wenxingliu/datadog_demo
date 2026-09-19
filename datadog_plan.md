# Datadog LLM Observability Implementation Plan

## Summary

Instrument the Streamlit tutor for Datadog LLM Observability. The app uses Datadog's Python `ddtrace` SDK with agentless export, so no local Datadog Agent and no `ddtrace-run` wrapper are required.

Primary success criteria: each answered question creates a Datadog LLM span named `tutor.answer_question` with service/app tags, latency, errors, token metrics when available, Datadog session grouping, input/output capture when enabled, and safe app metadata.

## Changes to be Implemented

- Add `ddtrace` to `requirements.txt`.
- Add Datadog local-dev variables to `.env.example` and `.env copy.example`.
- Load `.env` and enable Datadog LLM Observability in `tutor/datadog_bootstrap.py`.
- Disable Datadog auto integrations and create a manual Datadog LLM span around `answer_question()` in `tutor/observability.py`, so local Streamlit runs emit LLM traces without trying to send APM traces to `localhost:8126`.
- Generate one Datadog session id per Streamlit browser session and attach it to each LLM span from that session.
- Pass `top_k` from `app.py` into `answer_question()` so retrieval settings can be attached to the LLM span.

## Required Environment

```bash
LLM_PROVIDER=kimi
KIMI_API_KEY=<your-kimi-api-key>
KIMI_MODEL=kimi-k2.6
KIMI_BASE_URL=https://api.moonshot.ai/v1

# Optional if you switch LLM_PROVIDER=openai.
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-5-mini

DD_LLMOBS_ENABLED=1
DD_LLMOBS_AGENTLESS_ENABLED=1
DD_LLMOBS_ML_APP=<app-name>
DD_SERVICE=<service-name>
DD_ENV=local
DD_APM_TRACING_ENABLED=false
DD_OPENAI_SPAN_PROMPT_COMPLETION_SAMPLE_RATE=0
DD_OPENAI_LOGS_ENABLED=false
DD_LLMOBS_CAPTURE_IO=1
DD_SITE=<your-datadog-site>
DD_API_KEY=<your-datadog-api-key>
```

## Span Metadata

- Manual LLM span name: `tutor.answer_question`.
- Session id: generated UUID stored in `st.session_state.datadog_session_id`.
- Metadata: `mode`, `retrieved_chunk_count`, `top_k`, `vector_store_enabled`.
- Input/output capture: when `DD_LLMOBS_CAPTURE_IO=1`, span input is the user question and span output is the model answer.
- Tags: `component:data_science_tutor`, `tutor.mode`, `tutor.vector_store_enabled`, `tutor.retrieved_chunk_count`, `tutor.top_k`.
- Metrics: `input_tokens`, `output_tokens`, and `total_tokens` when present on the OpenAI response.

## Test Plan

```bash
env/bin/python -m pip install -r requirements.txt
env/bin/python -m pytest
env/bin/python -m compileall app.py tutor
```

Launch locally with `env/bin/streamlit run app.py`, ask one tutor question, and verify a `tutor.answer_question` span appears in Datadog LLM Observability under `ml_app:data-science-tutor` and `env:local`.