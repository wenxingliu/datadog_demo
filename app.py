from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st

from tutor.config import load_settings, require_pdf
from tutor.datadog_bootstrap import enable_llm_observability
from tutor.llm_client import answer_question, ensure_vector_store
from tutor.pdf_index import load_or_build_index, search_chunks
from tutor.prompts import MODES


st.set_page_config(page_title="Data Science Tutor", page_icon="DS", layout="wide")
enable_llm_observability()


@st.cache_resource(show_spinner=False)
def cached_index(pdf_path: str, index_path: str):
    return load_or_build_index(Path(pdf_path), Path(index_path))


def init_messages() -> None:
    if "datadog_session_id" not in st.session_state:
        st.session_state.datadog_session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about a concept from Principles of Data Science. I will decide whether to explain directly or guide you step by step.",
            }
        ]


settings = load_settings()
init_messages()

with st.sidebar:
    st.title("Tutor")
    mode = st.radio("Mode", list(MODES.keys()), index=0)
    top_k = st.slider("PDF context chunks", min_value=3, max_value=8, value=5)
    st.caption(f"Provider: `{settings.provider}`")
    st.caption(f"Model: `{settings.model}`")

st.title("Data Science PDF Tutor")
st.caption("Grounded in `Principles-of-Data-Science-WEB.pdf`")

try:
    require_pdf(settings.pdf_path)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if not settings.api_key:
    if settings.provider == "kimi":
        st.error("Set `KIMI_API_KEY` in `.env` before asking questions.")
    else:
        st.error("Set `OPENAI_API_KEY` in `.env` before asking questions.")
    st.stop()

with st.spinner("Preparing PDF index..."):
    chunks = cached_index(str(settings.pdf_path), str(settings.index_path))

if not chunks:
    st.error("No text could be extracted from the PDF.")
    st.stop()

if settings.provider != "openai":
    st.session_state.vector_store_id = None
elif "vector_store_id" not in st.session_state:
    try:
        with st.spinner("Preparing OpenAI file search..."):
            st.session_state.vector_store_id = ensure_vector_store(
                settings.api_key,
                settings.pdf_path,
                settings.state_path,
            )
    except Exception as exc:
        st.warning(f"OpenAI file search is unavailable, using local PDF context only. Details: {exc}")
        st.session_state.vector_store_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a data science question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = search_chunks(question, chunks, limit=top_k)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(
                    api_key=settings.api_key,
                    model=settings.model,
                    provider=settings.provider,
                    base_url=settings.base_url,
                    question=question,
                    mode=mode,
                    chunks=retrieved,
                    vector_store_id=st.session_state.vector_store_id,
                    top_k=top_k,
                    datadog_session_id=st.session_state.datadog_session_id,
                )
            except Exception as exc:
                answer = f"I could not call the model. Details: `{exc}`"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
