from types import SimpleNamespace

from tutor.observability import extract_token_metrics


def test_extract_token_metrics_from_chat_completions_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert extract_token_metrics(usage) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_extract_token_metrics_from_responses_usage_dict() -> None:
    usage = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}

    assert extract_token_metrics(usage) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }
