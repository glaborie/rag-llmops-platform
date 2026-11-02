"""Prompts, the answering chain and the extractive provider."""

from ragllmops.llm.extractive import REFUSAL_TEXT, ExtractiveLLM
from ragllmops.rag.prompts import DEFAULT_PROMPT_ID, get_prompt, is_refusal, list_prompts


def test_prompt_registry_contains_both_versions():
    assert list_prompts() == ["v1", "v2"]
    assert DEFAULT_PROMPT_ID == "v2"


def test_get_prompt_renders_question_and_context():
    system, user = get_prompt("v2").render(question="What is k?", context="[S1] k is 2")
    assert "What is k?" in user
    assert "[S1] k is 2" in user
    assert "cite" in system.lower()


def test_get_prompt_rejects_unknown_ids():
    try:
        get_prompt("v99")
    except KeyError as exc:
        assert "unknown prompt_id" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a KeyError")


def test_is_refusal_detects_the_canonical_sentence():
    assert is_refusal(REFUSAL_TEXT)
    assert not is_refusal("The limit is 600 requests per minute [S1]")


def test_extractive_provider_quotes_the_context():
    provider = ExtractiveLLM()
    result = provider.complete(
        system="s",
        user="Question: what is the cooldown\n\nContext: [S1] the cooldown is 120 seconds",
        context=["the cooldown is 120 seconds"],
    )
    assert "120 seconds" in result.text
    assert "[S1]" in result.text
    assert result.provider == "extractive"


def test_extractive_provider_refuses_without_context():
    result = ExtractiveLLM().complete(system="s", user="Question: what is the cooldown")
    assert result.text == REFUSAL_TEXT


def test_extractive_provider_refuses_when_nothing_matches():
    result = ExtractiveLLM().complete(
        system="s",
        user="Question: what is the cooldown",
        context=["completely unrelated prose about baking bread"],
    )
    assert result.text == REFUSAL_TEXT


def test_chain_answers_and_cites(platform):
    bundle = platform.current_bundle()
    result = platform.answer("What CPU target does the autoscaler use?", bundle=bundle)
    response = result.response

    assert response.answer
    assert response.citations
    assert response.retrieved_count == len(response.citations)
    assert response.bundle_version == bundle.version
    assert "compute" in {citation.doc_id for citation in response.citations}


def test_chain_logs_the_request(platform):
    platform.answer("What CPU target does the autoscaler use?")
    records = platform.log_store.fetch()
    assert len(records) == 1
    assert records[0].query.startswith("What CPU target")
    assert records[0].retrieved_ids


def test_chain_respects_the_context_budget(platform):
    bundle = platform.current_bundle()
    from dataclasses import replace

    tight = replace(bundle, max_context_chars=120)
    result = platform.answer("What CPU target does the autoscaler use?", bundle=tight)
    assert len(result.response.citations) <= 1
