"""Text utilities."""

from ragllmops.text import content_terms, normalize, sentence_split, stem, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_tokenize_splits_identifiers_on_their_separators():
    assert tokenize("Set max-poll-interval") == ["set", "max", "poll", "interval"]
    assert tokenize("tag it with cost_center") == ["tag", "it", "with", "cost", "center"]
    assert tokenize("max.poll.interval") == ["max", "poll", "interval"]


def test_normalize_collapses_whitespace():
    assert normalize("  a   b \n c ") == "a b c"


def test_content_terms_drops_stopwords_and_stems():
    terms = content_terms("The autoscaling policies are applied to instances")
    assert "the" not in terms
    assert "are" not in terms
    assert "autoscal" in terms


def test_stem_handles_plural_and_ing():
    assert stem("instances") == "instance"
    assert stem("policies") == "policy"
    assert stem("monitoring") == "monitor"
    assert stem("class") == "class"


def test_sentence_split_keeps_punctuation():
    text = "The limit is 600 requests. Rate limits apply per tenant."
    assert sentence_split(text) == ["The limit is 600 requests.", "Rate limits apply per tenant."]
