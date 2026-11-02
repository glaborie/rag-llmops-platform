"""Document loading and chunking."""

from ragllmops.ingest.chunking import chunk_document, chunk_documents, split_sections
from ragllmops.ingest.loader import load_documents, parse_document


def test_load_documents_reads_front_matter(corpus_dir):
    documents = load_documents(corpus_dir)
    assert len(documents) == 2
    by_id = {doc.doc_id: doc for doc in documents}
    assert by_id["compute"].title == "Compute Instances"
    assert by_id["compute"].category == "compute"
    assert "nw-standard" in by_id["compute"].body


def test_parse_document_rejects_missing_front_matter(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("# No front matter here", encoding="utf-8")
    try:
        parse_document(path)
    except ValueError as exc:
        assert "front-matter" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError")


def test_parse_document_rejects_missing_doc_id(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("---\ntitle: No Id\n---\n\nbody", encoding="utf-8")
    try:
        parse_document(path)
    except ValueError as exc:
        assert "doc_id" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError")


def test_split_sections_uses_headings():
    body = "preamble\n\n## One\nalpha\n\n## Two\nbeta"
    sections = split_sections(body)
    assert [s.heading for s in sections] == ["", "One", "Two"]
    assert sections[1].text == "alpha"


def test_chunk_document_produces_stable_ids(corpus_dir):
    document = load_documents(corpus_dir)[0]
    chunks = chunk_document(document)
    assert chunks
    assert [chunk.chunk_id for chunk in chunks] == [
        f"{document.doc_id}::{index}" for index in range(len(chunks))
    ]
    assert all(chunk.doc_id == document.doc_id for chunk in chunks)
    assert all(chunk.chunk_index == index for index, chunk in enumerate(chunks))


def test_chunk_document_keeps_heading_in_the_text(corpus_dir):
    document = load_documents(corpus_dir)[0]
    chunks = chunk_document(document)
    assert any("Instance families" in chunk.text for chunk in chunks)


def test_chunk_documents_preserves_document_order(corpus_dir):
    documents = load_documents(corpus_dir)
    chunks = chunk_documents(documents)
    doc_ids = [chunk.doc_id for chunk in chunks]
    assert doc_ids == sorted(doc_ids, key=lambda value: [d.doc_id for d in documents].index(value))


def test_long_sections_are_split_with_overlap(tmp_path):
    path = tmp_path / "long.md"
    path.write_text(
        "---\ndoc_id: long\ntitle: Long\ncategory: test\n---\n\n## Big\n"
        + ("Paragraph with some words in it. " * 80),
        encoding="utf-8",
    )
    document = parse_document(path)
    chunks = chunk_document(document, target_chars=400, max_chars=600, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 900 for chunk in chunks)
