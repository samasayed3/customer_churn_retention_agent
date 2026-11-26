"""Standalone tests for the vector DB tool (search_policies). Run from the repo root:
    python -m vector_db.build_vectordb       # once, before the tests
    python -m pytest tests/test_vector.py -v
"""
import json
import re

import pytest

from config import CHROMA_STORE_DIR, POLICIES_DIR
from vector_db import search
from vector_db.build_vectordb import chunk_documents, load_documents, load_topics
from vector_db.settings import CHUNK_SIZE
from vector_db.search import search_policies

ERROR_TYPES = {"validation_error", "not_found", "connection_error", "internal_error"}
POLICY_FILES = {f.name for f in POLICIES_DIR.glob("*.txt")}

needs_store = pytest.mark.skipif(
    not CHROMA_STORE_DIR.exists(),
    reason="vector store not built: run python -m vector_db.build_vectordb",
)


def assert_error(res, err_type=None):
    """Every error must follow the Tool Contract."""
    assert res["status"] == "error"
    assert res["tool"] == "search_policies"
    assert res["data"] is None
    assert isinstance(res["error"], dict)
    assert res["error"]["type"] in ERROR_TYPES
    assert res["error"]["message"]
    assert isinstance(res["error"]["missing_fields"], list)
    if err_type:
        assert res["error"]["type"] == err_type
    json.dumps(res)  # must be JSON-serialisable


# ----------------------- chunking (no model needed) ----------------------- #
@pytest.fixture(scope="module")
def chunks():
    ids, texts, metas = chunk_documents(load_documents(), load_topics())
    return ids, texts, metas


def test_chunk_ids_are_unique(chunks):
    ids, _, _ = chunks
    assert len(ids) == len(set(ids))


def test_chunks_respect_size(chunks):
    _, texts, _ = chunks
    assert max(len(t) for t in texts) <= CHUNK_SIZE


def test_no_chunk_is_only_a_heading(chunks):
    _, texts, _ = chunks
    heading_only = [t for t in texts if "\n" not in t and len(t) <= 80
                    and not t.endswith((".", "?"))]
    assert heading_only == []


def test_metadata_has_source_file_chunk_id_topic(chunks):
    ids, _, metas = chunks
    for cid, m in zip(ids, metas):
        assert set(m) == {"source_file", "chunk_id", "topic"}
        assert m["chunk_id"] == cid
        assert m["source_file"] in POLICY_FILES
        assert m["topic"]


def test_no_sentence_is_cut_between_chunks(chunks):
    """Every sentence of every policy must appear whole inside some chunk."""
    _, texts, _ = chunks
    blob = re.sub(r"\s+", " ", " ".join(texts))
    for f in POLICIES_DIR.glob("*.txt"):
        text = re.sub(r"\s+", " ", f.read_text(encoding="utf-8"))
        for sent in re.split(r"(?<=[.?])\s+", text):
            if len(sent) > 25:
                assert sent in blob, f"cut sentence in {f.name}: {sent[:70]}"


# ------------------------ success + contract ------------------------ #
@needs_store
def test_success_follows_contract():
    res = search_policies("How long is the notice period?", top_k=3)
    assert res["status"] == "ok"
    assert res["tool"] == "search_policies"
    assert res["error"] is None
    assert res["data"]["query"] == "How long is the notice period?"
    assert len(res["data"]["results"]) == 3
    json.dumps(res)


@needs_store
def test_result_fields_are_valid():
    results = search_policies("What is a High Spender?", top_k=5)["data"]["results"]
    for r in results:
        assert r["text"].strip()
        assert r["source_file"] in POLICY_FILES
        assert r["chunk_id"].startswith(r["source_file"].removesuffix(".txt"))
        assert r["topic"]
        assert 0 <= r["distance"] <= 2          # cosine distance
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)       # closest first


@needs_store
@pytest.mark.parametrize("top_k", [1, 2, 5, 10])
def test_top_k_is_respected(top_k):
    res = search_policies("retention offers", top_k=top_k)
    assert len(res["data"]["results"]) == top_k


# --------------------- retrieval quality (the point) ---------------------- #
# (question, phrase that must appear in one of the top-3 passages)
QUESTIONS = [
    ("How much is the early termination fee for a One year contract?", "100 USD"),
    ("How much is the early termination fee for a Two year contract?", "200 USD"),
    ("Which offer is for long-term customers?", "Loyalty Reward"),
    ("Who can get the Welcome Bonus?", "12 months or less"),
    ("Which payment methods count as manual?", "Electronic check or Mailed check"),
    ("What does a customer need for the Fiber Price Lock?", "Monthly Charges At Least 80"),
    ("Who is eligible for the Senior Care Package?", "SeniorCitizen equals 1"),
]


@needs_store
@pytest.mark.parametrize("question,phrase", QUESTIONS)
def test_finds_the_right_passage(question, phrase):
    results = search_policies(question, top_k=3)["data"]["results"]
    assert any(phrase in r["text"] for r in results), \
        f"{phrase!r} not in top-3 for: {question}"


@needs_store
def test_source_is_a_file_that_really_has_the_answer():
    # the One year fee is stated in both of these policies
    top = search_policies(
        "How much is the early termination fee for a One year contract?", top_k=1
    )["data"]["results"][0]
    assert top["source_file"] in {"contract_policy.txt", "cancellation_policy.txt"}


# --------------------------- validation errors --------------------------- #
@pytest.mark.parametrize("bad_query", ["", "   ", None, 123, ["a"]])
def test_bad_query_is_validation_error(bad_query):
    res = search_policies(bad_query)
    assert_error(res, "validation_error")
    assert res["error"]["missing_fields"] == ["query"]


@pytest.mark.parametrize("bad_k", [0, -1, 11, 3.5, "3", True, None])
def test_bad_top_k_is_validation_error(bad_k):
    assert_error(search_policies("notice period", top_k=bad_k), "validation_error")


# ------------------- tools never raise (error paths) -------------------- #
def test_missing_store_is_connection_error(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_model", object())      # skip model download
    monkeypatch.setattr(search, "_collection", None)
    monkeypatch.setattr(search, "CHROMA_STORE_DIR", tmp_path / "does_not_exist")
    assert_error(search_policies("notice period"), "connection_error")


def test_unexpected_failure_is_internal_error(monkeypatch):
    def boom():
        raise RuntimeError("something broke")
    monkeypatch.setattr(search, "_load", boom)
    res = search_policies("notice period")
    assert_error(res, "internal_error")
    assert "something broke" in res["error"]["message"]
