"""Unit and integration tests for ParseLegal C-RAG engine."""

import pytest

from engine.evaluator import DocumentGrader, HallucinationGrader
from engine.graph import CRAGWorkflowEngine
from engine.retriever import ClauseRetriever
from engine.state import DocumentChunk


@pytest.fixture
def sample_corpus():
    return [
        DocumentChunk(
            content="Clause 14 (Termination): The Lessor may terminate this agreement upon 30 days written notice if rent is unpaid.",
            page=4,
            clause_title="Termination",
        ),
        DocumentChunk(
            content="Clause 8 (Security Deposit): The Lessee shall pay INR 50,000 as refundable security deposit subject to damage deductions.",
            page=2,
            clause_title="Security Deposit",
        ),
        DocumentChunk(
            content="Clause 22 (Indemnity): The Lessee shall indemnify the Lessor against third party claims arising from unauthorized subletting.",
            page=7,
            clause_title="Indemnity",
        ),
    ]


def test_document_grader_relevance(sample_corpus):
    grader = DocumentGrader(threshold=0.2)
    is_rel, score = grader.grade(
        "What are the conditions for termination?", sample_corpus[0]
    )
    assert is_rel is True
    assert score > 0.3

    is_rel, score = grader.grade(
        "What pets are allowed in the building?", sample_corpus[0]
    )
    assert is_rel is False


def test_hallucination_grader():
    grader = HallucinationGrader()
    docs = [
        DocumentChunk(
            content="Clause 14 specifies 30 days notice for lease termination and default penalties."
        )
    ]

    # Grounded answer with citations
    grounded_ans = (
        "According to Clause 14, termination requires 30 days notice following default."
    )
    assert grader.grade(grounded_ans, docs) is True

    # Hallucinated answer citing fabricated concepts without citations
    empty_docs = []
    assert grader.grade("Some ungrounded response", empty_docs) is False


def test_crag_happy_path(sample_corpus):
    retriever = ClauseRetriever(sample_corpus)
    engine = CRAGWorkflowEngine(retriever)

    state = engine.run("What are the termination notice conditions?")

    assert state.is_grounded is True
    assert len(state.relevant_documents) > 0
    assert "Termination" in state.generation
    assert state.retry_count == 0  # direct hit without needing rewrite
    assert any("START" in t for t in state.trace)
    assert any("PASSED" in t for t in state.trace)


def test_crag_cyclic_self_correction():
    # Corpus contains rent and eviction clauses
    corpus = [
        DocumentChunk(
            content="Clause 5 (Default): Persistent non-payment leads to eviction and lease cancellation.",
            page=3,
            clause_title="Default",
        )
    ]
    retriever = ClauseRetriever(corpus)
    engine = CRAGWorkflowEngine(retriever)

    # Initial query uses slang "kick out" which requires query rewrite
    state = engine.run("How to evict a tenant?")

    # Engine should have processed through rewrite or converged with grounding
    assert state.generation != ""
    assert state.is_grounded is True
    assert len(state.trace) > 4


def test_crag_max_retries_boundary():
    # Completely empty corpus
    retriever = ClauseRetriever([])
    engine = CRAGWorkflowEngine(retriever)

    state = engine.run("What is quantum computing?")
    # Should safely terminate after reaching max retries without unhandled exception
    assert state.retry_count <= state.max_retries
    assert "END" in state.trace[-1]
