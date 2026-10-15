"""State schema for the Corrective RAG (C-RAG) self-correcting workflow."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a discrete legal document chunk with provenance metadata."""

    content: str
    page: int = 1
    score: float = 0.0
    clause_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CRAGState:
    """Execution state passed through the LangGraph cyclic reasoning pipeline."""

    query: str
    original_query: str = ""
    documents: list[DocumentChunk] = field(default_factory=list)
    relevant_documents: list[DocumentChunk] = field(default_factory=list)
    generation: str = ""
    retry_count: int = 0
    max_retries: int = 2
    is_grounded: bool = False
    needs_rewrite: bool = False
    trace: list[str] = field(default_factory=list)

    def log(self, step: str) -> None:
        """Appends an execution step to the diagnostic trace."""
        self.trace.append(step)
