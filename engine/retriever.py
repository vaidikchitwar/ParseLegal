"""Retriever abstraction supporting Chroma and in-memory test fixtures."""


from engine.state import DocumentChunk


class ClauseRetriever:
    """Retrieves relevant contractual clauses matching the search query."""

    def __init__(self, corpus: list[DocumentChunk] | None = None):
        self.corpus = corpus or []

    def retrieve(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        """Simple rank-based retrieval across indexed corpus."""
        query_terms = set(query.lower().split())
        scored: list[tuple] = []
        for doc in self.corpus:
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0] or self.corpus[
            :top_k
        ]
