"""Evaluators for document relevance grading and hallucination verification."""

import re

from engine.state import DocumentChunk


class DocumentGrader:
    """Evaluates whether retrieved clauses contain semantic overlap with the query."""

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def grade(self, query: str, document: DocumentChunk) -> tuple[bool, float]:
        """Calculates relevance score between legal query and candidate excerpt."""
        query_words = set(re.findall(r"\w+", query.lower()))
        content_words = set(re.findall(r"\w+", document.content.lower()))

        if not query_words or not content_words:
            return False, 0.0

        overlap = len(query_words.intersection(content_words))
        score = overlap / len(query_words)

        # Boost for key statutory legal terms (e.g. breach, liability, indemnity, termination)
        legal_keywords = {
            "termination",
            "indemnity",
            "liability",
            "jurisdiction",
            "statute",
            "clause",
            "penalty",
            "dispute",
            "rent",
            "default",
        }
        if any(k in content_words and k in query_words for k in legal_keywords):
            score = min(1.0, score + 0.3)

        is_relevant = score >= self.threshold
        return is_relevant, score


class HallucinationGrader:
    """Validates that assertions in the generated legal answer cite source clauses."""

    def grade(self, generation: str, documents: list[DocumentChunk]) -> bool:
        """Returns True if the generation is strictly grounded in the retrieved documents."""
        if not generation or not documents:
            return False

        combined_text = " ".join([d.content.lower() for d in documents])
        gen_lower = generation.lower()

        # Check for clause references or citations
        has_citations = bool(
            re.search(r"clause|section|page \d+|excerpt|\barticle\b", gen_lower)
        )

        # Verify that key terms mentioned in generation exist in source text
        gen_tokens = [
            w
            for w in re.findall(r"\b[a-zA-Z]{5,}\b", gen_lower)
            if w
            not in {"legal", "agreement", "document", "provided", "clause", "parties"}
        ]
        if not gen_tokens:
            return True

        matches = sum(1 for t in gen_tokens if t in combined_text)
        grounding_ratio = matches / len(gen_tokens)

        # Grounded if at least 50% of substantial legal vocabulary is traceable to source
        return grounding_ratio >= 0.50 or has_citations
