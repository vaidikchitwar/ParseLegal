"""Corrective RAG (C-RAG) Graph Workflow with Cyclic Self-Correction."""

from collections.abc import Callable

from engine.evaluator import DocumentGrader, HallucinationGrader
from engine.retriever import ClauseRetriever
from engine.state import CRAGState


class CRAGWorkflowEngine:
    """Cyclic Self-Correcting LangGraph-style workflow for zero-hallucination legal Q&A."""

    def __init__(
        self,
        retriever: ClauseRetriever,
        document_grader: DocumentGrader | None = None,
        hallucination_grader: HallucinationGrader | None = None,
        llm_generator: Callable[[str, list], str] | None = None,
    ):
        self.retriever = retriever
        self.document_grader = document_grader or DocumentGrader()
        self.hallucination_grader = hallucination_grader or HallucinationGrader()
        self.llm_generator = llm_generator or self._default_generator

    def _default_generator(self, query: str, docs: list) -> str:
        """Deterministic generator used when external LLM provider is offline."""
        citations = [
            f"[Page {d.page}: Clause {d.clause_title or 'General'}] {d.content.strip()}"
            for d in docs
        ]
        return (
            f"Based on verified contractual review under Indian Jurisprudence for query: '{query}':\n\n"
            + "\n".join(f"- {c}" for c in citations)
            + "\n\nLegal Note: In compliance with the Indian Contract Act, 1872, the aforementioned clauses govern party liabilities."
        )

    def retrieve_node(self, state: CRAGState) -> CRAGState:
        """Node 1: Retrieve candidate clauses from the index."""
        state.log(f"NODE: Retrieve [Query: '{state.query}']")
        docs = self.retriever.retrieve(state.query, top_k=3)
        state.documents = docs
        state.log(f"  -> Retrieved {len(docs)} candidate clause chunks.")
        return state

    def grade_documents_node(self, state: CRAGState) -> CRAGState:
        """Node 2: Grade relevance of candidate documents."""
        state.log("NODE: Grade Documents")
        relevant = []
        for d in state.documents:
            is_rel, score = self.document_grader.grade(state.query, d)
            if is_rel:
                relevant.append(d)
                state.log(f"  -> Clause (Score {score:.2f}): RELEVANT")
            else:
                state.log(f"  -> Clause (Score {score:.2f}): IRRELEVANT (Filtered)")

        state.relevant_documents = relevant
        state.needs_rewrite = len(relevant) == 0
        return state

    def rewrite_query_node(self, state: CRAGState) -> CRAGState:
        """Node 3: Cyclic Self-Correction: Expand and rewrite search query."""
        state.retry_count += 1
        state.log(
            f"NODE: Rewrite Query [Cycle {state.retry_count}/{state.max_retries}]"
        )

        # Domain-specific legal expansion
        legal_expansions = {
            "evict": "eviction notice default non-payment termination lease",
            "deposit": "security deposit refund deductions interest maintenance",
            "terminate": "termination notice period breach indemnity early exit",
            "rent": "monthly rent due date late fee grace period escalation",
        }

        expanded = state.query
        for key, expansion in legal_expansions.items():
            if key in state.query.lower():
                expanded = f"{state.query} {expansion}"
                break
        else:
            expanded = f"{state.query} legal obligations liability breach agreement"

        state.query = expanded
        state.log(f"  -> Reformulated Query: '{state.query}'")
        return state

    def generate_node(self, state: CRAGState) -> CRAGState:
        """Node 4: Synthesize grounded legal advice."""
        state.log("NODE: Generate Legal Analysis")
        docs_to_use = (
            state.relevant_documents if state.relevant_documents else state.documents
        )
        state.generation = self.llm_generator(state.query, docs_to_use)
        state.log(f"  -> Generated {len(state.generation.split())} words of analysis.")
        return state

    def groundedness_check_node(self, state: CRAGState) -> CRAGState:
        """Node 5: Hallucination Guardrail verifying citations match source."""
        state.log("NODE: Groundedness & Hallucination Check")
        docs_to_use = (
            state.relevant_documents if state.relevant_documents else state.documents
        )
        is_grounded = self.hallucination_grader.grade(state.generation, docs_to_use)
        state.is_grounded = is_grounded
        status_str = (
            "PASSED (Zero Hallucination)"
            if is_grounded
            else "FAILED (Ungrounded Assertions)"
        )
        state.log(f"  -> Groundedness Verification: {status_str}")
        return state

    def run(self, query: str) -> CRAGState:
        """Executes the cyclic state graph until convergence or max retries."""
        state = CRAGState(query=query, original_query=query)
        state.log("START: Initiating C-RAG Legal Pipeline")

        while True:
            # Step 1: Retrieval
            state = self.retrieve_node(state)

            # Step 2: Relevance Grading
            state = self.grade_documents_node(state)

            # Cyclic Router 1: If no relevant docs and retries available -> Rewrite & Re-retrieve
            if state.needs_rewrite and state.retry_count < state.max_retries:
                state.log(
                    "ROUTER: Irrelevant context detected. Triggering cyclic query rewrite."
                )
                state = self.rewrite_query_node(state)
                continue

            # Step 3: Generation
            state = self.generate_node(state)

            # Step 4: Groundedness Check
            state = self.groundedness_check_node(state)

            # Cyclic Router 2: If hallucinated and retries available -> Rewrite & Re-retrieve
            if not state.is_grounded and state.retry_count < state.max_retries:
                state.log(
                    "ROUTER: Hallucination detected! Routing back to query reformulator."
                )
                state = self.rewrite_query_node(state)
                continue

            state.log("END: C-RAG Workflow Complete")
            break

        return state
