"""
Multi-Turn Retriever for Ulysses-MTRAG.

Wraps BM25/Dense/Hybrid retrievers with conversation-aware query resolution.
Mirrors the 3 MTRAG retrieval variants:
  - lastturn:  only the last user turn (weakest, baseline)
  - concat:    concatenation of last 2 turns
  - rewrite:   LLM rewrites nonstandalone query to standalone (best expected)
  - questions: full conversation history as context prefix

Reference architecture from MTRAG (Katsis et al., TACL 2025):
  mtrag-human/retrieval_tasks/ has govt_lastturn.jsonl, govt_questions.jsonl, govt_rewrite.jsonl
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever
from mestrado.retrieval.hybrid import HybridRetriever


ContextStrategy = Literal["lastturn", "concat", "rewrite", "questions"]


@dataclass
class TurnContext:
    """Minimal context needed to represent a prior turn."""
    n: int
    query: str
    turn_type: str = "unknown"


class MultiTurnRetriever:
    """
    Wraps any single-turn retriever with conversation-aware query augmentation.

    Strategies (mirrors MTRAG variants):
      lastturn  → use only current query, no history (weakest)
      concat    → prepend last 2 queries to current (simple baseline)
      rewrite   → LLM rewrites nonstandalone to standalone (best expected)
      questions → full history formatted as: T1: q1 T2: q2 ... Tn: qn

    The rewrite strategy uses SessionContextManager (compact.ts port).
    """

    def __init__(
        self,
        base_retriever: Union[BM25Retriever, DenseRetriever, HybridRetriever],
        strategy: ContextStrategy = "rewrite",
        rewrite_model: str = "llama3.2:3b",
        max_history_turns: int = 4,
        provider: str = "ollama",
    ):
        self.base_retriever = base_retriever
        self.strategy = strategy
        self.max_history_turns = max_history_turns
        self.provider = provider
        self._ctx_manager = None

        if strategy == "rewrite":
            from mestrado.synthetic.session_context import SessionContextManager
            self._ctx_manager = SessionContextManager(model=rewrite_model, provider=provider)

    def retrieve(
        self,
        current_query: str,
        history: list[TurnContext],
        k: int = 12,
    ) -> list[Bill]:
        """
        Retrieve top-k bills considering conversation history.

        Args:
            current_query: The current turn's query text
            history: Prior turns (ordered, oldest first)
            k: Number of results to return

        Returns:
            List of Bill objects ranked by relevance
        """
        augmented_query = self._augment_query(current_query, history)
        return self.base_retriever.retrieve(augmented_query, k)

    def _augment_query(self, current_query: str, history: list[TurnContext]) -> str:
        """Apply the selected context strategy to augment the query."""

        if not history or self.strategy == "lastturn":
            return current_query

        recent = history[-self.max_history_turns:]

        if self.strategy == "concat":
            # Simple: concatenate last 2 prior queries + current
            prior_texts = " ".join(t.query[:80] for t in recent[-2:])
            return f"{prior_texts} {current_query}".strip()

        elif self.strategy == "questions":
            # Full history formatted as: |user|: "q1" |user|: "q2" ... |user|: "qn"
            # Matches MTRAG's govt_questions.jsonl format
            history_formatted = " ".join(
                f'|user|: "{t.query}"' for t in recent
            )
            return f"{history_formatted} |user|: \"{current_query}\""

        elif self.strategy == "rewrite":
            # LLM rewrites using session context — best strategy
            if self._ctx_manager is not None:
                # Sync the context manager with history
                from mestrado.synthetic.generator import Turn as GenTurn
                self._ctx_manager.reset()
                for t in recent:
                    self._ctx_manager.add_turn(
                        GenTurn(n=t.n, type=t.turn_type, query=t.query)
                    )
                return self._ctx_manager.resolve_query(current_query)
            return current_query  # fallback if ctx_manager not initialized

        return current_query

    def retrieve_with_all_strategies(
        self,
        current_query: str,
        history: list[TurnContext],
        k: int = 12,
    ) -> dict[str, list[Bill]]:
        """
        Run retrieval with all 4 strategies simultaneously.
        Useful for ablation study comparing strategies.

        Note: "rewrite" uses a fresh MultiTurnRetriever instance to avoid
        state contamination from the shared _ctx_manager.
        """
        results = {}

        # lastturn / concat / questions: safe to toggle self.strategy (stateless)
        for strat in ("lastturn", "concat", "questions"):
            original_strat = self.strategy
            self.strategy = strat
            results[strat] = self.retrieve(current_query, history, k)
            self.strategy = original_strat

        # rewrite: isolated instance — prevents accumulated context from
        # prior strategy calls contaminating the standalone query resolution
        rewrite_model = (
            getattr(self._ctx_manager, "model", "llama3.2:3b")
            if self._ctx_manager else "llama3.2:3b"
        )
        temp = MultiTurnRetriever(
            self.base_retriever,
            strategy="rewrite",
            rewrite_model=rewrite_model,
            provider=self.provider,
        )
        results["rewrite"] = temp.retrieve(current_query, history, k)

        return results


def build_multi_turn_retriever(
    bills: list[Bill],
    strategy: ContextStrategy = "rewrite",
    retriever_type: str = "hybrid",
    rewrite_model: str = "llama3.2:3b",
) -> MultiTurnRetriever:
    """
    Factory: build a fully indexed MultiTurnRetriever.

    Args:
        bills: List of Bill objects to index
        strategy: Context strategy (lastturn/concat/rewrite/questions)
        retriever_type: Base retriever (bm25/dense/hybrid)
        rewrite_model: Ollama model for query rewriting

    Returns:
        Ready-to-use MultiTurnRetriever
    """
    if retriever_type == "bm25":
        base = BM25Retriever()
        base.index(bills)
    elif retriever_type == "dense":
        base = DenseRetriever()
        base.index(bills)
    elif retriever_type == "hybrid":
        base = HybridRetriever.build(bills)
    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")

    return MultiTurnRetriever(
        base_retriever=base,
        strategy=strategy,
        rewrite_model=rewrite_model,
    )
