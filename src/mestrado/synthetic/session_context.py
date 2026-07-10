"""
Session Context Manager for multi-turn retrieval.

Port of Claude Code's compact.ts + findRelevantMemories.ts patterns
applied to legislative multi-turn conversations.

The core problem: turns 2+ often contain "nonstandalone" queries that
reference previous turns implicitly (e.g., "E quais dessas estão em tramitação?").
Without context, a retrieval system cannot resolve "dessas" (those).

This class maintains a compressed session context and rewrites
nonstandalone queries into standalone (resolvable) queries.

Reference: compact.ts (src/services/compact/compact.ts)
           findRelevantMemories.ts (src/memdir/findRelevantMemories.ts)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from mestrado.synthetic.generator import Turn


@dataclass
class SessionState:
    """Compressed representation of the session so far — analog to compact.ts state."""
    main_topic: str = ""
    entities_mentioned: list[str] = field(default_factory=list)  # bill names, topics
    constraints_accumulated: list[str] = field(default_factory=list)  # temporal, type filters
    turn_summaries: list[str] = field(default_factory=list)


class SessionContextManager:
    """
    Manages conversation context for multi-turn retrieval.

    Implements the compact.ts pattern from Claude Code:
    - Maintains compressed session state across turns
    - Rewrites nonstandalone queries to standalone form
    - Prevents context window explosion for long sessions

    Token budget: ~2000 tokens for full session history (like microCompact threshold).
    """

    COMPACT_SYSTEM = """Você é um assistente especializado em resumir sessões de busca legislativa.
Seja conciso e preserve apenas informações essenciais para o próximo turno de busca."""

    REWRITE_SYSTEM = """Você é um assistente especializado em reformular queries legislativas.
Converta queries que dependem de contexto anterior em queries independentes (standalone)."""

    def __init__(self, model: str = "llama3.2:3b", max_history_tokens: int = 2000,
                 provider: str = "ollama"):
        self.model = model
        self.max_history_tokens = max_history_tokens
        # Backend: "ollama" (local, default) or "deepinfra"/"openrouter" (API).
        # Additive — the Ollama path is unchanged.
        self.provider = provider
        self._api_client = None
        self._state: SessionState | None = None
        self._full_history: list[Turn] = []

    def _get_api_client(self):
        if self._api_client is None:
            import openai
            from mestrado import config
            if self.provider == "deepinfra":
                self._api_client = openai.OpenAI(
                    base_url="https://api.deepinfra.com/v1/openai", api_key=config.deepinfra.api_key)
            elif self.provider == "openrouter":
                self._api_client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=config.openrouter.api_key)
            else:
                raise ValueError(f"Unknown API provider: {self.provider!r}")
        return self._api_client

    def _rewrite_llm(self, prompt: str) -> str:
        """Provider-agnostic query rewrite (ollama local or OpenAI-compatible API)."""
        if self.provider == "ollama":
            import ollama
            response = ollama.generate(
                model=self.model, prompt=prompt, system=self.REWRITE_SYSTEM,
                options={"temperature": 0.1, "num_predict": 128})
            return response["response"]
        resp = self._get_api_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.REWRITE_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=128)
        return resp.choices[0].message.content or ""

    def reset(self):
        """Reset state for a new conversation."""
        self._state = SessionState()
        self._full_history = []

    def add_turn(self, turn: Turn):
        """Add a turn to the session history and update compressed state."""
        self._full_history.append(turn)
        self._update_state(turn)

    def _update_state(self, turn: Turn):
        """Update compressed session state after each turn — like compact.ts incremental update."""
        if self._state is None:
            self._state = SessionState(main_topic=turn.query)
            return

        # Incremental compact: just append a short summary, not full history
        summary = turn.query[:100] + ("..." if len(turn.query) > 100 else "")
        self._state.turn_summaries.append(f"T{turn.n} ({turn.type}): {summary}")

        # Extract temporal constraints
        temporal_keywords = ["2020", "2021", "2022", "2023", "2024", "recentes", "anteriores"]
        for kw in temporal_keywords:
            if kw in turn.query and kw not in self._state.constraints_accumulated:
                self._state.constraints_accumulated.append(kw)

    def get_compact_context(self) -> str:
        """
        Return compressed session context for query rewriting.
        Analogous to the compact.ts output injected into system prompt.
        """
        if not self._state:
            return ""

        parts = [f"Tema principal: {self._state.main_topic}"]

        if self._state.turn_summaries:
            parts.append("Histórico compactado:")
            parts.extend(f"  {s}" for s in self._state.turn_summaries[-4:])  # last 4 turns

        if self._state.constraints_accumulated:
            parts.append(f"Restrições acumuladas: {', '.join(self._state.constraints_accumulated)}")

        return "\n".join(parts)

    def resolve_query(self, query: str) -> str:
        """
        Rewrite a nonstandalone query into standalone form using session context.

        This is the core multi-turn operation — analogous to the sideQuery pattern
        in Claude Code where context is injected without polluting the main conversation.

        Example:
          Session context: "Propostas sobre transporte de equinos em rodovias federais"
          query: "E quais dessas estão em tramitação?"
          → resolved: "Quais propostas sobre transporte de equinos em rodovias federais
                        estão atualmente em tramitação?"
        """
        if not self._full_history:
            return query  # first turn, nothing to resolve

        # Heuristic: if query doesn't contain pronoun references, it's already standalone
        REFERENCE_PRONOUNS = ["esses", "essas", "elas", "eles", "esse", "essa",
                               "aqueles", "aquelas", "dessa", "dessas", "desse",
                               "nisso", "nelas", "neles", "delas", "deles"]

        query_lower = query.lower()
        has_reference = any(pron in query_lower.split() for pron in REFERENCE_PRONOUNS)

        if not has_reference and len(query.split()) > 5:
            return query  # likely already standalone

        compact_context = self.get_compact_context()

        prompt = f"""Contexto da sessão de pesquisa legislativa:
{compact_context}

Query atual (pode referenciar o contexto): {query}

Reescreva a query de forma independente (standalone) para que um sistema de busca
possa recuperar documentos sem conhecer o contexto anterior.
Mantenha a intenção original. Seja conciso.

Responda APENAS com a query reescrita, sem explicação:"""

        try:
            resolved = self._rewrite_llm(prompt).strip().strip('"')
            return resolved if resolved else query
        except Exception:
            return query  # fallback: use original query

    def get_augmented_query(self, turn: Turn) -> str:
        """
        Get query augmented with session context for dense retrieval.

        Different from resolve_query: instead of rewriting, concatenates
        a short context prefix. Useful for embedding-based retrieval.
        """
        if not self._full_history or turn.n == 1:
            return turn.query

        context_prefix = "; ".join(
            t.query[:50] for t in self._full_history[-2:]
        )
        return f"[contexto: {context_prefix}] {turn.query}"
