"""
Ulysses-MTRAG Synthetic Dataset Generator.

Generates multi-turn legislative research conversations from single-turn
Ulysses-RFCorpus queries (Vitório et al., LRE 2024).

Methodology inspired by MTRAG-synthetic (Katsis et al., TACL 2025):
200 GPT-4 generated conversations accepted in a top NLP venue.

Architecture:
  692 seeds (Ulysses-RFCorpus) → cluster sessions → generate turns → judge relevance
  Turn 1: ground truth from original Conle specialists (human, inherited)
  Turns 2+: LLM-as-judge (different model from generator to avoid bias)
"""
from __future__ import annotations

import json
import pickle
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from mestrado.data.schema import Bill, FeedbackItem, Query


TurnType = Literal["initial", "follow_up", "clarification", "nonstandalone", "temporal", "comparative"]
AnswerType = Literal["answerable", "partial", "unanswerable", "conversational"]


@dataclass
class Turn:
    n: int
    type: TurnType
    query: str
    answer_type: AnswerType = "answerable"
    # Populated after retrieval + judging
    retrieved_bill_names: list[str] = field(default_factory=list)
    judgments: dict[str, float] = field(default_factory=dict)  # bill_name -> relevance [0,1]


@dataclass
class Conversation:
    session_id: str
    seed: Query
    turns: list[Turn]
    topic_cluster: int = -1

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "topic_cluster": self.topic_cluster,
            "seed_query_id": self.seed.query_id,
            "turns": [
                {
                    "n": t.n,
                    "type": t.type,
                    "query": t.query,
                    "answer_type": t.answer_type,
                    "retrieved_bill_names": t.retrieved_bill_names,
                    "judgments": t.judgments,
                }
                for t in self.turns
            ],
        }


class UlyssesMTRAGGenerator:
    """
    Generates synthetic multi-turn conversations from Ulysses-RFCorpus seeds.

    Turn 1 inherits ground truth from 54 Conle specialists (human).
    Turns 2+ use LLM-as-judge (generator ≠ judge model to avoid correlation).

    Based on MTRAG-synthetic methodology (Katsis et al., TACL 2025):
      - GPT-4 generated conversations accepted at TACL
      - This class uses local Ollama models for cost-free generation
    """

    TURN_TYPE_DISTRIBUTION = {
        # Based on MTRAG statistics (Katsis et al., 2025)
        "follow_up": 0.45,        # follow-up on previous answer (most common)
        "clarification": 0.20,    # ask for clarification
        "nonstandalone": 0.15,    # implicitly references previous turn
        "temporal": 0.10,         # adds temporal constraint
        "comparative": 0.10,      # compare two bills or themes
    }

    GENERATION_SYSTEM_PROMPT = """Você é um consultor legislativo sênior da Câmara dos Deputados fazendo
pesquisa sobre proposições legislativas. Você está interagindo com um sistema de busca legislativa.
Sua linguagem é formal mas direta. Você aprofunda progressivamente sua pesquisa em cada turno."""

    def __init__(
        self,
        generator_model: str = "llama3.1:8b",
        judge_model: str = "phi3:medium",
        seed_n_relevant_min: int = 3,
        n_turns: int = 4,
        random_seed: int = 42,
        provider: str = "ollama",
    ):
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.seed_n_relevant_min = seed_n_relevant_min
        self.n_turns = n_turns
        self.rng = random.Random(random_seed)
        self._rng_lock = threading.Lock()  # rng is shared across worker threads
        # Backend for follow-up generation: "ollama" (local, default — original
        # behavior preserved) or "deepinfra"/"openrouter" (OpenAI-compatible API).
        # Additive: the Ollama path is unchanged, so thesis reuse is unaffected.
        self.provider = provider
        self._api_client = None  # lazy OpenAI-compatible client

        if generator_model == judge_model:
            print(
                f"[WARNING] Generator and judge are the same model ({generator_model}). "
                "This may introduce correlation bias. Consider using different models."
            )

    def _get_api_client(self):
        """Lazily build an OpenAI-compatible client for the chosen API provider.
        Mirrors the pattern in mestrado.generation.rag (DeepInfraRAG/OpenRouterRAG)."""
        if self._api_client is None:
            import openai
            from mestrado import config
            if self.provider == "deepinfra":
                self._api_client = openai.OpenAI(
                    base_url="https://api.deepinfra.com/v1/openai",
                    api_key=config.deepinfra.api_key,
                )
            elif self.provider == "openrouter":
                self._api_client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=config.openrouter.api_key,
                )
            else:
                raise ValueError(f"Unknown API provider: {self.provider!r}")
        return self._api_client

    def _generate_turn_json(self, prompt: str) -> dict:
        """Provider-agnostic follow-up generation. Returns a parsed dict with
        keys 'query'/'answer_type'. Works with Ollama (local) or an
        OpenAI-compatible API (deepinfra/openrouter)."""
        if self.provider == "ollama":
            import ollama  # local backend — optional dependency
            response = ollama.generate(
                model=self.generator_model,
                prompt=prompt,
                system=self.GENERATION_SYSTEM_PROMPT,
                format="json",
                options={"temperature": 0.7, "num_predict": 256},
            )
            raw = response["response"]
        else:
            resp = self._get_api_client().chat.completions.create(
                model=self.generator_model,
                messages=[
                    {"role": "system", "content": self.GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content
        # Robust JSON extraction (tolerates markdown fences / extra prose)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        return json.loads(match.group(0)) if match else {}

    def select_seeds(self, queries: list[Query]) -> list[Query]:
        """Select queries with enough relevant bills to serve as session seeds."""
        return [
            q for q in queries
            if sum(1 for f in q.feedback if f.label in ("r", "pr")) >= self.seed_n_relevant_min
        ]

    def cluster_sessions(
        self, seeds: list[Query], n_sessions: int = 100, embeddings_cache: Path | None = None
    ) -> list[list[Query]]:
        """
        Cluster queries by semantic similarity to form coherent sessions.
        Each cluster becomes a candidate session (pick best seed from cluster).
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        cache_key = embeddings_cache
        if cache_key and cache_key.exists():
            with open(cache_key, "rb") as f:
                embeddings = pickle.load(f)
        else:
            model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
            texts = [q.text for q in seeds]
            embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
            if cache_key:
                with open(cache_key, "wb") as f:
                    pickle.dump(embeddings, f)

        n_clusters = min(n_sessions, len(seeds))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters: dict[int, list[Query]] = {}
        for q, label in zip(seeds, labels):
            clusters.setdefault(int(label), []).append(q)

        return list(clusters.values())

    def select_seed_from_cluster(self, cluster: list[Query]) -> Query:
        """Pick the query with the most relevant documents as cluster representative."""
        return max(cluster, key=lambda q: sum(1 for f in q.feedback if f.label == "r"))

    def generate_conversation(self, seed: Query, session_id: str, cluster: int = -1) -> Conversation:
        """Generate a full multi-turn conversation from a seed query."""
        turns: list[Turn] = []

        # Turn 1: the seed query itself (ground truth = Conle specialists)
        turns.append(Turn(n=1, type="initial", query=seed.text, answer_type="answerable"))

        # Turns 2+: LLM-generated follow-ups
        for t in range(2, self.n_turns + 1):
            turn_type = self._sample_turn_type()
            prompt = self._build_generation_prompt(seed, turns, t, turn_type)

            try:
                parsed = self._generate_turn_json(prompt)
                query = parsed.get("query", "").strip()
                answer_type = parsed.get("answer_type", "answerable")
            except Exception:
                # Fallback: deterministic template follow-up
                query = self._generate_simple_followup(seed, turns, turn_type)
                answer_type = "answerable"

            if not query:
                query = seed.text  # last resort: repeat seed

            turns.append(Turn(n=t, type=turn_type, query=query, answer_type=answer_type))

        return Conversation(session_id=session_id, seed=seed, turns=turns, topic_cluster=cluster)

    def _sample_turn_type(self) -> TurnType:
        types = list(self.TURN_TYPE_DISTRIBUTION.keys())
        weights = list(self.TURN_TYPE_DISTRIBUTION.values())
        with self._rng_lock:  # thread-safe under parallel generation
            return self.rng.choices(types, weights=weights, k=1)[0]

    def _build_generation_prompt(
        self, seed: Query, history: list[Turn], turn_n: int, turn_type: TurnType
    ) -> str:
        history_str = "\n".join(
            f"  Turno {t.n} ({t.type}): {t.query}" for t in history
        )

        type_instructions = {
            "follow_up": "Aprofunde em um aspecto específico da consulta anterior.",
            "clarification": "Peça esclarecimento sobre algo na resposta anterior ou na consulta inicial.",
            "nonstandalone": "Faça uma pergunta que implicitamente referencia o turno anterior (use 'esses', 'elas', 'aqueles' sem especificar).",
            "temporal": "Adicione uma restrição temporal (ex: 'mais recentes', 'de 2020 em diante', 'anteriores a 2018').",
            "comparative": "Peça comparação entre dois aspectos ou subgrupos do tema.",
        }

        instruction = type_instructions.get(turn_type, "Continue a pesquisa naturalmente.")

        return f"""Histórico da sessão legislativa:
{history_str}

Tema central (seed): {seed.text}

Gere o turno {turn_n} da conversa. Tipo: {turn_type}.
Instrução: {instruction}

Responda APENAS com JSON válido:
{{
  "query": "a pergunta do consultor neste turno",
  "answer_type": "answerable" | "partial" | "unanswerable",
  "rationale": "breve justificativa (1 frase)"
}}
"""

    def _generate_simple_followup(self, seed: Query, history: list[Turn], turn_type: TurnType) -> str:
        """Fallback: generate a simple follow-up without JSON."""
        templates = {
            "follow_up": f"Quais são os aspectos mais específicos sobre {seed.text}?",
            "clarification": "Pode detalhar melhor as propostas mais recentes mencionadas?",
            "nonstandalone": "E quais dessas estão atualmente em tramitação?",
            "temporal": "Existem versões mais recentes, de 2020 em diante?",
            "comparative": "Como essas propostas diferem entre si em termos de escopo?",
        }
        return templates.get(turn_type, f"Mais informações sobre {seed.text}?")

    def generate_dataset(
        self,
        queries: list[Query],
        n_sessions: int = 100,
        output_path: Path | None = None,
        embeddings_cache: Path | None = None,
        max_workers: int = 1,
    ) -> list[Conversation]:
        """
        Generate the full Ulysses-MTRAG dataset.

        Args:
            queries: All Ulysses-RFCorpus queries
            n_sessions: Number of conversations to generate
            output_path: If provided, save dataset to JSON
            embeddings_cache: Cache for sentence embeddings
            max_workers: Parallel API workers (1 = sequential, original behavior).
                         OpenRouter/DeepInfra accept concurrent requests; turns
                         within a conversation stay sequential (they need history),
                         but conversations are generated in parallel.

        Returns:
            List of generated conversations
        """
        seeds = self.select_seeds(queries)
        print(f"[Generator] {len(seeds)} seeds selected (≥{self.seed_n_relevant_min} relevant bills)")

        clusters = self.cluster_sessions(seeds, n_sessions=n_sessions, embeddings_cache=embeddings_cache)
        print(f"[Generator] {len(clusters)} topic clusters formed")

        tasks = [
            (i, self.select_seed_from_cluster(cluster), f"ulysses_mt_{i:04d}")
            for i, cluster in enumerate(clusters)
        ]

        conversations: list[Conversation] = []
        if max_workers and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(f"[Generator] generating {len(tasks)} conversations with {max_workers} workers ...")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self.generate_conversation, seed, sid, i): sid
                    for (i, seed, sid) in tasks
                }
                done = 0
                for fut in as_completed(futures):
                    conversations.append(fut.result())
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        print(f"[Generator]   {done}/{len(tasks)} conversations done")
            conversations.sort(key=lambda c: c.session_id)  # restore deterministic order
        else:
            for (i, seed, sid) in tasks:
                print(f"[Generator] [{i+1}/{len(tasks)}] Session {sid}: seed='{seed.text[:60]}...'")
                conversations.append(self.generate_conversation(seed, session_id=sid, cluster=i))

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "name": "Ulysses-MTRAG",
                            "version": "0.1.0-synthetic",
                            "description": "Synthetic multi-turn legislative RAG benchmark. "
                            "Extension of Ulysses-RFCorpus (Vitório et al., LRE 2024). "
                            "Methodology: MTRAG-synthetic (Katsis et al., TACL 2025).",
                            "n_conversations": len(conversations),
                            "n_turns_per_conv": self.n_turns,
                            "generator_model": self.generator_model,
                            "judge_model": self.judge_model,
                            "base_corpus": "Ulysses-RFCorpus (Vitório et al., 2024)",
                            "license": "CC BY 4.0 (same as base corpus)",
                        },
                        "conversations": [c.to_dict() for c in conversations],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[Generator] Dataset saved to {output_path}")

        return conversations
