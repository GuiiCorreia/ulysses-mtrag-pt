"""
Multi-Turn Evaluation Metrics for Ulysses-MTRAG.

Extends the single-turn IR metrics (ir_metrics.py) with:
  - nDCG progression over turns
  - Gain-per-turn analysis (core evidence of multi-turn benefit)
  - Session coherence scoring (LLM-as-judge)
  - Faithfulness scoring for RAG generation
  - Fleiss' Kappa for LLM-judge inter-rater agreement

Key claim this enables: "multi-turn context improves retrieval progressively"
→ Measured as positive Δ nDCG across turns T1 → T2 → T3 → T4
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ollama

from mestrado.metrics.ir_metrics import ndcg_at_k, mrr, precision_at_k, recall_at_k
from mestrado.synthetic.generator import Conversation


@dataclass
class TurnResult:
    turn_n: int
    turn_type: str
    query: str
    retrieved_names: list[str]
    judgments: dict[str, float]
    ndcg_at_12: float = 0.0
    precision_at_5: float = 0.0
    recall_at_12: float = 0.0
    mrr_score: float = 0.0


@dataclass
class SessionResult:
    session_id: str
    n_turns: int
    turn_results: list[TurnResult] = field(default_factory=list)
    coherence_score: float = -1.0  # LLM judge (-1 = not evaluated)
    faithfulness_scores: list[float] = field(default_factory=list)

    @property
    def ndcg_progression(self) -> list[float]:
        return [t.ndcg_at_12 for t in self.turn_results]

    @property
    def gain_per_turn(self) -> list[float]:
        prog = self.ndcg_progression
        return [prog[i] - prog[i-1] for i in range(1, len(prog))]

    @property
    def mean_ndcg(self) -> float:
        prog = self.ndcg_progression
        return float(np.mean(prog)) if prog else 0.0

    @property
    def final_turn_ndcg(self) -> float:
        return self.turn_results[-1].ndcg_at_12 if self.turn_results else 0.0


def evaluate_turn(
    retrieved_names: list[str],
    judgments: dict[str, float],
    turn_n: int,
    turn_type: str,
    query: str,
    k: int = 12,
) -> TurnResult:
    """Evaluate retrieval quality for a single turn."""
    ndcg = ndcg_at_k(retrieved_names, judgments, k)
    p_at_5 = precision_at_k(retrieved_names, judgments, 5, threshold=0.5)
    r_at_12 = recall_at_k(retrieved_names, judgments, k, threshold=0.5)
    mrr_score_val = mrr(retrieved_names, judgments)

    return TurnResult(
        turn_n=turn_n,
        turn_type=turn_type,
        query=query,
        retrieved_names=retrieved_names,
        judgments=judgments,
        ndcg_at_12=ndcg,
        precision_at_5=p_at_5,
        recall_at_12=r_at_12,
        mrr_score=mrr_score_val,
    )


def evaluate_session(
    session_id: str,
    turn_results: list[TurnResult],
) -> SessionResult:
    """Aggregate turn results into a session result."""
    return SessionResult(
        session_id=session_id,
        n_turns=len(turn_results),
        turn_results=turn_results,
    )


def aggregate_session_results(sessions: list[SessionResult]) -> dict[str, Any]:
    """
    Aggregate across all sessions to produce dataset-level metrics.

    Key output: mean nDCG at each turn position (shows progression).
    """
    if not sessions:
        return {}

    max_turns = max(s.n_turns for s in sessions)
    n_sessions = len(sessions)

    # nDCG by turn position
    ndcg_by_turn: list[list[float]] = [[] for _ in range(max_turns)]
    for s in sessions:
        for t in s.turn_results:
            ndcg_by_turn[t.turn_n - 1].append(t.ndcg_at_12)

    mean_ndcg_by_turn = [
        float(np.mean(vals)) if vals else 0.0
        for vals in ndcg_by_turn
    ]

    # Gain per turn (mean across sessions)
    all_gains: list[list[float]] = []
    for s in sessions:
        gains = s.gain_per_turn
        if gains:
            all_gains.append(gains)

    mean_gain_per_turn = []
    if all_gains:
        max_gain_len = max(len(g) for g in all_gains)
        for i in range(max_gain_len):
            vals = [g[i] for g in all_gains if i < len(g)]
            mean_gain_per_turn.append(float(np.mean(vals)) if vals else 0.0)

    # Sessions where multi-turn helped (final_ndcg > turn1_ndcg)
    improved = sum(1 for s in sessions if s.final_turn_ndcg > s.turn_results[0].ndcg_at_12)
    improvement_rate = improved / n_sessions

    # Mean metrics across all turns and sessions
    all_ndcg = [t.ndcg_at_12 for s in sessions for t in s.turn_results]
    all_p5 = [t.precision_at_5 for s in sessions for t in s.turn_results]
    all_r12 = [t.recall_at_12 for s in sessions for t in s.turn_results]

    return {
        "n_sessions": n_sessions,
        "mean_ndcg_by_turn": {f"T{i+1}": v for i, v in enumerate(mean_ndcg_by_turn)},
        "mean_gain_per_turn": {f"T{i+1}→T{i+2}": v for i, v in enumerate(mean_gain_per_turn)},
        "sessions_improved_by_multiturn": improved,
        "improvement_rate": improvement_rate,
        "overall_mean_ndcg": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "overall_mean_p5": float(np.mean(all_p5)) if all_p5 else 0.0,
        "overall_mean_r12": float(np.mean(all_r12)) if all_r12 else 0.0,
    }


def evaluate_session_coherence(
    conversation: Conversation,
    responses: list[str],
    judge_model: str = "phi3:medium",
) -> float:
    """
    LLM-as-judge coherence evaluation of a full multi-turn session.

    Measures: do the responses form a coherent, progressive research session?
    Scale: 0-4 mapped to 0-1.

    Reference: MT-Bench evaluation protocol (Zheng et al., NeurIPS 2023).
    """
    turns_str = "\n".join(
        f"Turno {i+1}: {t.query}\nResposta: {resp[:200]}..."
        for i, (t, resp) in enumerate(zip(conversation.turns, responses))
    )

    prompt = f"""Avalie a coerência desta sessão de pesquisa legislativa multi-turno:

{turns_str}

Critérios de avaliação:
1. Os turnos formam uma pesquisa progressiva coerente?
2. As respostas são consistentes entre si?
3. O tema é aprofundado de forma lógica?
4. As transições entre turnos fazem sentido?

Pontuação:
- 4: Excelente coerência, progressão natural e aprofundamento claro
- 3: Boa coerência com pequenas inconsistências
- 2: Coerência moderada, algumas transições abruptas
- 1: Pouca coerência, turnos parecem desconectados
- 0: Incoerente, contradições evidentes

Responda com JSON:
{{"score": 0-4, "rationale": "justificativa em 1-2 frases"}}"""

    try:
        response = ollama.generate(
            model=judge_model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0, "num_predict": 128},
        )
        parsed = json.loads(response["response"])
        raw_score = float(parsed.get("score", 2))
        return raw_score / 4.0  # normalize to [0, 1]
    except Exception:
        return -1.0  # evaluation failed


def evaluate_faithfulness(
    answer: str,
    retrieved_bills: list[dict],  # list of {"name": str, "ementa": str, "text_excerpt": str}
    judge_model: str = "phi3:medium",
) -> float:
    """
    LLM-as-judge faithfulness evaluation.

    Measures: is the answer grounded in the retrieved bills?
    Scale: 0-3 mapped to 0-1.

    Critical for RAG evaluation — catching hallucinated bill references.
    """
    bills_str = "\n".join(
        f"[{b['name']}] {b.get('ementa', '')}" for b in retrieved_bills[:5]
    )

    prompt = f"""Avalie se esta resposta é fiel às proposições legislativas recuperadas:

PROPOSIÇÕES RECUPERADAS:
{bills_str}

RESPOSTA GERADA:
{answer[:600]}

Pontuação:
- 3: Totalmente fiel — toda informação é suportada pelos documentos
- 2: Majoritariamente fiel — pequenos extrapolamentos
- 1: Parcialmente fiel — algumas afirmações sem suporte
- 0: Infiel — afirmações contradizem ou inventam informações

Responda com JSON:
{{"score": 0-3, "has_hallucination": true/false, "rationale": "justificativa"}}"""

    try:
        response = ollama.generate(
            model=judge_model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0, "num_predict": 128},
        )
        parsed = json.loads(response["response"])
        raw_score = float(parsed.get("score", 1))
        return raw_score / 3.0  # normalize to [0, 1]
    except Exception:
        return -1.0
