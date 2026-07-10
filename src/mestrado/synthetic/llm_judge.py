"""
LLM-as-Judge for multi-turn relevance assessment.

Evaluates relevance of bills for turns 2+ in Ulysses-MTRAG conversations.
Turn 1 uses ground truth from Conle specialists (human, from Ulysses-RFCorpus).

Methodology:
  - Generator model ≠ Judge model (to avoid correlation bias)
  - Inter-rater agreement measured via Fleiss' Kappa (multiple judges)
  - Scores mapped to [0, 1] to match Ulysses-RFCorpus schema (r=1, pr=0.5, i=0)

References:
  - Zheng et al. (2023): Judging LLM-as-a-Judge with MT-Bench (NeurIPS 2023)
  - MTRAG-synthetic uses GPT-4 as judge (Katsis et al., TACL 2025)
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from statistics import mean

import ollama


RELEVANCE_TO_SCORE = {"r": 1.0, "pr": 0.5, "i": 0.0}
SCORE_TO_RELEVANCE = {1.0: "r", 0.5: "pr", 0.0: "i"}


@dataclass
class JudgmentResult:
    bill_name: str
    relevance_class: str  # "r", "pr", "i"
    score: float          # 0.0, 0.5, or 1.0
    confidence: str       # "high", "medium", "low"
    rationale: str


class LLMJudge:
    """
    LLM-as-Judge for relevance assessment of bills for multi-turn queries.

    Use a model DIFFERENT from the generator to avoid correlation bias.
    For inter-rater agreement, instantiate multiple LLMJudge objects
    with different models and measure Fleiss' Kappa.
    """

    JUDGE_SYSTEM = """Você é um avaliador de relevância de proposições legislativas para consultas de assessores.
Avalie pela EVIDÊNCIA DIRETA no conteúdo (ementa + trecho), não por palavras-chave superficiais nem por tema geral.
Não seja sistematicamente leniente nem severo — aplique os critérios de forma consistente e justifique pela evidência.
Critérios:
- "r"  = trata DIRETAMENTE do objeto específico da consulta (mesmo tema E mesmo escopo/ação).
- "pr" = toca o tema de forma tangencial, ou trata de tema correlato mas não do objeto específico.
- "i"  = não trata do objeto da consulta, mesmo que compartilhe palavras ou área geral."""

    def __init__(self, model: str = "phi3:medium"):
        self.model = model

    def judge_single(self, query: str, bill_name: str, bill_ementa: str, bill_text_excerpt: str = "") -> JudgmentResult:
        """
        Judge relevance of a single bill for a query.

        Args:
            query: The turn query (already resolved to standalone form)
            bill_name: Bill identifier (e.g., "PL 3650/2021")
            bill_ementa: Bill summary (ementa)
            bill_text_excerpt: First 500 chars of bill text (optional, for context)
        """
        context = f"Ementa: {bill_ementa}"
        if bill_text_excerpt:
            context += f"\n\nTrecho do texto: {bill_text_excerpt[:500]}..."

        prompt = f"""Avalie a relevância desta proposição legislativa para a consulta:

CONSULTA: {query}

PROPOSIÇÃO: {bill_name}
{context}

Classifique a relevância como:
- "r" = muito relevante (aborda diretamente o tema da consulta)
- "pr" = parcialmente relevante (aborda o tema de forma tangencial)
- "i" = irrelevante (não aborda o tema da consulta)

Responda com JSON:
{{
  "relevance": "r" | "pr" | "i",
  "confidence": "high" | "medium" | "low",
  "rationale": "justificativa em 1-2 frases"
}}"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=self.JUDGE_SYSTEM,
                format="json",
                options={"temperature": 0.0, "num_predict": 256},  # deterministic
            )
            parsed = json.loads(response["response"])
            rel = parsed.get("relevance", "i")
            if rel not in ("r", "pr", "i"):
                rel = "i"

            return JudgmentResult(
                bill_name=bill_name,
                relevance_class=rel,
                score=RELEVANCE_TO_SCORE[rel],
                confidence=parsed.get("confidence", "medium"),
                rationale=parsed.get("rationale", ""),
            )
        except Exception:
            return JudgmentResult(
                bill_name=bill_name,
                relevance_class="i",
                score=0.0,
                confidence="low",
                rationale="[judgment failed]",
            )

    def judge_batch(
        self,
        query: str,
        bills: list[dict],  # list of {"name": str, "ementa": str, "text": str}
    ) -> dict[str, float]:
        """
        Judge relevance of multiple bills for a query.

        Returns:
            dict mapping bill_name -> relevance_score [0, 0.5, 1.0]
        """
        judgments = {}
        for bill in bills:
            result = self.judge_single(
                query=query,
                bill_name=bill["name"],
                bill_ementa=bill.get("ementa", ""),
                bill_text_excerpt=bill.get("text", "")[:500],
            )
            judgments[bill["name"]] = result.score
        return judgments


class _RemoteJudgeBase(LLMJudge):
    """
    Base para juízes que usam LLMClient (DeepInfra, OpenRouter).
    Evita duplicação de código entre OpenRouterJudge e DeepInfraJudge.
    """

    def __init__(self, model: str, provider: str) -> None:
        from mestrado.utils.llm_client import get_client
        self.model = model
        self._client = get_client(provider)

    def judge_single(self, query: str, bill_name: str, bill_ementa: str, bill_text_excerpt: str = "") -> JudgmentResult:
        context = f"Ementa: {bill_ementa}"
        if bill_text_excerpt:
            context += f"\n\nTrecho: {bill_text_excerpt[:500]}..."

        prompt = (
            f"Avalie a relevância da proposição para a consulta.\n\n"
            f"CONSULTA: {query}\n\n"
            f"PROPOSIÇÃO: {bill_name}\n{context}\n\n"
            f"Identifique o OBJETO ESPECÍFICO da consulta e decida, pela evidência, "
            f"se a proposição o trata.\n\n"
            f"Responda em JSON (nesta ordem):\n"
            f'{{"rationale": "1 frase: objeto da consulta e se a proposição trata disso", '
            f'"relevance": "r" | "pr" | "i", "confidence": "high" | "medium" | "low"}}'
        )
        messages = [
            {"role": "system", "content": self.JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._client.complete(self.model, messages, max_tokens=256, json_mode=True)
            parsed = json.loads(text)
            rel = parsed.get("relevance", "i")
            if rel not in ("r", "pr", "i"):
                rel = "i"
            return JudgmentResult(
                bill_name=bill_name,
                relevance_class=rel,
                score=RELEVANCE_TO_SCORE[rel],
                confidence=parsed.get("confidence", "medium"),
                rationale=parsed.get("rationale", ""),
            )
        except Exception:
            return JudgmentResult(bill_name=bill_name, relevance_class="i", score=0.0, confidence="low", rationale=f"[{self._client.provider} judgment failed]")


class OpenRouterJudge(_RemoteJudgeBase):
    """
    LLM-as-Judge via OpenRouter API.

    Recomendado como 3º juiz:
      - meta-llama/llama-3.3-70b-instruct (~$0.40/M tokens, forte em PT-BR)
      - qwen/qwen-2.5-72b-instruct (~$0.35/M tokens, multilingual)

    Obter API key: https://openrouter.ai
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from mestrado.config import openrouter as cfg
        if not (api_key or cfg.api_key):
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")
        # Override api_key if provided explicitly
        if api_key:
            import os as _os
            _os.environ.setdefault("OPENROUTER_API_KEY", api_key)
        super().__init__(model=model or cfg.judge_model, provider="openrouter")


class DeepInfraJudge(_RemoteJudgeBase):
    """
    LLM-as-Judge via DeepInfra API.

    Recomendado como 2º ou 3º juiz:
      - meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo (~$0.35/M tokens)
      - meta-llama/Meta-Llama-3.1-8B-Instruct (~$0.07/M tokens, rápido)

    Obter API key: https://deepinfra.com
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from mestrado.config import deepinfra as cfg
        if not (api_key or cfg.api_key):
            raise ValueError("DEEPINFRA_API_KEY não configurada no .env")
        if api_key:
            import os as _os
            _os.environ.setdefault("DEEPINFRA_API_KEY", api_key)
        super().__init__(model=model or cfg.judge_model, provider="deepinfra")


class MaritacaJudge(LLMJudge):
    """
    LLM-as-Judge usando Gaia/Sabiá (Maritaca AI) — especializado em PT-BR.

    Usa autenticação própria ("Key {token}") diferente do padrão Bearer.
    Obter API key: https://maritaca.ai — Modelos: sabia-3, sabia-2-medium
    """

    MARITACA_URL = "https://chat.maritaca.ai/api/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from mestrado.config import maritaca as cfg
        self.model = model or cfg.judge_model
        self._api_key = api_key or cfg.api_key
        if not self._api_key:
            raise ValueError("MARITACA_API_KEY não configurada no .env")

    def judge_single(self, query: str, bill_name: str, bill_ementa: str, bill_text_excerpt: str = "") -> JudgmentResult:
        context = f"Ementa: {bill_ementa}"
        if bill_text_excerpt:
            context += f"\n\nTrecho: {bill_text_excerpt[:500]}..."

        prompt = (
            f"Avalie a relevância desta proposição legislativa para a consulta:\n\n"
            f"CONSULTA: {query}\n\nPROPOSIÇÃO: {bill_name}\n{context}\n\n"
            f'Responda com JSON:\n{{"relevance": "r" | "pr" | "i", '
            f'"confidence": "high" | "medium" | "low", "rationale": "justificativa"}}'
        )
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }).encode()
        req = urllib.request.Request(
            self.MARITACA_URL,
            data=payload,
            headers={"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(content)
            rel = parsed.get("relevance", "i")
            if rel not in ("r", "pr", "i"):
                rel = "i"
            return JudgmentResult(
                bill_name=bill_name,
                relevance_class=rel,
                score=RELEVANCE_TO_SCORE[rel],
                confidence=parsed.get("confidence", "medium"),
                rationale=parsed.get("rationale", ""),
            )
        except Exception:
            return JudgmentResult(bill_name=bill_name, relevance_class="i", score=0.0, confidence="low", rationale="[maritaca judgment failed]")


def get_judge(provider: str | None = None, model: str | None = None) -> LLMJudge:
    """
    Retorna o juiz LLM correto baseado no provedor.

    - "ollama"      → LLMJudge (local, usa biblioteca ollama)
    - "deepinfra"   → DeepInfraJudge
    - "openrouter"  → OpenRouterJudge
    - "maritaca"    → MaritacaJudge
    - None          → lê LLM_PROVIDER do .env (default: deepinfra)

    Uso típico (3 juízes independentes para Fleiss' Kappa):
        j1 = get_judge("ollama", model="phi3:medium")          # juiz local
        j2 = get_judge("deepinfra", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
        j3 = get_judge("deepinfra", model="meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo")
    """
    from mestrado.config import llm_provider as prov_cfg, deepinfra as di_cfg, openrouter as or_cfg
    p = provider or prov_cfg.provider

    if p == "ollama":
        return LLMJudge(model=model or "phi3:medium")
    if p == "openrouter":
        return OpenRouterJudge(model=model or or_cfg.judge_model)
    if p == "maritaca":
        return MaritacaJudge(model=model)
    # default: deepinfra
    return DeepInfraJudge(model=model or di_cfg.judge_model)


def compute_fleiss_kappa(judgments_by_judge: list[dict[str, float]], items: list[str]) -> float:
    """
    Compute Fleiss' Kappa for inter-rater agreement across multiple LLM judges.

    Args:
        judgments_by_judge: List of judgment dicts (one per judge)
        items: List of item names (bill names)

    Returns:
        Fleiss' Kappa coefficient [-1, 1]. >0.6 = substantial agreement.
    """
    import numpy as np

    n_judges = len(judgments_by_judge)
    n_items = len(items)
    categories = [0.0, 0.5, 1.0]
    n_cats = len(categories)

    # Build rating matrix [n_items x n_cats]
    matrix = np.zeros((n_items, n_cats))
    for i, item in enumerate(items):
        for judge_judgments in judgments_by_judge:
            score = judge_judgments.get(item, 0.0)
            cat_idx = categories.index(min(categories, key=lambda c: abs(c - score)))
            matrix[i, cat_idx] += 1

    # Fleiss' Kappa formula
    p_j = matrix.sum(axis=0) / (n_items * n_judges)  # proportion per category
    P_bar = (matrix * (matrix - 1)).sum() / (n_items * n_judges * (n_judges - 1))
    P_bar_e = (p_j ** 2).sum()

    if P_bar_e == 1.0:
        return 1.0

    kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
    return float(kappa)


def aggregate_judgments(
    judgments_by_judge: list[dict[str, float]], strategy: str = "mean"
) -> dict[str, float]:
    """
    Aggregate judgments from multiple LLM judges.

    Args:
        judgments_by_judge: List of judgment dicts
        strategy: "mean", "majority", or "conservative" (min score)

    Returns:
        Aggregated judgment dict
    """
    all_items = set()
    for j in judgments_by_judge:
        all_items.update(j.keys())

    aggregated = {}
    for item in all_items:
        scores = [j.get(item, 0.0) for j in judgments_by_judge]

        if strategy == "mean":
            aggregated[item] = mean(scores)
        elif strategy == "majority":
            # Round to nearest valid score
            avg = mean(scores)
            aggregated[item] = min([0.0, 0.5, 1.0], key=lambda c: abs(c - avg))
        elif strategy == "conservative":
            aggregated[item] = min(scores)
        else:
            aggregated[item] = mean(scores)

    return aggregated
