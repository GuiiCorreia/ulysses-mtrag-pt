"""
Context Compactor for SLM experiments — Experiment 8.

Generates structured summaries of bills using Ollama, so they fit
within SLM context windows (4K–8K tokens typical).

DIRECT PORT of src/services/compact/prompt.ts structure:
- BASE_COMPACT_PROMPT 9-section structure → PL_COMPACT_PROMPT 5-section structure
- <analysis> scratchpad pattern → same pattern preserved
- formatCompactSummary() → _strip_analysis_tags()
- POST_COMPACT_MAX_TOKENS_PER_FILE = 5000 → MAX_TOKENS_PER_SUMMARY = 512

The key insight from compact.ts: structured summarization preserves
the most retrieval-relevant information while dramatically reducing size.
For PLs: ementa + area + beneficiarios + mecanismo + keywords = ~512 tokens
vs. full text = ~3000–10000 tokens → 6–20× compression ratio.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
import pickle

from mestrado.data.schema import Bill
from mestrado.utils.token_estimation import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — adapted from BASE_COMPACT_PROMPT in compact/prompt.ts
# ---------------------------------------------------------------------------

PL_COMPACT_PROMPT = """\
Analise o projeto de lei abaixo e produza um resumo estruturado.

<analysis>
Primeiro, faça um rascunho interno identificando:
- O que o PL propõe especificamente?
- Quem é beneficiado?
- Que área temática?
- Quais termos técnicos legislativos descrevem isso?
</analysis>

Após a análise, produza um JSON com exatamente estas 5 chaves:
{{
  "ementa_expandida": "objetivo principal em 1-2 frases claras e específicas",
  "area_tematica": "uma das categorias: saúde | educação | tributário | ambiental | trabalhista | previdenciário | segurança | transporte | agricultura | habitação | outro",
  "beneficiarios": "quem é diretamente afetado (ex: agricultores familiares, servidores públicos)",
  "mecanismo": "como a lei opera (ex: proibe X, cria Y, altera Z, incentiva W)",
  "keywords": ["termo1", "termo2", "termo3", "termo4", "termo5", "termo6", "termo7", "termo8"]
}}

PROJETO DE LEI:
Nome: {name}
Tipo: {sig_tipo}
Ementa oficial: {ementa}
Texto (primeiros 1500 chars): {text_excerpt}
"""

# Adapted from formatCompactSummary() — strips analysis scratchpad
_ANALYSIS_TAG_PATTERN = re.compile(r"<analysis>.*?</analysis>", re.DOTALL)


@dataclass
class BillSummary:
    """Structured summary of a bill — compact representation for SLM context."""
    bill_name: str
    ementa_expandida: str
    area_tematica: str
    beneficiarios: str
    mecanismo: str
    keywords: list[str]

    def to_retrieval_text(self) -> str:
        """
        Combines all fields into retrieval-optimized text.
        More information-dense than raw ementa alone.
        """
        return (
            f"{self.ementa_expandida} "
            f"Área: {self.area_tematica}. "
            f"Beneficiários: {self.beneficiarios}. "
            f"Mecanismo: {self.mecanismo}. "
            f"Termos: {', '.join(self.keywords)}"
        )

    def to_context_text(self) -> str:
        """Formatted text for LLM context injection."""
        return (
            f"**{self.bill_name}**\n"
            f"Resumo: {self.ementa_expandida}\n"
            f"Área: {self.area_tematica} | Beneficiários: {self.beneficiarios}\n"
            f"Termos-chave: {', '.join(self.keywords)}"
        )

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.to_context_text(), "ementa")


class ContextCompactor:
    """
    Generates structured summaries of bills using Ollama.

    Adapted from compact.ts / prompt.ts:
    - Structured summarization with analysis scratchpad
    - Token budget management
    - Batch processing with caching

    Usage:
        compactor = ContextCompactor()
        summaries = compactor.summarize_batch(bills[:1000])
        compactor.save_cache("results/bill_summaries.pkl")
    """

    # Analogous to POST_COMPACT_MAX_TOKENS_PER_FILE = 5000
    MAX_TOKENS_PER_SUMMARY = 512

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        from mestrado.config import ollama as cfg
        import ollama as ollama_lib
        self._ollama = ollama_lib
        self.model = model or cfg.generator_model
        self._cache: dict[str, BillSummary] = {}

    def summarize(self, bill: Bill) -> BillSummary:
        """Generate a structured summary for one bill."""
        if bill.name in self._cache:
            return self._cache[bill.name]

        prompt = PL_COMPACT_PROMPT.format(
            name=bill.name,
            sig_tipo=bill.sig_tipo,
            ementa=bill.txt_ementa[:500],
            text_excerpt=bill.text[:1500],
        )

        try:
            response = self._ollama.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0, "num_predict": 600},
            )
            raw = response.get("response", "{}")
            # Strip <analysis> tags (pattern from formatCompactSummary)
            raw = _ANALYSIS_TAG_PATTERN.sub("", raw).strip()
            data = json.loads(raw)
        except Exception as e:
            logger.debug(f"Summary generation failed for {bill.name}: {e}")
            data = {}

        summary = BillSummary(
            bill_name=bill.name,
            ementa_expandida=data.get("ementa_expandida", bill.txt_ementa[:300]),
            area_tematica=data.get("area_tematica", "outro"),
            beneficiarios=data.get("beneficiarios", ""),
            mecanismo=data.get("mecanismo", ""),
            keywords=data.get("keywords", [])[:8],
        )
        self._cache[bill.name] = summary
        return summary

    def summarize_batch(
        self,
        bills: list[Bill],
        show_progress: bool = True,
    ) -> dict[str, BillSummary]:
        """
        Summarize multiple bills. Returns {bill_name: BillSummary}.
        Skips already-cached bills.
        """
        to_process = [b for b in bills if b.name not in self._cache]
        logger.info(
            f"Summarizing {len(to_process)} bills "
            f"({len(self._cache)} already cached)..."
        )

        iterator = to_process
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(to_process, desc="Summarizing bills")

        for bill in iterator:
            self.summarize(bill)

        return {name: s for name, s in self._cache.items() if name in {b.name for b in bills}}

    def save_cache(self, path: Path | str) -> None:
        """Save summaries cache to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._cache, f)
        logger.info(f"Saved {len(self._cache)} summaries to {path}")

    def load_cache(self, path: Path | str) -> None:
        """Load previously saved summaries."""
        path = Path(path)
        if path.exists():
            with open(path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info(f"Loaded {len(self._cache)} cached summaries from {path}")
