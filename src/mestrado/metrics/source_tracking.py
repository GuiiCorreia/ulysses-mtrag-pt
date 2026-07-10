"""
Source tracking metrics.

Measures if RAG responses correctly cite and reference retrieved documents.
Critical for RAG evaluation as it measures factual grounding and attribution.

Emerging methodology for 2026 RAG evaluation.

References:
  - RAGAS (Retrieval Augmented Generation Assessment) framework
  - Gao et al. (2023). Evaluating Retrieval-Augmented Generation Systems.
  - Custom methodology for legislative/legal RAG citation tracking.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Set

logger = logging.getLogger(__name__)


@dataclass
class SourceTrackingMetrics:
    """Source tracking metrics for RAG evaluation.

    Attributes:
        citation_precision: Fraction of citations that are valid (in context)
        citation_recall: Fraction of relevant documents that are cited
        citation_f1: F1 score for citation performance
        hallucination_rate: Fraction of statements without source support
        avg_citations_per_response: Average number of citations per response
        source_coverage: Fraction of responses with at least one citation
    """
    citation_precision: float
    citation_recall: float
    citation_f1: float
    hallucination_rate: float
    avg_citations_per_response: float
    source_coverage: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "citation_f1": round(self.citation_f1, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "avg_citations_per_response": round(self.avg_citations_per_response, 2),
            "source_coverage": round(self.source_coverage, 4),
        }


SOURCE_TRACKING_PROMPT = """\
Avalie a precisão das citações na resposta em relação aos documentos fornecidos.

Documentos disponíveis (IDs válidos):
{valid_doc_ids}

Resposta gerada:
{response}

Instruções:
1. Identifique todas as menções a documentos na resposta (ex: "Acórdão 123", "PL 456").
2. Verifique se cada documento mencionado está na lista de documentos válidos.
3. Ignore menções genéricas que não são citações específicas (ex: "a legislação", "os acórdãos").

Retorne APENAS um número de 0.0 a 1.0:
- 1.0 = Todas as citações são válidas (todos os documentos citados estão no contexto)
- 0.7 = A maioria das citações são válidas, mas 1-2 inválidas
- 0.5 = Metade das citações são válidas
- 0.3 = Poucas citações válidas, muitas inválidas
- 0.0 = Nenhuma citação válida ou citações completamente incorretas

Score:
"""


def extract_citations(response: str) -> List[str]:
    """
    Extract document citations from a response.

    Looks for patterns like:
    - "Acórdão 123", "ACÓRDÃO 123", "Acórdão n. 123"
    - "PL 456", "Projeto de Lei 456", "PL 456/2024"
    - "[DOC123]", "(DOC123)", "DOC123"

    Args:
        response: Generated RAG response

    Returns:
        List of cited document IDs
    """
    citations = []

    # Pattern 1: Acórdão/ACÓRDÃO
    acordao_pattern = r'[Aa]c[oó]rdã[o]\s+(?:n\.?\s*)?(\d+(?:/\d+)?)'
    citations.extend(re.findall(acordao_pattern, response))

    # Pattern 2: PL/Projeto de Lei
    pl_pattern = r'(?:[Pp][Ll]\s*|Projeto\s*[Dd]e\s*[Ll]ei\s*)(\d+(?:/\d+)?)'
    citations.extend(re.findall(pl_pattern, response))

    # Pattern 3: DOC123, [DOC123], (DOC123)
    doc_pattern = r'(?:\[\(DOC)?(\d+)(?:\)\])'
    citations.extend(re.findall(doc_pattern, response))

    # Pattern 4: Direct number references (context-dependent)
    # This is more general and may need filtering
    number_pattern = r'(?<!\d)(\d{3,6})(?!\d)'
    # Only include if surrounded by words like "documento", "doc", etc.
    number_context = r'(?:documento|doc|acórdão|pl|projeto)\s*[:\#]?\s*(\d{3,6})'
    citations.extend(re.findall(number_context, response, re.IGNORECASE))

    # Normalize citations
    normalized = []
    for citation in citations:
        citation = citation.strip()
        if citation and len(citation) >= 3:  # Minimum length to be meaningful
            normalized.append(citation)

    return list(set(normalized))  # Remove duplicates


def evaluate_citation_precision(
    response: str,
    valid_doc_ids: Set[str],
    judge_model: Any,
) -> float:
    """
    Evaluate citation precision using LLM-as-judge.

    Args:
        response: Generated RAG response
        valid_doc_ids: Set of valid document IDs (retrieved documents)
        judge_model: LLM instance for evaluation

    Returns:
        Precision score (0.0-1.0)
    """
    if not response:
        return 1.0  # No response, no incorrect citations

    citations = extract_citations(response)

    if not citations:
        return 1.0  # No citations, but also no incorrect ones

    # Use LLM-as-judge for more robust evaluation
    valid_ids_str = ", ".join(sorted(valid_doc_ids))
    prompt = SOURCE_TRACKING_PROMPT.format(valid_doc_ids=valid_ids_str, response=response)

    try:
        result = judge_model._client.chat.completions.create(
            model=judge_model.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )

        content = result.choices[0].message.content.strip()

        # Parse score (reuse from relevance_metrics)
        from .relevance_metrics import parse_llm_score
        score = parse_llm_score(content)

        logger.debug(f"Citation precision: {score:.2f}, found {len(citations)} citations")
        return score

    except Exception as e:
        logger.warning(f"Citation precision evaluation failed: {e}. Using heuristic")

        # Fallback: simple heuristic
        valid_citations = 0
        for citation in citations:
            citation_norm = citation.lower().strip()
            for valid_id in valid_doc_ids:
                if valid_id.lower() in citation_norm or citation_norm in valid_id.lower():
                    valid_citations += 1
                    break

        return valid_citations / len(citations) if citations else 1.0


def evaluate_citation_recall(
    response: str,
    relevant_doc_ids: Set[str],
) -> float:
    """
    Evaluate citation recall: fraction of relevant documents that are cited.

    Args:
        response: Generated RAG response
        relevant_doc_ids: Set of relevant document IDs (ground truth)

    Returns:
        Recall score (0.0-1.0)
    """
    if not relevant_doc_ids:
        return 1.0  # No relevant documents to cite

    citations = extract_citations(response)

    if not citations:
        return 0.0  # No citations, missed all relevant documents

    # Count how many relevant documents are cited
    cited_relevant = 0
    for citation in citations:
        citation_norm = citation.lower().strip()
        for relevant_id in relevant_doc_ids:
            if relevant_id.lower() in citation_norm or citation_norm in relevant_id.lower():
                cited_relevant += 1
                break

    return cited_relevant / len(relevant_doc_ids)


def evaluate_source_tracking(
    responses: List[str],
    retrieved_doc_ids_list: List[Set[str]],
    relevant_doc_ids_list: List[Set[str]],
    judge_model: Any,
) -> SourceTrackingMetrics:
    """
    Evaluate comprehensive source tracking metrics.

    Args:
        responses: List of generated RAG responses
        retrieved_doc_ids_list: List of sets of retrieved document IDs (one per response)
        relevant_doc_ids_list: List of sets of relevant document IDs (ground truth, one per response)
        judge_model: LLM instance for citation precision evaluation

    Returns:
        SourceTrackingMetrics with all tracking metrics
    """
    if len(responses) != len(retrieved_doc_ids_list):
        raise ValueError(f"Length mismatch: {len(responses)} responses vs {len(retrieved_doc_ids_list)} retrieved doc sets")

    if len(responses) != len(relevant_doc_ids_list):
        raise ValueError(f"Length mismatch: {len(responses)} responses vs {len(relevant_doc_ids_list)} relevant doc sets")

    citation_precisions = []
    citation_recalls = []
    citation_f1s = []
    citation_counts = []

    for response, retrieved_ids, relevant_ids in zip(responses, retrieved_doc_ids_list, relevant_doc_ids_list):
        # Citation precision: are cited docs valid (in retrieved)?
        precision = evaluate_citation_precision(response, retrieved_ids, judge_model)
        citation_precisions.append(precision)

        # Citation recall: are relevant docs cited?
        recall = evaluate_citation_recall(response, relevant_ids)
        citation_recalls.append(recall)

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        citation_f1s.append(f1)

        # Count citations
        citations = extract_citations(response)
        citation_counts.append(len(citations))

    # Calculate aggregated metrics
    avg_precision = sum(citation_precisions) / len(citation_precisions) if citation_precisions else 0.0
    avg_recall = sum(citation_recalls) / len(citation_recalls) if citation_recalls else 0.0
    avg_f1 = sum(citation_f1s) / len(citation_f1s) if citation_f1s else 0.0
    avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0.0
    source_coverage = sum(1 for c in citation_counts if c > 0) / len(citation_counts) if citation_counts else 0.0

    # Hallucination rate: responses with 0 citations (and presumably should have some)
    hallucination_rate = 1.0 - source_coverage

    return SourceTrackingMetrics(
        citation_precision=avg_precision,
        citation_recall=avg_recall,
        citation_f1=avg_f1,
        hallucination_rate=hallucination_rate,
        avg_citations_per_response=avg_citations,
        source_coverage=source_coverage,
    )


def calculate_source_tracking_batch(
    results: Dict[str, Dict],
    retrieved_docs_list: List[List[Any]],  # List of Bill objects
    relevant_doc_ids_list: List[Set[str]],
    judge_model: Any,
) -> Dict[str, SourceTrackingMetrics]:
    """
    Calculate source tracking metrics for multiple models.

    Args:
        results: Dictionary with model results
                 Format: {model_name: {"responses": [...], ...}}
        retrieved_docs_list: List of retrieved documents (same order for all models)
        relevant_doc_ids_list: List of relevant document ID sets
        judge_model: LLM instance for evaluation

    Returns:
        Dictionary mapping model names to SourceTrackingMetrics
    """
    source_tracking_results = {}

    for model, model_results in results.items():
        responses = model_results.get("responses", [])

        if not responses:
            logger.warning(f"No responses found for model {model}")
            continue

        # Build retrieved doc IDs list for this model
        # (Assuming same retrieval for all models, but could vary)
        retrieved_doc_ids_list = []
        for i in range(len(responses)):
            if i < len(retrieved_docs_list):
                retrieved_ids = {str(doc.name) for doc in retrieved_docs_list[i]}
            else:
                retrieved_ids = set()
            retrieved_doc_ids_list.append(retrieved_ids)

        # Truncate relevant doc IDs to match number of responses
        relevant_ids_truncated = relevant_doc_ids_list[:len(responses)]

        try:
            metrics = evaluate_source_tracking(
                responses=responses,
                retrieved_doc_ids_list=retrieved_doc_ids_list,
                relevant_doc_ids_list=relevant_ids_truncated,
                judge_model=judge_model,
            )
            source_tracking_results[model] = metrics
        except Exception as e:
            logger.error(f"Error calculating source tracking for {model}: {e}")
            continue

    return source_tracking_results
