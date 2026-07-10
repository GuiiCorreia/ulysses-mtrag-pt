"""
JurisTCU Loader - Carrega dataset JurisTCU do cache HuggingFace.

Baseado na estrutura descoberta:
- doc.csv: 16045 documentos (acórdãos TCU)
  - KEY: ID do documento
  - ENUNCIADO: Texto do acórdão (usar como txt_ementa e text)
  - TEMA, SUBTEMA, etc.: metadados

- query.csv: 150 queries
  - ID: ID da query
  - TEXT: Texto da query
  - SOURCE: Fonte da query

- qrel.csv: 2250 julgamentos de relevância
  - QUERY_ID: ID da query (de query.csv)
  - DOC_ID: ID do documento (KEY de doc.csv)
  - SCORE: Relevância (0-3 ou 1-3)
  - ENGINE: Método de busca
  - RANK: Rank de relevância

Referência: Ribeiro et al. (2025) - JurisTCU: A New Corpus for Brazilian Legal IR
"""
import os
import pandas as pd
from typing import Dict, List, Tuple

from mestrado.data.schema import Bill, Query, FeedbackItem


class JurisTCULoader:
    """Loader para o dataset JurisTCU (TCU - Tribunal de Contas da União)."""

    def __init__(self, cache_path: str = None):
        """
        Inicializa o loader.

        Args:
            cache_path: Caminho para o cache do HuggingFace JurisTCU.
                       Se None, usa o cache padrão do HuggingFace.
        """
        if cache_path is None:
            cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--LeandroRibeiro--JurisTCU")
            snapshot = "ac7bea9e580626a586ffea69b245d26f5a73d44e"
            self.cache_dir = os.path.join(cache_base, "snapshots", snapshot)
        else:
            self.cache_dir = cache_path

        print(f"JurisTCU Loader inicializado")
        print(f"Cache: {self.cache_dir}")

    def _load_doc_csv(self) -> pd.DataFrame:
        """Carrega doc.csv do cache."""
        doc_path = os.path.join(self.cache_dir, "doc.csv")
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"doc.csv não encontrado em {doc_path}")

        df = pd.read_csv(doc_path)
        print(f"✅ doc.csv carregado: {len(df)} documentos")
        return df

    def _load_query_csv(self) -> pd.DataFrame:
        """Carrega query.csv do cache."""
        query_path = os.path.join(self.cache_dir, "query.csv")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"query.csv não encontrado em {query_path}")

        df = pd.read_csv(query_path)
        print(f"✅ query.csv carregado: {len(df)} queries")
        return df

    def _load_qrel_csv(self) -> pd.DataFrame:
        """Carrega qrel.csv do cache."""
        qrel_path = os.path.join(self.cache_dir, "qrel.csv")
        if not os.path.exists(qrel_path):
            raise FileNotFoundError(f"qrel.csv não encontrado em {qrel_path}")

        df = pd.read_csv(qrel_path)
        print(f"✅ qrel.csv carregado: {len(df)} julgamentos")
        return df

    def _create_bills(self, doc_df: pd.DataFrame) -> Dict[str, Bill]:
        """Cria objetos Bill a partir do doc.csv."""
        print(f"\n📋 Criando objetos Bill ({len(doc_df)} documentos)...")

        bills = {}

        # Limpar texto HTML se presente
        def clean_html(text):
            if pd.isna(text):
                return ""
            text_str = str(text)
            # Remover tags HTML
            import re
            text_str = re.sub(r'<[^>]+>', ' ', text_str)
            # Remover espaços múltiplos
            text_str = re.sub(r'\s+', ' ', text_str)
            return text_str

        for idx, row in doc_df.iterrows():
            key = str(int(row['KEY']))  # KEY pode ser float, converter para int string

            # ENUNCIADO: sumário do acórdão (usado como ementa)
            enunciado_raw = clean_html(row['ENUNCIADO'])

            # EXCERTO: texto jurídico completo do acórdão
            # JUÁ (Pereira et al. 2025): "we kept only the EXCERTO field" para indexação JurisTCU
            excerto_raw = clean_html(row.get('EXCERTO', '') or '')

            # bill.text = ENUNCIADO + EXCERTO (campo indexado pelo BM25)
            text = (enunciado_raw + " " + excerto_raw).strip() if excerto_raw else enunciado_raw

            # Criar Bill
            # IMPORTANTE: bill.name = key (= DOC_ID em qrel.csv) para que a
            # avaliação nDCG/MRR encontre os documentos pelo mesmo ID que o feedback usa.
            bill = Bill(
                code=key,
                sig_tipo="TCU",  # Tribunal de Contas da União
                name=key,  # KEY == DOC_ID em qrel.csv; garante matching na avaliação
                txt_ementa=enunciado_raw,  # Ementa completa, sem truncamento
                text=text,
                em_tramitacao="Concluído",  # Acórdãos são decisões finais
                situacao="Julgado",
                preprocessed_tokens=[],  # BM25 usa bill.text via _tokenize_text()
            )

            bills[key] = bill

        print(f"✅ {len(bills)} objetos Bill criados")
        return bills

    def _create_queries(self, query_df: pd.DataFrame, qrel_df: pd.DataFrame) -> List[Query]:
        """Cria objetos Query a partir de query.csv e qrel.csv."""
        print(f"\n📋 Criando objetos Query ({len(query_df)} queries)...")

        queries = []

        # Criar mapa de qrels para cada query
        qrel_map = {}
        for _, row in qrel_df.iterrows():
            query_id = int(row['QUERY_ID'])
            doc_id = str(int(row['DOC_ID']))  # Converter para string consistente
            score = int(row['SCORE'])

            if query_id not in qrel_map:
                qrel_map[query_id] = []
            qrel_map[query_id].append((doc_id, score))

        # Criar Queries
        for idx, row in query_df.iterrows():
            query_id = int(row['ID'])
            text = str(row['TEXT'])

            # Buscar qrels para esta query
            feedback_items = []
            if query_id in qrel_map:
                for doc_id, score in qrel_map[query_id]:
                    # Converter score para r/pr/i
                    # JurisTCU usa escala 0-3, mapeamos para r/pr/i
                    if score >= 2:
                        label = 'r'  # relevante
                    elif score == 1:
                        label = 'pr'  # parcialmente relevante
                    else:
                        label = 'i'  # irrelevante

                    feedback = FeedbackItem(
                        bill_id=doc_id,
                        label=label,
                        score=float(score),
                        score_normalized=float(score) / 3.0,  # Normalizar para 0-1
                    )
                    feedback_items.append(feedback)

            query = Query(
                query_id=str(query_id),
                text=text,
                feedback=feedback_items,
                extra_results=[],
                date_created="",  # Não disponível
                num_doc_feedback=len(feedback_items),
            )

            queries.append(query)

        print(f"✅ {len(queries)} objetos Query criados")
        print(f"   Queries com relevante: {sum(1 for q in queries if q.has_relevant)}")

        return queries

    def load(self) -> Tuple[Dict[str, Bill], List[Query]]:
        """
        Carrega dataset JurisTCU completo.

        Returns:
            Tuple[bills, queries] onde:
                - bills: Dict[str, Bill] - Dicionário de documentos
                - queries: List[Query] - Lista de queries
        """
        print("=" * 80)
        print("CARREGANDO JURISTCU DO CACHE LOCAL")
        print("=" * 80)

        # Carregar arquivos
        doc_df = self._load_doc_csv()
        query_df = self._load_query_csv()
        qrel_df = self._load_qrel_csv()

        # Criar objetos
        bills = self._create_bills(doc_df)
        queries = self._create_queries(query_df, qrel_df)

        print("\n" + "=" * 80)
        print("JURISTCU CARREGADO COM SUCESSO")
        print("=" * 80)
        print(f"   Documentos: {len(bills)}")
        print(f"   Queries: {len(queries)}")
        print(f"   Qrels: {len(qrel_df)}")
        print("=" * 80)

        return bills, queries


# Função de conveniência
def load_juristcu(cache_path: str = None) -> Tuple[Dict[str, Bill], List[Query]]:
    """
    Carrega JurisTCU com parâmetros padrão.

    Args:
        cache_path: Caminho para o cache (opcional)

    Returns:
        Tuple[bills, queries]
    """
    loader = JurisTCULoader(cache_path=cache_path)
    return loader.load()


if __name__ == "__main__":
    # Teste do loader
    try:
        bills, queries = load_juristcu()

        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Total de documentos: {len(bills)}")
        print(f"   Total de queries: {len(queries)}")
        print(f"   Queries com relevante: {sum(1 for q in queries if q.has_relevant)}")
        print(f"   Qrels totais: {sum(len(q.feedback) for q in queries)}")

        # Mostrar exemplos
        if queries:
            print(f"\n📋 EXEMPLO DE QUERY:")
            query = queries[0]
            print(f"   ID: {query.query_id}")
            print(f"   Texto: {query.text}")
            print(f"   Relevante: {query.has_relevant}")
            print(f"   Qrels: {len(query.feedback)}")
            if query.feedback:
                print(f"   Primeiro qrel: {query.feedback[0].bill_code} ({query.feedback[0].relevance})")

        if bills:
            print(f"\n📋 EXEMPLO DE DOCUMENTO:")
            first_key = list(bills.keys())[0]
            bill = bills[first_key]
            print(f"   ID: {bill.code}")
            print(f"   Nome: {bill.name}")
            print(f"   Ementa: {bill.txt_ementa[:200]}...")
            print(f"   Texto: {bill.text[:200]}...")

    except Exception as e:
        print(f"❌ Erro ao testar loader: {e}")
        import traceback
        traceback.print_exc()
