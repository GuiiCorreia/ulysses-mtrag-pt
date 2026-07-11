# Ulysses-MTRAG: Um Benchmark Sintético Multi-Turno para Geração Aumentada por Recuperação sobre Texto Legal Brasileiro

> **Tradução para leitura** da versão de submissão (ICTAI 2026, double-blind), atualizada após a revisão de escrita de 2026-07-11. As citações aparecem como (Autor, ano). Esta tradução espelha o `main.tex` — **o texto oficial para submissão é o em inglês**; esta versão em português é só para conferência/leitura sua e da orientadora.

---

## Resumo

A geração aumentada por recuperação (RAG) para assistência jurídica é inerentemente conversacional; no entanto, os recursos existentes de recuperação de informação jurídica em português brasileiro são de turno único e param na recuperação. Apresentamos o **Ulysses-MTRAG**, um benchmark sintético de RAG multi-turno para o diálogo legislativo brasileiro. Cada sessão ancora seu primeiro turno em uma consulta real de especialista que carrega julgamentos humanos de relevância (Ulysses-RFCorpus); um LLM então gera turnos de continuação coerentes, e um juiz LLM distinto do gerador avalia a relevância dos documentos recuperados.

Usando o benchmark, respondemos a três perguntas.

**Primeira:** como a qualidade da recuperação evolui ao longo dos turnos, e o histórico do diálogo ajuda? Sob quatro estratégias de contexto e dois recuperadores, a qualidade degrada significativamente com a profundidade do turno, e a melhor estratégia é **dependente do recuperador**: a recuperação densa não melhora com o histórico em relação ao turno corrente sozinho, enquanto o BM25 lexical é significativamente melhor com o histórico completo do diálogo, que atua como expansão de consulta.

**Segunda:** a escala do modelo melhora a fidelidade da geração? Um estudo controlado por tamanho e família (de 7B a 72B parâmetros) mostra que a família do modelo domina a contagem de parâmetros: um bom modelo de 8B é mais fiel que modelos de 14B e 31B de outras famílias e competitivo com um modelo de 72B nove vezes maior, enquanto a degradação da recuperação se propaga para a geração apenas fracamente (Spearman ρ ≤ 0,21).

**Terceira:** quão confiável é o juiz LLM de relevância? Frente aos rótulos de especialistas pré-existentes, a concordância no pool completo é modesta (0,58) — dominada por documentos do pool que os especialistas nunca avaliaram — mas sobe para **0,94** no subconjunto que eles rotularam explicitamente.

Disponibilizaremos o benchmark, os prompts e o código; um repositório anonimizado acompanha esta submissão.

**Palavras-chave:** geração aumentada por recuperação, diálogo multi-turno, PLN jurídico, recuperação de informação, LLM-como-juiz, português brasileiro.

---

## I. Introdução

A busca por informação jurídica é conversacional: um analista refina uma pergunta inicial ao longo de vários turnos. A geração aumentada por recuperação (RAG) (Lewis et al., 2020) é o paradigma dominante para fundamentar esse tipo de assistência, mas os recursos de RI jurídica em português brasileiro — Ulysses-RFCorpus (Vitório et al.), JurisTCU (Fernandes et al.) e o benchmark JUÁ (Pereira et al.) — são de **turno único** e param na recuperação. O RAG jurídico multi-turno em português, até onde sabemos, ainda não foi estudado.

Atacamos essa lacuna com o **Ulysses-MTRAG**, um benchmark sintético multi-turno para o diálogo legislativo brasileiro. Em vez de transplantar uma receita existente, nossa construção mira tanto escalabilidade quanto validade ecológica: cada sessão é ancorada em uma consulta **real** de especialista que carrega julgamentos humanos de relevância, e apenas os turnos de continuação são gerados por LLM. Esse desenho ancorado em humanos distingue o Ulysses-MTRAG dos benchmarks multi-turno anteriores. O MTRAG (Katsis et al., 2025) é inteiramente escrito por humanos — preciso, porém não escalável. O CORAL (Cheng et al., 2025) deriva conversas automaticamente da Wikipédia — escalável, porém sem ground truth de relevância de especialistas. Ancorar em julgamentos de especialistas pré-existentes nos permite fazer algo que nenhum dos dois desenhos anteriores viabiliza: auditar o juiz LLM de relevância contra especialistas **dentro** do próprio benchmark. Nossas contribuições são:

- **Ulysses-MTRAG**: um benchmark de RAG jurídico multi-turno com 100 sessões de quatro turnos sobre 105.669 projetos de lei, com primeiros turnos ancorados em humanos, continuações geradas por LLM e rótulos de relevância em pool atribuídos por um juiz distinto do gerador.
- Uma **análise de recuperação multi-turno** sobre quatro estratégias de contexto e dois recuperadores, com testes pareados de significância, mostrando degradação por turno e uma interação estratégia–recuperador: o uso ótimo do histórico do diálogo se inverte entre a recuperação densa e a lexical.
- Um **estudo de geração controlado por tamanho e família** (7B–72B) mostrando que o melhor modelo de 8B supera modelos de 14B e 31B de outras famílias e é estatisticamente indistinguível de um modelo de 72B nove vezes maior. Aumentar a escala não é, portanto, um caminho confiável para RAG jurídico fiel neste domínio.
- Evidência de que **a degradação da recuperação se propaga para a geração apenas fracamente**: a fidelidade declina de forma bem menos acentuada que a qualidade da recuperação ao longo dos turnos, e se correlaciona fracamente com ela no nível do turno.
- Uma **avaliação de confiabilidade** do juiz LLM de relevância contra rótulos de especialistas pré-existentes, desemaranhando o fator de confusão da incompletude do pooling: no subconjunto de documentos que os especialistas rotularam explicitamente, a concordância bruta é 0,94.

**Questões de pesquisa.**
**RQ1**: Como a qualidade da recuperação evolui ao longo dos turnos, e estratégias de contexto cientes do histórico superam o uso apenas da consulta corrente?
**RQ2**: Quão confiável é o juiz LLM de relevância frente aos rótulos de especialistas?
**RQ3**: A escala do modelo melhora a geração fundamentada em diálogo, e a degradação da recuperação se propaga para as respostas geradas?

---

## II. Trabalhos Relacionados

**Avaliação de RAG e LLM-como-juiz.** Juízes LLM são hoje padrão para avaliar geração (Zheng et al., 2023; Liu et al., 2023; Kim et al., 2024) e RAG especificamente (Es et al., 2024), incluindo a avaliação de relevância (Thomas et al., 2024; Upadhyay et al., 2024; Faggioli et al., 2023). Eles carregam vieses conhecidos — posição (Wang et al., 2024), verbosidade (Zheng et al., 2023) e autopreferência (Panickssery et al., 2024) — o que motiva nosso uso de um juiz distinto do gerador. Rótulos de relevância também são altamente sensíveis ao prompt (Arabzadeh & Clarke, 2025); em vez de um rigor cego, adotamos uma rubrica com *reason-then-label* (Liu et al., 2023) e quantificamos a concordância com rótulos de especialistas, reportando estatísticas de concordância (Calderon et al., 2025; Hashemi et al., 2024).

**RAG multi-turno / conversacional.** Benchmarks de QA e RAG conversacionais (Katsis et al., 2025; Cheng et al., 2025; Qu et al., 2020) e reescrita conversacional de consultas (Anantha et al., 2021; Elgohary et al., 2019; Mao et al., 2023) estabelecem as estratégias de contexto que comparamos. Se o histórico ajuda é uma questão em aberto: o ORConvQA (Qu et al., 2020) encontra benefício do histórico em dados humanos, enquanto pipelines sintéticos tendem a produzir perguntas autossuficientes (Vlachos et al., 2025) — descontextualizadas no sentido de Choi et al. (2021) — uma tendência que nosso benchmark permite quantificar e testar sob estresse. A degradação em diálogos longos também está documentada (Laban et al., 2025). Em relação ao MTRAG e ao CORAL, o Ulysses-MTRAG difere em três eixos: proveniência dos turnos (semente humana + continuações sintéticas, vs. totalmente humano no MTRAG e totalmente automático no CORAL), proveniência dos rótulos de relevância (julgamentos de especialistas no turno-semente, viabilizando a auditoria do juiz dentro do benchmark) e domínio (projetos de lei em português, incluindo um cenário de língua de poucos recursos que nenhum dos dois cobre).

**Recuperação de documentos longos e chunking.** A indexação em nível de passagem de documentos longos é a mitigação canônica quando as entradas excedem os limites dos modelos (Dai & Callan, 2019; Karpukhin et al., 2020), embora a granularidade ótima da unidade de recuperação siga em debate (Jiang et al., 2024). Modelos de embedding têm limites rígidos de tokens (BGE-M3: 8.192 tokens; Chen et al., 2024) e degradam em entradas longas (Zhou et al., 2024); o BM25 permanece uma baseline forte, especialmente em domínios lexicalmente ricos (Robertson & Zaragoza, 2009; Thakur et al., 2021).

**RI jurídica em português e dados sintéticos.** Construímos sobre os recursos brasileiros de RI jurídica (Vitório et al.; Fernandes et al.; Pereira et al.; Domingos Júnior et al.; Luz de Araujo et al., 2018). A geração de dados sintéticos com LLMs está bem estabelecida para RI (Bonifácio et al., 2022; Dai et al., 2023) e como proxy de avaliação (Chiang & Lee, 2023); ancorar dados sintéticos em ground truth humano mitiga os riscos da geração totalmente recursiva (Shumailov et al., 2024).

---

## III. Configuração de Recuperação

Nossos experimentos usam dois recuperadores padrão: o lexical BM25 (Robertson & Zaragoza, 2009) e o denso BGE-M3 (Chen et al., 2024). O corpus do benchmark é composto por projetos de lei, cujos resumos oficiais (**ementas**) são curtos e cabem confortavelmente na janela do codificador. Os experimentos multi-turno, portanto, indexam cada projeto pelo resumo inteiro, sem necessidade de chunking.

**Uma fronteira de comprimento, e por que nosso benchmark a evita.** Antes da análise multi-turno, verificamos se essa pilha de recuperação tem uma limitação de comprimento em texto legal brasileiro, usando o subdomínio regulatório do JUÁ (NormasTCU; normas de ~12 mil tokens, n=46 consultas) como teste de estresse. E tem: ali, a recuperação densa de documento inteiro **colapsa** — nenhuma norma relevante aparece em nenhum top-10, resultando em nDCG@10 = 0,000 em todas as consultas — enquanto o BM25 lexical atinge 0,328 (Tabela I). O mecanismo é o comprimento: cada norma excede em muito a janela de 8.192 tokens do BGE-M3, de modo que o vetor único indexado por documento é calculado a partir de um prefixo truncado que omite as passagens relevantes à consulta (Chen et al., 2024; Zhou et al., 2024). O remédio padrão em nível de passagem — dividir cada norma em blocos (*chunks*) de menos de 8 mil tokens, indexar os blocos e mapear os blocos recuperados de volta aos documentos (Dai & Callan, 2019; Karpukhin et al., 2020) — recupera a recuperação densa até a paridade com o BM25 (nDCG@10 = 0,330; o BM25 ainda mantém recall melhor). Isso confirma que a fronteira é real e nos dá uma mitigação testada, mas o nosso próprio corpus de projetos de lei nunca se aproxima dela: os resumos dos projetos ficam bem abaixo de 8 mil tokens, então o benchmark multi-turno abaixo os indexa por inteiro, sem necessidade de chunking.

**Tabela I — Recuperação no NormasTCU (n=46).** A recuperação densa de documento inteiro colapsa em normas de ~12 mil tokens; a indexação por blocos recupera a paridade com o BM25.

| Método | nDCG@5 | nDCG@10 | R@10 | MRR |
|---|---|---|---|---|
| BM25 (texto completo) | 0,287 | **0,328** | 0,339 | 0,454 |
| Denso, documento inteiro | 0,000 | 0,000 | 0,000 | 0,000 |
| Denso, nível de bloco | 0,317 | **0,330** | 0,294 | 0,559 |

---

## IV. O Benchmark Ulysses-MTRAG

**Construção.** A construção das sessões tem três etapas: seleção de sementes, geração de turnos e controle de qualidade.

*Seleção de sementes.* Partimos das 692 consultas da versão pública do Ulysses-RFCorpus, cada uma carregando julgamentos graduados de relevância (relevante / parcialmente relevante / irrelevante) produzidos pelos consultores legislativos da Câmara dos Deputados (Vitório et al.), sobre um corpus de 105.669 projetos de lei.¹ Mantemos apenas as consultas com pelo menos três projetos (parcialmente) relevantes, embutimos os textos das consultas restantes com BGE-M3 e as agrupamos com k-means (k=100, semente fixa). Uma consulta representativa por cluster vira uma semente de sessão, resultando em um conjunto diverso e não redundante de 100 sementes; cada semente vira o Turno 1 e herda seus rótulos de relevância de especialistas.

> ¹ As contagens referem-se à versão pública que usamos (692 consultas; 105.669 projetos); o artigo do dataset reporta 693 consultas e 105.681 documentos.

*Geração de turnos.* Os Turnos 2–4 são gerados por um LLM instruído (Qwen2.5-72B), condicionado ao histórico completo da sessão e instruído a produzir uma próxima pergunta coerente de um tipo **prescrito** — continuação (*follow-up*), esclarecimento, não autossuficiente (*non-standalone*), comparativa ou temporal. Desenhamos esse esquema de cinco tipos em torno dos desafios conversacionais que o MTRAG enfatiza: perguntas não autossuficientes e comparativas, e respondibilidade (Katsis et al., 2025).

*Controle de qualidade.* O gerador é instruído a manter o tópico e não repetir turnos anteriores. Descartamos gerações degeneradas — vazias, duplicadas ou fora de tópico — e as regeneramos. O resultado são 100 sessões de quatro turnos (400 turnos no total); a Fig. 1 mostra uma delas. Um gerador especializado em português (p.ex., Gaia/Sabiá) fica como trabalho futuro.

**Tabela II — Composição do Ulysses-MTRAG:** distribuições realizadas de tipo de turno e tipo de resposta sobre os 300 turnos gerados (T2–T4). Os tipos de turno são prescritos por amostragem no momento da geração; os tipos de resposta são os rotulados pelo gerador.

| Tipo de turno | % | Tipo de resposta | % |
|---|---|---|---|
| continuação | 47,0 | respondível | 99,3 |
| esclarecimento | 20,3 | parcial | 0,7 |
| não autossuficiente | 11,3 | não respondível | 0,0 |
| comparativa | 11,0 | | |
| temporal | 10,3 | | |

A Tabela II reporta a composição realizada. Como a distribuição de tipos é um insumo da construção (amostrada por desenho), não a lemos como um achado; o que **é** informativo é que apenas 11,3% dos turnos gerados são explicitamente não autossuficientes — de modo que o benchmark, como outros pipelines sintéticos de QA conversacional (Vlachos et al., 2025), consiste majoritariamente de perguntas autossuficientes (descontextualizadas; Choi et al., 2021) sobre um tópico compartilhado. Tratamos isso como uma propriedade mensurável da receita de construção sintética e testamos suas consequências sob estresse na Seção VI (RQ1), incluindo uma análise estratificada por tipo de turno. Quase todos os turnos são respondíveis (99,3%); a quase ausência de turnos não respondíveis é uma limitação que discutimos na Seção VII.

**Rótulos de relevância.** Para cada turno, os documentos recuperados por todas as estratégias de contexto são reunidos em um pool e rotulados uma única vez por um juiz LLM (Llama-3.3-70B), **distinto do gerador** para evitar viés de autopreferência (Panickssery et al., 2024), usando uma rubrica graduada com *reason-then-label* (Liu et al., 2023). **Todos** os escores de recuperação deste artigo, incluindo o Turno 1, usam esses rótulos do juiz, de modo que as comparações entre turnos repousam sobre uma única fonte de rótulos; os rótulos de especialistas do Turno 1 são reservados para auditar o próprio juiz (Seção VI, RQ2). Rótulos graduados são binarizados em 0,5 (parcialmente relevante conta como relevante) sempre que a concordância binária é computada.

**Fig. 1 (exemplo trabalhado)** — Uma sessão de quatro turnos semeada por uma consulta real do Ulysses-RFCorpus (T1, que adicionalmente carrega rótulos de especialistas), com continuações geradas por LLM (T2–T4). Para cada turno mostramos o projeto de lei mais bem recuperado e seu rótulo de relevância do juiz LLM. As consultas são reproduzidas verbatim no português original, incluindo erros de digitação presentes no corpus-fonte. O projeto recuperado **muda** ao longo dos turnos e degrada para apenas **parcialmente relevante** no T4.

- **T1** *(semente, rótulo humano)*: "Criação de fundo para prevenção de desastres naturais e e recuperação de seus efeitos, para…" → top: **PL 295/2007** — *relevante*
- **T2** *(continuação)*: "Quais são os principais critérios e mecanismos de distribuição dos recursos do fundo para…" → top: **PL 5067/2016** — *relevante*
- **T3** *(esclarecimento)*: "Em quais situações específicas os recursos do fundo para prevenção de desastres naturais e…" → top: **PL 10898/2018** — *relevante*
- **T4** *(esclarecimento)*: "Como são definidas as prioridades para a alocação de recursos do fundo em situações de…" → top: **PL 4450/2020** — *parcialmente relevante*

---

## V. Configuração da Avaliação Multi-Turno

**Recuperação.** Para cada turno recuperamos os top-k (k=10) projetos sob quatro estratégias de contexto: **last-turn** (apenas a consulta corrente), **concat** (consultas anteriores pré-anexadas), **full-history** (todos os turnos anteriores) e **rewrite** (um modelo Llama-3.1-8B reescreve a consulta para a forma autossuficiente) (Mao et al., 2023; Elgohary et al., 2019). Por turno e por recuperador, reunimos em pool os documentos recuperados pelas quatro estratégias e julgamos o pool uma única vez (pooling padrão), pontuando cada estratégia com nDCG@10 sobre os rótulos compartilhados. Um cache persistente de julgamentos atribui rótulos idênticos a qualquer par (consulta, documento) que apareça nos pools de ambos os recuperadores. Rodamos a comparação completa de estratégias sob **ambos** os recuperadores — denso BGE-M3 e lexical BM25 (Seção III) — para testar se as conclusões sobre o histórico do diálogo são específicas do recuperador.

**Geração.** Para testar se a degradação da recuperação se propaga para as respostas, os top-5 projetos recuperados via last-turn de cada turno são passados, com um prompt ciente do histórico, a cinco modelos de linguagem pequenos (SLMs) de pesos abertos e instruídos. Escolhemos esses cinco para cobrir a faixa de tamanhos com qualidade competitiva: Qwen2.5-7B, Qwen3-8B, Ministral-8B, Phi-4 (14B) e Gemma-4-31B. Esse espalhamento permite que o efeito da escala seja lido diretamente da comparação. Pontuamos cada resposta em três eixos adaptados da avaliação de geração do MTRAG (Katsis et al., 2025): (i) **fidelidade ao contexto** (sem referência, estilo RAGAS; Es et al., 2024), julgada pelo Llama-3.3-70B; (ii) **similaridade à referência** — uma resposta-oráculo — via BERTScore-F1 (Zhang et al., 2020) e ROUGE-L (Lin, 2004); e (iii) uma comparação por juiz LLM contra a resposta do oráculo. O oráculo (Qwen2.5-72B) recebe o **mesmo** contexto recuperado e histórico que os modelos avaliados. Dois fatos de desenho precisam ser ditos abertamente: o oráculo é o mesmo modelo que gerou as perguntas de continuação, e ele compartilha família com dois modelos avaliados. O juiz é distinto tanto dos geradores quanto do oráculo, e nossas afirmações principais repousam sobre o eixo de fidelidade sem referência, que não usa o oráculo; empiricamente, as métricas baseadas em referência tampouco favorecem a família do oráculo (o Phi-4 lidera ambas, Seção VI).

**Implementação.** A geração usa temperatura 0,1 (máx. 1.024 tokens); o julgamento usa temperatura 0. A recuperação densa usa embeddings BGE-M3 com FAISS (Douze et al., 2024); o BM25 é o BM25Okapi (k₁=1,5, b=0,75); o chunking do NormasTCU usa blocos de 3.000 caracteres com sobreposição de 400; o agrupamento de sementes usa k-means com semente aleatória fixa. O BERTScore usa BERT multilíngue. Os modelos são acessados por APIs compatíveis com OpenAI. Todos os prompts (geração de turnos, reescrita de consultas, rubrica de relevância, julgamento de fidelidade), identificadores de modelos e arquivos de configuração estão incluídos no repositório anonimizado que acompanha esta submissão.

---

## VI. Resultados

**RQ1 — degradação por turno e a interação estratégia–recuperador.**
A Tabela III reporta o nDCG@10 por recuperador, estratégia e turno; as estratégias coincidem no T1, onde não existe histórico. Sob ambos os recuperadores, a qualidade da recuperação **declina significativamente conforme a conversa se aprofunda**: para o last-turn denso, o nDCG@10 cai de 0,876 (semente) para 0,617 no T3 (Δ=0,26; Wilcoxon pareado p<10⁻¹²; Fig. 2). A maior queda isolada (T1→T2) coincide com a troca de consultas-semente escritas por humanos para continuações sintéticas, de modo que lemos a degradação com a profundidade primariamente no declínio contínuo dentro dos turnos sintéticos.

**Qual estratégia de contexto é melhor depende do recuperador.** Para o denso BGE-M3, nenhuma estratégia ciente do histórico supera o last-turn (médias dentro de ≈0,02; last-turn é indistinguível de concat e full-history, Holm p≥0,17, e melhor que rewrite, p_adj=0,023): concatenar o histórico dilui o embedding único da consulta. Para o BM25 lexical o ranking **se inverte** — full-history é a melhor e last-turn a **pior** estratégia (0,741 vs. 0,668 de média geral; diferença de +0,097 em T2–T4, Wilcoxon pareado p<10⁻⁷) — porque anexar os turnos anteriores atua como expansão de consulta, fornecendo termos jurídicos lexicalmente correspondentes. Usar apenas o turno corrente é, portanto, um padrão seguro para a recuperação densa, mas uma escolha ruim para a recuperação lexical, onde vale o oposto. Essa interação alerta contra tirar conclusões sobre contexto conversacional a partir de um único recuperador.

Estratificar os resultados densos por tipo de turno localiza o resultado nulo do last-turn: turnos explicitamente não autossuficientes são os mais difíceis (last-turn nDCG@10 0,52 vs. 0,64–0,73 para os demais tipos) e as estratégias densas cientes do histórico não os recuperam (concat 0,40; full-history 0,43; rewrite 0,40), enquanto o full-history ajuda apenas modestamente em turnos de esclarecimento (0,69 vs. 0,66) e temporais (0,68 vs. 0,64) — consistente com as perguntas autossuficientes que pipelines sintéticos sabidamente produzem (Vlachos et al., 2025), descontextualizadas no sentido de Choi et al. (2021); em contraste com o ORConvQA (Qu et al., 2020), onde o histórico ajuda em diálogo humano. Uma ressalva: os documentos do pool são julgados contra o texto do próprio turno, o que pode favorecer o last-turn justamente nos turnos não autossuficientes; julgamento ciente do histórico é trabalho futuro.

**Tabela III — Recuperação multi-turno (nDCG@10) por recuperador, estratégia de contexto e posição do turno** (100 conversas; por recuperador, um único pool julgado por LLM pontua todas as estratégias por turno, incluindo o T1 — por isso as estratégias coincidem no T1). Melhor estratégia por recuperador (média geral) em negrito. A ordenação **se inverte** entre recuperação densa e lexical.

| Recup. | Estratégia | T1 | T2 | T3 | T4 | média |
|---|---|---|---|---|---|---|
| BGE-M3 | last-turn | 0,876 | 0,727 | 0,617 | 0,636 | **0,714** |
| BGE-M3 | rewrite | 0,876 | 0,718 | 0,587 | 0,604 | 0,696 |
| BGE-M3 | concat | 0,876 | 0,718 | 0,621 | 0,557 | 0,693 |
| BGE-M3 | full-history | 0,876 | 0,695 | 0,605 | 0,597 | 0,693 |
| BM25 | full-history | 0,900 | 0,755 | 0,670 | 0,637 | **0,741** |
| BM25 | concat | 0,900 | 0,731 | 0,624 | 0,449 | 0,676 |
| BM25 | rewrite | 0,900 | 0,652 | 0,587 | 0,564 | 0,676 |
| BM25 | last-turn | 0,900 | 0,641 | 0,580 | 0,551 | 0,668 |

**Fig. 2** — nDCG@10 de recuperação por turno para as quatro estratégias de contexto, sob recuperação densa (esquerda) e lexical (direita) (100 conversas). A qualidade cai após o turno-semente em ambos, mas a melhor estratégia **se inverte**: last-turn lidera nos embeddings densos; full-history lidera no BM25.

**RQ2 — confiabilidade do juiz frente aos rótulos de especialistas.**
Auditamos o juiz LLM contra os rótulos de especialistas pré-existentes no pool do Turno 1 da execução densa, reportando a concordância bruta junto do κ de Cohen (Cohen, 1960) e do Kappa Ajustado por Prevalência e Viés (PABAK) (Byrt et al., 1993), que corrige o κ para prevalência assimétrica das classes. Sobre o pool completo (n=1.000 pares consulta–documento), a concordância é modesta (bruta 0,58; κ=0,27): o juiz rotula 69% dos documentos do pool como relevantes, enquanto os rótulos esparsos dos especialistas cobrem bem menos (29%) — exatamente o padrão esperado quando um recuperador traz à tona documentos topicamente relevantes que os anotadores originais nunca avaliaram, e a convenção de pooling os conta como irrelevantes (Voorhees, 2000). O fator de confusão é testável: restringindo a comparação aos n=305 documentos do pool que os especialistas rotularam **explicitamente** (incluindo o "irrelevante" explícito), a concordância bruta sobe para **0,938** (PABAK 0,875). Nesse subconjunto o κ permanece modesto (0,39) porque 95% dos documentos explicitamente rotulados são relevantes — o clássico paradoxo do kappa de alta concordância sob prevalência assimétrica (Feinstein & Cicchetti, 1990; Byrt et al., 1993). Tomadas em conjunto, as duas visões indicam que o juiz acompanha de perto os julgamentos de especialistas onde a opinião de especialistas é conhecida, e que a discordância no pool completo é dominada pela incompletude dos rótulos, não por erro do juiz; o julgamento de relevância também varia entre os próprios avaliadores humanos (Voorhees, 2000). Uma anotação humana dedicada sobre o pool completo recuperado permanece um trabalho futuro desejável (Seção VII).

**RQ3 — a escala ajuda a geração, e a degradação se propaga?**
A Tabela IV reporta a qualidade de geração por modelo com as contagens de parâmetros. **A família do modelo domina a contagem de parâmetros.** O modelo aberto mais fiel é o Qwen3-8B (0,808; IC 95% [0,778, 0,837]), que supera significativamente o Phi-4 de 14B (0,762) e o Gemma-4 de 31B (0,725; Δ pareado = 0,083; Wilcoxon p<10⁻⁵): escalar para um modelo maior de uma família **diferente** não ajuda aqui. A escala ajuda **dentro** de uma família, mas com retornos fortemente decrescentes: entre os modelos Qwen a fidelidade sobe 0,781 (7B) → 0,808 (8B) → 0,841 (72B, o oráculo), mas apenas o salto completo de 10× é significativo (7B→72B, p=0,001); o passo de 9× de 8B para 72B não é (+0,033; p=0,074), e o 72B ainda responde a perguntas que ele próprio escreveu, uma vantagem que infla seu escore. Na prática, um modelo de 8B bem escolhido é, portanto, competitivo com um modelo de 72B nove vezes maior; a contagem de parâmetros sozinha prediz a fidelidade de forma bem menos confiável do que a família e o treinamento do modelo. Fidelidade e similaridade à referência também se desacoplam: o Qwen3-8B é o mais fiel, mas o Phi-4 é o mais próximo da resposta do oráculo (BERTScore 0,830; ROUGE-L 0,472) — e como o Phi-4 não é da família do oráculo, o viés de família não domina as métricas baseadas em referência. O terceiro eixo planejado, uma comparação por juiz LLM contra a resposta do oráculo, não discriminou entre os modelos (todos ≈0,50); reportamos isso como um achado negativo — avaliar em uma única escala a equivalência de duas respostas jurídicas de forma livre é grosseiro demais — e o excluímos da Tabela IV.

A degradação da recuperação **se propaga para a geração, mas fracamente**. A fidelidade média cai de 0,802 (T1) para 0,751 (T4) e o ROUGE-L de 0,434 para 0,358; o declínio é significativo para os dois modelos mais fiéis (Qwen3-8B −0,113, p=0,006; Qwen2.5-7B −0,101, p=0,008) e plano para os demais (p>0,27), de modo que os líderes convergem para o pelotão conforme o diálogo se aprofunda (Fig. 3). Correlacionar diretamente o nDCG@10 de recuperação por turno com as métricas de geração sobre as 2.000 unidades (turno, modelo) produz associações positivas fracas (Spearman ρ=0,16 para fidelidade; 0,21 para ROUGE-L): o aterramento é em grande parte preservado mesmo quando a recuperação degrada. Registramos o limite interpretativo dessa métrica: a fidelidade recompensa o aterramento em **qualquer** contexto que tenha sido recuperado, então ela mede controle de alucinação, não utilidade da resposta — uma resposta fiel a projetos irrelevantes pontua alto. A similaridade à referência cobre parcialmente essa lacuna, mas como o oráculo responde a partir do mesmo contexto degradado, os declínios de ROUGE-L entre turnos refletem em parte a deriva da própria referência. Por fim, repetir a rodada de julgamento de fidelidade desloca os escores absolutos em até ≈0,05 preservando o ranking dos modelos; lemos, portanto, todos os escores de juiz comparativamente.

**Tabela IV — Qualidade de geração multi-turno** (100 conversas, 400 turnos por modelo): fidelidade ao contexto sem referência e similaridade à referência-oráculo (BERTScore-F1, ROUGE-L). Melhor por coluna (excluindo o oráculo) em negrito; ICs bootstrap de 95% na fidelidade estão dentro de ±0,03. †Modelo-oráculo (escreveu as continuações e serve de referência de similaridade), mostrado como âncora de escala da mesma família — apenas fidelidade.

| Modelo | Params | Fidel. | BERTScore | ROUGE-L |
|---|---|---|---|---|
| Qwen3-8B | 8B | **0,808** | 0,800 | 0,398 |
| Qwen2.5-7B | 7B | 0,781 | 0,786 | 0,398 |
| Phi-4 | 14B | 0,762 | **0,830** | **0,472** |
| Ministral-8B | 8B | 0,752 | 0,762 | 0,271 |
| Gemma-4-31B | 31B | 0,725 | 0,796 | 0,372 |
| Qwen2.5-72B† | 72B | 0,841 | — | — |

**Fig. 3** — Fidelidade da geração por turno para os cinco modelos de linguagem pequenos (100 conversas). Os modelos mais fiéis declinam significativamente e convergem para o pelotão, enquanto a recuperação cai de forma bem mais acentuada (Fig. 2).

---

## VII. Discussão e Limitações

**O que o benchmark mede.** O Ulysses-MTRAG quantifica uma propriedade da construção sintética de benchmarks multi-turno: as continuações são majoritariamente autossuficientes (11,3% não autossuficientes, por desenho), de modo que a recuperação ciente do histórico não consegue mostrar seu valor — nossa análise estratificada confirma que o histórico ajuda apenas em turnos de esclarecimento e temporais. Apresentamos isso como um achado diagnóstico sobre pipelines sintéticos multi-turno em um domínio jurídico de poucos recursos. Benchmarks que pretendam estressar o contexto conversacional devem impor uma cota maior de turnos não autossuficientes na geração; deixamos três extensões como trabalho futuro: (a) esse esquema de geração com cota maior, (b) a elicitação de turnos não respondíveis (99,3% dos nossos são respondíveis), que o MTRAG (Katsis et al., 2025) usa para testar a abstenção, e (c) uma anotação humana dedicada do pool completo recuperado para estreitar o limite de confiabilidade do juiz da RQ2.

**Ameaças à validade.**
- **(i)** Avaliamos dois recuperadores (denso e lexical) e encontramos que o ranking de estratégias se inverte entre eles; a recuperação híbrida e outros codificadores densos podem se comportar de forma diferente novamente, então os rankings específicos não devem ser generalizados além das duas famílias testadas.
- **(ii)** O gerador de continuações, o oráculo e dois modelos avaliados são aparentados (mesmo modelo ou família). Mitigamos isso apoiando as afirmações principais no eixo de fidelidade sem referência, julgado por um modelo de outra família, e observando que um modelo de fora da família lidera as métricas baseadas em referência.
- **(iii)** A comparação entre famílias tem um modelo por ponto de tamanho, então família e tamanho estão parcialmente confundidos. Adicionamos uma escada intra-família Qwen (7B/8B/72B) para isolar a escala, mas a âncora de 72B também é a autora das perguntas, uma vantagem que não conseguimos remover por completo.
- **(iv)** Escores baseados em juiz variam entre rodadas de julgamento (≈0,05 em valor absoluto, embora o ranking seja estável) e são julgados por LLM sobre um pool de recuperação; valores absolutos devem, portanto, ser lidos como rankings comparativos, com a confiabilidade delimitada na RQ2.
- **(v)** O benchmark cobre um corpus legislativo, sessões de quatro turnos e continuações sintéticas; diálogo jurídico escrito por humanos pode se comportar de forma diferente (Qu et al., 2020).

---

## VIII. Conclusão

Apresentamos o Ulysses-MTRAG, um benchmark sintético multi-turno de RAG ancorado em humanos para o diálogo legislativo brasileiro. A recuperação degrada significativamente com a profundidade do diálogo, e a melhor forma de usar o histórico do diálogo mostrou-se dependente do recuperador: inerte para a recuperação densa, mas um ganho significativo para o BM25 lexical. A fidelidade da geração é governada mais pela família do modelo do que pela escala — um modelo de 8B bem escolhido é competitivo com um modelo de 72B nove vezes maior — e a degradação da recuperação se propaga para as respostas geradas apenas fracamente. Por fim, o juiz LLM de relevância concorda com os rótulos de especialistas em 94% dos documentos que os especialistas avaliaram explicitamente, com a discordância no pool completo dominada pela incompletude dos rótulos, e não por erro do juiz. O benchmark, os prompts e o código serão disponibilizados; um repositório anonimizado acompanha a revisão.

---

## Notas de terminologia (para leitura)

Termos técnicos mantidos em inglês no paper e seus sentidos:

| Termo no paper | Sentido |
|---|---|
| *last-turn* | estratégia que busca usando **só a pergunta do turno atual** |
| *concat* | busca com as perguntas anteriores **concatenadas** à atual |
| *full-history* | busca usando **todo o histórico** da conversa |
| *rewrite* | um LLM **reescreve** a pergunta para forma autossuficiente antes de buscar |
| *non-standalone* | turno que **não se sustenta sozinho** (depende do contexto anterior) |
| *pooling* | juntar os documentos recuperados por todas as estratégias num único conjunto a julgar |
| *faithfulness* (fidelidade) | a resposta se apoia **somente** no contexto recuperado (anti-alucinação) |
| *oracle* (oráculo) | resposta de um modelo forte usada como **referência** de comparação |
| *judge* (juiz) | LLM que atribui os rótulos de relevância/fidelidade |
| *kappa paradox* | κ baixo **apesar** de concordância alta, quando quase tudo é de uma classe só |
| *PABAK* | kappa ajustado por prevalência e viés (corrige o paradoxo acima) |
| nDCG@10 | qualidade do **ranking** dos 10 primeiros resultados (1,0 = ideal) |

---

## Referências

*(As 49 obras citadas no texto, em ordem alfabética; títulos mantidos no original. Todas as 52 entradas do `.bib` — 49 citadas + 3 de apoio — foram verificadas na fonte em 2026-07-11: arXiv / ACL Anthology / DOI / Springer. Zero fabricações.)*

1. Abdin, M. et al. "Phi-4 Technical Report." arXiv:2412.08905, 2024.
2. Anantha, R. et al. "Open-Domain Question Answering Goes Conversational via Question Rewriting." *NAACL*, 2021.
3. Arabzadeh, N.; Clarke, C. L. A. "A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment." *SIGIR*, 2025. arXiv:2504.12408.
4. Bonifácio, L. et al. "InPars: Data Augmentation for Information Retrieval using Large Language Models." *SIGIR*, 2022.
5. Byrt, T.; Bishop, J.; Carlin, J. B. "Bias, prevalence and kappa." *Journal of Clinical Epidemiology*, 46(5):423–429, 1993.
6. Calderon, N.; Reichart, R.; Dror, R. "The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs." *ACL*, 2025.
7. Chen, J. et al. "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." *Findings of ACL*, 2024.
8. Cheng, Y. et al. "CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmented Generation." *Findings of NAACL*, 2025.
9. Chiang, C.-H.; Lee, H.-y. "Can Large Language Models Be an Alternative to Human Evaluations?" *ACL*, 2023.
10. Choi, E. et al. "Decontextualization: Making Sentences Stand-Alone." *TACL*, 9, 2021.
11. Cohen, J. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement*, 20(1):37–46, 1960.
12. Dai, Z.; Callan, J. "Deeper Text Understanding for IR with Contextual Neural Language Modeling." *SIGIR*, 2019.
13. Dai, Z. et al. "Promptagator: Few-shot Dense Retrieval From 8 Examples." *ICLR*, 2023.
14. Douze, M. et al. "The Faiss Library." arXiv:2401.08281, 2024.
15. Elgohary, A.; Peskov, D.; Boyd-Graber, J. "Can You Unpack That? Learning to Rewrite Questions-in-Context." *EMNLP-IJCNLP*, 2019.
16. Es, S. et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *EACL System Demonstrations*, 2024.
17. Faggioli, G. et al. "Perspectives on Large Language Models for Relevance Judgment." *ICTIR*, 2023.
18. Feinstein, A. R.; Cicchetti, D. V. "High agreement but low kappa: I. The problems of two paradoxes." *Journal of Clinical Epidemiology*, 43(6):543–549, 1990.
19. Fernandes, L. C. et al. "JurisTCU: A Brazilian Portuguese Information Retrieval Dataset with Query Relevance Judgments." *Language Resources and Evaluation*, 2026. arXiv:2503.08379.
20. Grattafiori, A. et al. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.
21. Hashemi, H. et al. "LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts." *ACL*, 2024.
22. Jiang, Z.; Ma, X.; Chen, W. "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs." arXiv:2406.15319, 2024.
23. Domingos Júnior, J. et al. "BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law." arXiv:2505.15916, 2025.
24. Karpukhin, V. et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020.
25. Katsis, Y. et al. "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems." *TACL*, 13, 2025.
26. Kim, S. et al. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." *EMNLP*, 2024.
27. Laban, P. et al. "LLMs Get Lost In Multi-Turn Conversation." arXiv:2505.06120, 2025.
28. Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.
29. Lin, C.-Y. "ROUGE: A Package for Automatic Evaluation of Summaries." *Text Summarization Branches Out (ACL Workshop)*, 2004.
30. Liu, Y. et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." *EMNLP*, 2023.
31. Luz de Araujo, P. H. et al. "LeNER-Br: A Dataset for Named Entity Recognition in Brazilian Legal Text." *PROPOR*, 2018.
32. Mao, K. et al. "Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search." *Findings of EMNLP*, 2023.
33. Panickssery, A.; Bowman, S. R.; Feng, S. "LLM Evaluators Recognize and Favor Their Own Generations." *NeurIPS*, 2024.
34. Pereira, J. et al. "JUÁ — A Benchmark for Information Retrieval in Brazilian Legal Text Collections." arXiv:2604.06098, 2026.
35. Qu, C. et al. "Open-Retrieval Conversational Question Answering." *SIGIR*, 2020.
36. Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, 2024.
37. Robertson, S.; Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009.
38. Shumailov, I. et al. "AI models collapse when trained on recursively generated data." *Nature*, 631:755–759, 2024.
39. Thakur, N. et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS Datasets and Benchmarks*, 2021.
40. Thomas, P. et al. "Large Language Models can Accurately Predict Searcher Preferences." *SIGIR*, 2024.
41. Upadhyay, S. et al. "UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor." arXiv:2406.06519, 2024.
42. Vitório, D. et al. "Building a relevance feedback corpus for legal information retrieval in the real-case scenario of the Brazilian Chamber of Deputies." *Language Resources and Evaluation*, 59:1257–1277, 2025.
43. Vlachos, C. et al. "Building Open-Retrieval Conversational Question Answering Systems by Generating Synthetic Data and Decontextualizing User Questions." *SIGDIAL*, 2025. arXiv:2507.04884.
44. Voorhees, E. M. "Variations in relevance judgments and the measurement of retrieval effectiveness." *Information Processing & Management*, 36(5):697–716, 2000.
45. Wang, P. et al. "Large Language Models are not Fair Evaluators." *ACL*, 2024.
46. Yang, A. et al. "Qwen3 Technical Report." arXiv:2505.09388, 2025.
47. Zhang, T. et al. "BERTScore: Evaluating Text Generation with BERT." *ICLR*, 2020.
48. Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS Datasets and Benchmarks*, 2023.
49. Zhou, Y. et al. "Length-Induced Embedding Collapse in PLM-based Models." *ACL*, 2025.
