# Roadmap de pesquisas por etapa — reject inference (v3.0.0)

**Fonte única (single source of truth) deste roadmap.** Este arquivo, versionado no
repositório em `docs/reject_inference_roadmap.md`, é a versão canônica. Não deve existir
cópia paralela mantida em paralelo fora do repo. Achados novos entram aqui, via commit,
com data e motivo — o histórico do Git é o registro de mudanças.

Registro de ferramentas e técnicas a pesquisar/aplicar em cada etapa futura do projeto de
crédito, com a justificativa de POR QUE estão aqui. O objetivo deste arquivo: se a
informação sumir do contexto de uma conversa, este registro faz a gente reconhecer na hora
que existe uma prática/ferramenta a considerar, em vez de reinventar ou esquecer.

**Regra de uso**: nenhum item aqui é "feito". São candidatos com gatilho de fase. A
pesquisa de cada um é disparada quando a fase correspondente abrir, com resultado
atualizado (não de memória), respeitando a régua "a ferramenta só entra se o problema
exigir".

**Este documento não é um entregável** (Model Card, Data Card, README). É um registro de
trabalho, no espírito de `scope.md` e `cleaning_decisions.md`.

## Origem deste registro

Surgiu durante a Fase 1, ao tratar uma linha malformada (1 em 27.648.741) no CSV de
recusados. A investigação levantou que existe uma categoria de ferramentas para qualidade
de dados que não estávamos considerando, e o Mateus pediu para guardar com justificativa.
Ao registrar, ampliou-se para as demais etapas onde cada uma tem boas práticas próprias
que ainda NÃO foram pesquisadas.

## 1. Ingestão / qualidade de dados — PARCIALMENTE pesquisado (Fase 1/2)

**Estado**: pesquisado na Fase 1. Aplicado agora (Célula 2): parsing robusto do Spark
(`multiLine`, `quote`, `escape`) + modo `PERMISSIVE` com coluna `_corrupt_record` como
rede auditável. Isso resolve o problema de AGORA (1 linha malformada) sem ferramenta
extra.

Candidato para a Fase 2 (validação declarativa de qualidade):

- **PyDeequ / Deequ** (AWS, nativo em Spark) — preferido para o caso dos recusados, por
  ser Spark-nativo e casar com o volume (27,6M linhas). Validação baseada em restrições e
  métricas sobre DataFrame Spark.
- **Great Expectations (GX)** — Python-nativo, integra com Spark/pandas/SQL, gera Data
  Docs (documentação de qualidade legível). Casa com o estilo do projeto (Data Card, Model
  Card, FACTS). Bom se quisermos documentação de qualidade automática.
- **Pandera** — mais leve, validação de schema dentro do script Python, estilo teste
  unitário de dataframe. Bom para checagens rápidas em pandas/Polars/PySpark.

**Justificativa para NÃO usar agora**: rodar um framework de validação para 1 linha em
27,6M seria bazuca em mosquito e violaria a régua da ferramenta justificada. Entra quando
o pipeline de ingestão dos recusados estiver de pé na Fase 2, como suíte de validação
sobre a ingestão — vira linha de currículo honesta e complementa o monitor de PSI/drift
que o projeto de crédito já tem ("validação declarativa de qualidade + PSI/drift" é dupla
forte para um time de crédito).

**Mapa mental de escolha** (da pesquisa): GX é Python-nativo; dbt tests são
warehouse-nativos; Soda Core é YAML-first e multi-fonte; pandera e Deequ são
especialistas. Padrão de mercado: dbt para testes na transformação, GX para validação de
dados brutos na ingestão, Soda para monitoramento contínuo em produção.

**A pesquisar quando abrir a Fase 2**: implementação concreta do escolhido (provável
PyDeequ), não só a categoria. Versão atual, integração com Databricks, exemplo de suíte.

## 2. Reject inference (técnica central) — NÃO pesquisado (Fase 2)

**Estado**: categoria conhecida do plano (augmentation, reclassification, parcelling,
fuzzy), mas as técnicas, suas armadilhas e a literatura atual NÃO foram pesquisadas.

**Justificativa para registrar**: é o coração da v3.0.0 e tem literatura própria com
armadilhas conhecidas (ex.: reject inference mal feito pode só reforçar o viés do modelo
original em vez de corrigi-lo). Merece pesquisa dedicada e atual, não decisão de memória.

**Gatilho**: abrir a Fase 2, DEPOIS de ter o perfil do `Risk_Score` por ano da Fase 1 (o
perfil muda o desenho do thin model — se o Risk_Score for esparso em faixas de anos, isso
afeta quais features o thin model pode usar).

**A pesquisar**: estado atual das técnicas de reject inference em crédito, comparação
honesta entre elas, e como validar que a inferência ajudou em vez de só propagar viés.

## 3. Validação/comparação de modelos com viés de seleção — NÃO pesquisado (Fase 3)

**Estado**: não pesquisado.

**Justificativa para registrar**: a Fase 3 compara o modelo com reject inference contra o
v2.0.0 por lucro. Mas as populações diferem (um viu só aprovados, o outro incorpora
inferência sobre recusados). Comparar isso de forma honesta é questão metodológica própria
— não dá para usar a mesma comparação de lucro cega às diferenças de população.

**Gatilho**: abrir a Fase 3.

**A pesquisar**: métricas e desenhos de avaliação apropriados quando as populações de
treino/avaliação diferem por viés de seleção; como reportar o ganho sem superestimar.

## Números confirmados na Fase 1 (para atualizar os documentos de decisão)

- **População recusada**: 27.648.741 linhas (medido). Corrige o `[CONFIRMAR] ~27M` dos
  docs.
- **Proporção recusados : aprovados** = 27.648.741 : 673.314 = **~41x** (corrige o "~12x"
  que aparecia em documento de decisão — estava errado por mais de 3x).
- **Arquivo bruto**: `rejected_2007_to_2018Q4.csv.gz`, 255.470.782 bytes (~243,6 MiB),
  Kaggle `wordsforthewise/lending-club`, CC0-1.0.
- **Colunas reais (9)**: Amount Requested, Application Date, Loan Title, Risk_Score,
  Debt-To-Income Ratio, Zip Code, State, Employment Length, Policy Code. (Nota: NÃO traz
  FICO nem grade — o desenho do thin model precisa considerar isso.)
- **Qualidade de parsing**: 1 linha malformada, recuperada por parsing robusto; 0
  corrompidas após o ajuste (auditado via `_corrupt_record` — checagem final ainda
  pendente de rodar sem erro, ver nota abaixo).
- **Ambiente**: Databricks Community Edition foi aposentado no fim de 2025; substituído
  pelo Databricks Free Edition (serverless, gratuito). Os docs que citam "Community
  Edition" precisam trocar para "Free Edition".

**Nota de estado (não confirmada dos números acima)**: a checagem `_corrupt_record` da
Célula 2 ainda não completou sem erro — bateu em `UNSUPPORTED_FEATURE.QUERY_ONLY_CORRUPT_RECORD_COLUMN`
do Spark (precisa materializar/`cache()` o DataFrame antes da query de auditoria). O "0
corrompidas" acima é o resultado esperado com base no diagnóstico de leitura (Etapa
anterior, script `scratch_reject_parsing_diagnostic.py`), não ainda uma reexecução
confirmada dentro do notebook principal.

**Pendente de medição na Fase 1** (ainda não rodou): perfil do `Risk_Score` por ano (a
hipótese é que seja irregular; a Célula 5 mede), volume por safra, e o comparativo
aprovados vs recusados (Célula 6).

## Achados de pesquisa retroativa (feita durante a Fase 1, antes de propagar)

Motivo desta seção: o Mateus pediu pesquisa retroativa para pegar erros antes de
propagarem para a Fase 2. Três achados, confirmados por fonte independente (estudo que
usou os mesmos dois datasets Lending Club + dicionário de dados oficial via Kaggle
jonchan2003).

### A (ATUALIZADO com medição por ano) — risk_score: ausência é TEMPORAL, não uniforme

Medido na Fase 1 (Célula 5, DuckDB): risk_score presente em 33,1% dos recusados no
agregado (66,9% ausente), MAS a cobertura NÃO é uniforme no tempo:

- 86% presente até 2014
- despenca para 17,85% em 2015
- ~54% em 2017
- 6,83% em 2018

Isto é mais crítico que o "70% ausente" agregado (a estimativa de fonte independente que
motivou este achado originalmente). Duas consequências para a Fase 2:

1. risk_score É utilizável nas safras antigas (<=2014), onde está quase completo — não é
   uma feature descartável, é uma feature com disponibilidade dependente de época. Abre uma
   opção de desenho que o número agregado tinha fechado.
2. ALERTA de validação temporal. O projeto usa validação walk-forward (treina no passado,
   testa no futuro). risk_score é rico nas safras antigas (treino) e quase vazio nas
   recentes (teste). Usá-lo como feature no thin model cria uma feature cuja disponibilidade
   MUDA entre treino e teste — quebra silenciosamente a validação temporal. Se for usado,
   tem que ser com tratamento explícito de ausência por época (flag + sentinela, como o
   projeto já faz para rollouts de bureau nos aprovados), NUNCA como feature crua assumida
   presente.

Decisão a tomar na Fase 2 (não agora): ou (a) excluir risk_score do thin model e usar só
features presentes em ambas as épocas (amount, dti com ressalva, emp_length, geografia), ou
(b) incluí-lo com mecanismo de ausência temporal explícito e checar se sobrevive à
validação walk-forward. Pesquisar na abertura da Fase 2, junto com as técnicas de reject
inference.

Dados por ano: `reports/reject/risk_score_coverage_by_year.csv` (versionado no repo).

### B. `dti` NÃO é diretamente comparável entre aprovados e recusados — evita erro silencioso

- **Aprovados**: `dti` definido pelo dicionário LC = pagamentos mensais de dívida /
  obrigações totais, EXCLUINDO hipoteca e o empréstimo LC, sobre renda mensal
  autodeclarada. Já tratado no projeto (limpeza de dti>100, exclusão de joint
  applications).
- **Recusados**: `dti` vem como string com "%", sem garantia de mesma base de cálculo; é o
  valor da aplicação NEGADA, possivelmente autodeclarado e não verificado.

Implicação: comparar os dois `dti` vale como forma/ordem de grandeza da distribuição, NÃO
como números na mesma escala. No thin model, não tratar os dois `dti` como a mesma feature
sem uma nota metodológica. Erro que propagaria direto para o núcleo da Fase 2 se não
registrado. A comparação da Célula 6 deve ser lida com essa ressalva.

### C. `Employment Length` É consistente entre os dois — comparável

Dicionário LC: tempo de emprego em anos, 0 a 10 (0 = menos de 1 ano, 10 = 10+). Os
recusados usam o mesmo formato textual ("< 1 year", "10+ years"). Comparável, com atenção
só ao parsing texto->número. Sem ressalva metodológica além do parsing.

### E. Geografia: feature candidata forte, mas comparação adiada para a Fase 2

- **Recusados**: têm `state`, raramente nulo (medido na Célula 6). Dimensão presente e
  quase completa.
- **Aprovados**: `addr_state` foi descartada de `loans_clean.parquet` na classificação do
  notebook 03 (não era feature do modelo de aprovados). O dado existe no CSV cru
  (`accepted_*.csv`), mas não no parquet limpo.
- **Implicação para a Fase 2**: geografia é feature candidata FORTE para o thin model —
  presente e quase completa em ambas as populações, diferente de `risk_score` (66,9%
  ausente nos recusados, ver Achado A). Ao montar o dataset de modelagem na Fase 2,
  regenerar `addr_state` dos aprovados a partir do CSV cru para permitir a comparação e o
  uso como feature.
- **Nesta fase (1)**: perfilada só a geografia dos recusados. Comparação
  aprovados-vs-recusados adiada, por decisão de escopo (Fase 1 é descritiva).

### Números medidos que fecham a Fase 1 (confirmados no nosso arquivo)

- risk_score presente em 33,1% dos recusados (66,9% ausente) — confirma Achado A; thin
  model NÃO pode depender de risk_score.
- Valor solicitado quase idêntico: aprovados mean 13.091 / mediana 11.000; recusados mean
  13.133 / mediana 10.000. Não é o valor pedido que separa as populações.
- DTI diverge (recusados 26,58 vs aprovados 17,55 de média), consistente com a base de
  cálculo diferente (Achado B) — ler como forma, não escala idêntica.
- Auditoria de parsing: 0 linhas corrompidas em 27.648.741.

Dados: `reports/reject/approved_vs_rejected_comparison.csv` (versionado no repo).

## Confirmações que fortalecem o projeto

Proporção ~41x (27,6M recusados : 673k aprovados) confirmada por estudo independente (81%
recusados após limpeza, 93% antes). O argumento de volume para Spark é real e documentado
por terceiros, não artefato nosso.

## Decisões de arquitetura

### D. Híbrido Spark + DuckDB (ingestão vs análise local) — DECIDIDO e aplicado

**Contexto**: rodando local no Windows, a escrita nativa de Parquet do Spark e depois a
leitura de volta (`spark.read.parquet`) falham por falta de `winutils.exe`/`HADOOP_HOME`
(a camada de listagem de diretório do Hadoop). Instalar winutils foi rejeitado (binário de
terceiro, frágil, e resolveria só o sintoma).

**Decisão**: arquitetura híbrida, cada ferramenta onde é melhor.

- PySpark faz a ingestão do gzip e a escrita particionada. Roda nativo no Databricks
  (Linux, Hadoop configurado); no local, a escrita é feita via pyarrow (Arrow, sem dicts
  Python, fatiando anos grandes por mês para não estourar heap).
- DuckDB faz leitura, perfil e agregações locais sobre o Parquet particionado, com
  `hive_partitioning=true` sobre o layout `app_year=XXXX/part-*.parquet`. Single-node,
  streaming (dados maiores que a RAM), zero dependência de Hadoop/JVM.

**Por que é boa prática, não gambiarra**: Spark é a ferramenta certa para ingestão em
escala e escrita distribuída; DuckDB é a ferramenta certa para análise local sobre Parquet
num único nó. Forçar o Spark a fazer no Windows algo que ele faz mal ali, só para usar uma
ferramenta só, seria pior. "Spark para ingestão em escala, DuckDB para análise local" é uma
decisão de arquitetura defensável e alinhada ao que o mercado de dados vem adotando.

**Efeito no currículo** (quando a fase fechar): linha honesta "PySpark para ingestão
particionada em escala e DuckDB para análise sobre Parquet", ancorada em código real no
GitHub. NÃO alegar domínio de nenhuma das duas — uso real neste projeto, no padrão de
honestidade do resto do portfólio (linha defensável, com prova, sem inflar).

**Pré-requisito técnico registrado**: `duckdb` instalado via pip no `.venv` (pip puro, sem
binário de terceiro). No Databricks, o ramo Spark é usado; DuckDB é o caminho local.

**Se esta decisão sumir do contexto**: ao reencontrar o erro de winutils na
leitura/escrita de Parquet no Windows, NÃO instalar winutils — usar DuckDB
(leitura/análise) e pyarrow (escrita), que é o caminho já validado nesta fase.

## Checagens de repo pendentes (leitura de código, feitas pelo Claude Code)

### 1. Como o pipeline dos aprovados tratou registros malformados de CSV — RESPONDIDO

Pergunta original: `accepted_*.csv` também tem campos de texto livre (`emp_title`, `desc`,
`title`); se a limpeza foi pandas sem captura de malformados, pode haver linhas
silenciosamente nulificadas/deslocadas não contadas.

**Achado**:
- Confirmado que `emp_title`, `desc`, `title` existem e são texto livre
  (`docs/column_inventory.csv`: `emp_title` 7.39% missing, `desc` 94.42% missing, `title`
  1.03% missing).
- `notebooks/03_build_processed.ipynb` lê o CSV via
  `pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE, low_memory=False)`, **sem** override de
  `on_bad_lines`/`error_bad_lines` — ou seja, o default do pandas esteve em vigor
  (`on_bad_lines='error'` nas versões recentes, que levanta erro em linha com contagem de
  campos incompatível, em vez de pular silenciosamente).
- O total bruto lido (`n_total_file`) somou **exatamente 2.260.701** — idêntico ao total
  independente confirmado via `wc -l` (`docs/FACTS.md` linha 24) e à checagem cruzada do
  notebook 01 (`total_rows (2,260,701) == soma das exclusões + N final (2,260,701)? True`).
- Isso é evidência forte (não prova absoluta) de que a leitura bruta não perdeu nem
  deslocou linha nenhuma: se houvesse corrupção de parsing como a que apareceu nos
  recusados, é pouco provável que o total lido batesse exatamente com o `wc -l`
  independente. O parser C do pandas também tem tratamento de aspas/escape mais maduro e
  padrão-RFC4180 por padrão do que o leitor CSV do Spark, que precisou de opções
  explícitas (`multiLine`, `quote`, `escape`) para o mesmo tipo de padrão.
- **Ressalva honesta**: não existe no pipeline dos aprovados um equivalente ao
  `_corrupt_record` (auditoria linha-a-linha de conteúdo malformado) — a evidência é por
  contagem total batendo, não por auditoria direta de cada linha. Suficiente para não
  bloquear o Data Card v3, mas vale registrar a diferença de rigor entre as duas metades
  se o Data Card v3 for comparar os dois pipelines de ingestão lado a lado.

### 2. Varredura dos documentos pelo "~12x" errado — RESPONDIDO

Busca em `docs/`, `references/`, `README.md` deste repositório (`credit-default-
prediction-lendingclub`) não encontrou nenhuma ocorrência de "~12x" ou proporção
recusados:aprovados. O número errado não está neste repositório — nenhum documento aqui
(todos do escopo v2.0.0/v2.0.1, aprovados apenas) menciona essa proporção, já que reject
inference é trabalho novo iniciado nesta Fase 1. O documento de decisão que contém o
"~12x" vive fora deste repositório (provavelmente na sessão/planejamento externo de onde
este roadmap também veio) — precisa ser localizado e corrigido lá, não aqui.
