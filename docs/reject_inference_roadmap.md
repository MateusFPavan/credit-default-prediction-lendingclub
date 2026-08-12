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

## 2. Reject inference (técnica central) — PESQUISADO (Fase 2), ainda não implementado

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

**Pesquisa concluída**: ver seção "Fase 2 — método de reject inference (pesquisado, ainda
não implementado)" logo abaixo. Implementação continua pendente (gatilho não disparado).

## Fase 2 — método de reject inference (pesquisado, ainda não implementado)

Pesquisa feita antes de abrir a Fase 2. Conclusão-chave: a área é DIVIDIDA entre prática de
mercado (faz reject inference) e ceticismo acadêmico (questiona se funciona). O valor de
portfólio está em conduzir a Fase 2 CIENTE dessa controvérsia, não aplicando uma técnica
cegamente.

### As técnicas (o que existe)

- **Augmentation (re-weighting) e Parcelling**: as duas únicas que a literatura crítica NÃO
  descartou de imediato. Legítimas dependendo da especificação e do mecanismo de
  missingness. Candidatas para uso.
- **Fuzzy augmentation, Reclassification, Twins**: mais populares em ferramentas comerciais
  (SAS, MATLAB), mas uma tese do Inria (Ehrhardt) as considerou inúteis / difíceis de
  justificar. Usar com ceticismo, ou só como comparação, não como método principal.
- Todas partem do mesmo esqueleto: treina scorecard nos aprovados, pontua os rejeitados,
  infere good/bad, recombina e retreina.

### A controvérsia (o que muda a postura)

- **Academia** (Hand & Henley, Inria, pacote R `scoringTools`): reject inference no caso
  MNAR depende de julgamento de especialista e NÃO pode ser testado; "inferência confiável
  é impossível; a única abordagem robusta é aceitar uma amostra de rejeitados e observar".
- **Mercado/regulador**: espera reject inference feito e documentado; não fazer subestima
  risco por faixa de score.
- **Resolução para o projeto**: fazer como EXPERIMENTO HONESTO — medir se ajudou, não
  assumir.

### Achado mais recente e mais relevante (usa o Lending Club!)

Paper "The Illusion of Improvement" (Scarone & Baeza-Yates, arXiv:2606.18479, ECML PKDD
2026):

- **Modo de falha estrutural**: em ciclo de retreino, acurácia sobe enquanto
  recall/qualidade de rejeição cai — "ilusão de melhora". Métricas padrão (acurácia/AUC)
  são ENGANOSAS sob viés de seleção.
- **Solução proposta**: exploração controlada — aprovar deliberadamente 2-5% dos
  rejeitados e observar resultado real; basta isso para diagnosticar a severidade do
  problema.
- **CRÍTICO para nós**: testaram no Lending Club (entre 3 datasets). No Lending Club, o
  viés de sobrevivência distorce a métrica em apenas ~2,6% (secundário), vs ~30% em outro
  dataset. Ou seja: no NOSSO dataset, o efeito de reject inference tende a ser PEQUENO.

### Desenho da Fase 2 (decorrente da pesquisa)

1. Diagnosticar primeiro se o viés sequer importa no Lending Club (esperado: pequeno,
   ~2,6%). Uma conclusão honesta "efeito marginal, com evidência" vale mais que melhora
   inflada.
2. Aplicar augmentation e/ou parcelling (as legítimas), não fuzzy/twins/reclassification
   como método principal.
3. NÃO usar acurácia/AUC como métrica de decisão — usar qualidade de rejeição e,
   idealmente, o enquadramento de exploração controlada. Conecta direto com a métrica de
   LUCRO que o projeto de crédito já usa (o projeto já rejeita AUC como métrica de
   decisão).
4. Validar sob walk-forward (como o resto do projeto), documentando a interação com o
   risk_score temporalmente ausente (Achado A) e o dti não comparável (Achado B).
5. Thin model usa features presentes nas duas populações: amount, dti (com ressalva),
   emp_length, geografia — NÃO risk_score como feature crua.

### Valor de currículo

Transforma a Fase 2 de "apliquei fuzzy augmentation" (ingênuo, o que a academia despreza)
em "conduzi um estudo honesto sobre se reject inference ajuda no Lending Club, ciente da
controvérsia, com a metodologia de avaliação correta e validação temporal". A segunda
versão é a que credencia junto a um time de crédito sério. Referência central e atual:
arXiv:2606.18479 (2026).

### Ferramentas candidatas (pesquisar implementação na abertura da Fase 2)

- Pacote R `scoringTools` (augmentation, parcelling etc.) — referência de implementação.
- Em Python, implementar as duas técnicas legítimas manualmente (são simples) ou verificar
  se há equivalente mantido. Decidir na abertura da fase.

## Fase 2 — decisão de implementação (pesquisado)

Não existe biblioteca Python madura/dedicada de reject inference. A única implementação de
referência é em R: pacote `scoringTools` (augmentation, parcelling, fuzzy,
reclassification, twins). Em Python não há equivalente mantido.

**ARMADILHA de nomenclatura**: `scikit-fallback` (Python) NÃO serve. É "reject option"
(modelo se recusar a prever casos ambíguos), coisa diferente de reject inference de
crédito. Não perder tempo com ele apesar do nome parecido.

**Decisão**: implementar augmentation e parcelling à mão em Python. As técnicas são
simples (parcelling: pontuar rejeitados com o modelo dos aprovados, binar por faixa de
score, atribuir good/bad na proporção da taxa de inadimplência esperada por faixa, com
fator multiplicador conservador — rejeitados devem ter bad rate 2-5x maior que aprovados).
Implementar na mão FORTALECE o portfólio (mostra domínio do mecanismo, não só chamada de
API) e dá controle para encaixar na validação walk-forward e na avaliação por LUCRO do
projeto — coisa que biblioteca pronta não faria.

**Confirmação empírica da escolha de técnica**: estudo com banco de crédito ao consumidor
francês (Kozodoi/Lessmann et al. e trabalhos correlatos) concluiu que reweighting e
parcelling produzem resultados mais precisos e relevantes que fuzzy augmentation e a
correção de dois estágios de Heckman. Ou seja, as duas escolhidas são as que empiricamente
ganham, não só "as não descartadas".

**Referência de implementação** (consultar, não copiar): `scoringTools` (R) no GitHub,
para ver como `augmentation()` e `parcelling()` são estruturadas; portar a lógica para
Python.

## Fase 2 — plano faseado (decidido)

Base Bayesiana do Mateus: conhece o conceito, sem prática recente. Por isso: FASEADO.
Entrega a 2a (que domina) primeiro; decide a 2b (Bayesiana) depois, com intuição já
construída pela 2a. Prática antes de teoria pesada.

### Fundamento (achado que ancora tudo — Kozodoi & Lessmann 2025, EJOR)

Testado em dados reais COM grupo de controle sem viés (raro): "reject inference é um
problema difícil com potencial MODESTO de melhorar o scorecard; tratar o viés de amostragem
na AVALIAÇÃO é caminho muito mais promissor". Ganho medido: ~8% de lucro ao usar avaliação
ciente do viés para decidir taxas de aceitação. Ou seja: o ouro está na AVALIAÇÃO, não na
inferência. Isso alinha 100% com a tese do projeto (lucro > AUC).

### FASE 2a (fazer agora — Mateus domina)

Objetivo: baseline honesto + BASL, medir se inferência ajuda no Lending Club.

1. **Modelo base**: treinar nos aprovados (reusar o XGB do projeto v2.0.0 como ponto de
   partida; features presentes nas duas populações — amount, dti c/ ressalva, emp_length,
   geografia; NÃO risk_score cru, ver Achado A).
2. **Baseline de reject inference**: implementar à mão em Python:
   - augmentation/parcelling (pontuar rejeitados, binar por score, atribuir good/bad na
     proporção da bad rate esperada, fator conservador 2-5x).
3. **BASL (bias-aware self-learning)**: extensão de self-learning com:
   - estágio de FILTRAGEM (selecionar rejeitados mais confiáveis, não todos),
   - estágio de ROTULAGEM (rotular os selecionados),
   - estágio de TREINO (retreinar no conjunto aumentado),
   - PARADA ANTECIPADA (evitar o feedback loop de degradação). Filosofia BASL: priorizar
     performance preditiva, NÃO "desviesar" a todo custo.
4. **Avaliação da 2a**: sob walk-forward (como o resto do projeto), métrica de LUCRO (não
   AUC). Medir se BASL/parcelling batem o modelo só-aprovados. Resultado esperado: ganho
   pequeno (Lending Club tem viés ~2,6%, secundário — Scarone & Baeza-Yates 2026).
   Conclusão honesta de "efeito pequeno, com evidência" é resultado VÁLIDO e sofisticado.

### FASE 2b (decidir depois — exige reaquecer Bayesiano)

Avaliação ciente do viés (o "ouro"). Duas variantes, decidir conforme fôlego:

- **Mínima** (sem Bayesiano pesado): enquadramento de exploração controlada (Scarone &
  Baeza-Yates 2026) — medir "qualidade de rejeição" além de acurácia; mostrar que avaliação
  ingênua (AUC no holdout de aprovados) superestima a performance.
- **Completa** (framework Bayesiano de Kozodoi): avaliação Bayesiana sobre população
  conjunta aprovados+rejeitados; estimar performance futura real. É o ouro, e o mais caro
  de aprender (estimativa: 1-2 semanas se reaquecer Bayesiano; 3-6 semanas se do zero).
  Defender mal é pior que não ter — só ir à completa se der pra explicar com segurança.

### Estimativas de esforço (para planejar)

- 2a completa (baseline + BASL + avaliação por lucro walk-forward): ~alguns dias a 1
  semana.
- 2b mínima: poucos dias adicionais.
- 2b completa (Bayesiano): +1-6 semanas conforme base Bayesiana no momento.

### Regra de ouro do resultado (gerenciar expectativa)

O valor da Fase 2 está no MÉTODO (experimento honesto, avaliação correta), não em provar
que reject inference é milagroso. Ganho pequeno é o resultado provável e é uma conclusão
forte quando bem comunicada. Não inflar.

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

### Estado da Fase 1: FECHADA

Fase 1 concluída e commitada localmente (commits 49c2349, 9f7617e, cfa0a70; sem push).
Tudo rodou limpo de ponta a ponta (notebook `16_reject_ingestion_profile.py`, Células 1-7):

- Ingestão do gzip (27.648.741 linhas) e escrita em Parquet particionado por ano — OK.
- Auditoria de parsing (`_corrupt_record`): 0 linhas corrompidas após parsing robusto — OK.
- Perfil de colunas, Risk_Score por ano, comparativo aprovados vs recusados — OK.
- Dois perfis-chave gravados como CSV em `reports/reject/`
  (`risk_score_coverage_by_year.csv`, `approved_vs_rejected_comparison.csv`).
- Manifesto de proveniência gravado (`reports/reject/reject_manifest.json`).

Arquitetura: PySpark (ingestão/escrita) + DuckDB (leitura/perfil local), ver Decisão D.

### Pendente para as próximas etapas (não bloqueia a Fase 1)

- **Validação no Databricks Free Edition**: rodar o mesmo notebook no ambiente Linux
  serverless, onde o ramo Spark nativo funciona (prova a linha "Databricks" do currículo).
  Único passo restante para encerrar a Fase 1 por completo.
- **Fase 2**: geografia dos aprovados (regenerar addr_state do CSV cru para a comparação);
  técnicas de reject inference (pesquisa dedicada); desenho do thin model considerando o
  Risk_Score temporalmente ausente (Achado A) e o dti não comparável (Achado B).
- **Melhoria de eficiência (candidato)**: o notebook reescreve os 27,6M do zero a cada
  rodada. Separar a ingestão pesada (roda uma vez) das análises leves (leem o Parquet
  pronto) evitaria reprocessar tudo a cada ajuste — útil na Fase 2, quando a iteração é
  frequente. Não feito na Fase 1 de propósito (não valia reabrir uma fase fechando).

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
