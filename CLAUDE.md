# Convenções do projeto (lidas pelo Claude Code)

## Git
- NUNCA fazer push sem autorização explícita do Mateus. Commit local sempre.
- Um propósito por commit; código e documentação em commits separados.
- Antes de commit: `git status` e confirmar o diretório certo (`pwd`, `git remote -v`).
- Scratches de diagnóstico ficam FORA do controle de versão.
- Verificar o ESTADO real após ação crítica (git log/status) — não confiar no relato.

## Segurança
- Grep de credencial antes de commitar qualquer notebook que toca dados.
- Nunca versionar token/segredo. Credencial em ambiente novo vai em célula isolada, apagada
  antes de exportar.

## Dados
- Investigar o dado cru antes de tratar; só tratar coluna com mecanismo. Não filtrar por regra
  genérica sem investigar.
- Reproduzir artefato canônico exatamente (byte a byte) ao portar lógica; não recriar à mão.

## Testes
- Rodar a suíte antes de commitar, se houver.
- NÃO alterar um teste para fazê-lo passar — corrigir a causa.

## Sobrescrita de arquivo
- Validar o CONTEÚDO (grep de um marcador esperado) antes de sobrescrever — não confiar no nome.

## Específico deste repo (crédito Lending Club)
<!-- ultima atualizacao: 2026-09-15 -->

### População e escopo
- População analítica: só empréstimos com desfecho concluído (Fully Paid / Charged Off).
- Corte de maturidade: 36m emitidos até Dez/2015; 60m até Dez/2013 (evita viés de maturidade).
  NÃO reincluir vintages imaturas — é decisão documentada em docs/scope.md.
- FEATURE_SET tem 78 features (application_type removido em P-045), 4 categóricas.

### Reject inference (v3.0.0)
- Tese fechada: RI não é validável no Lending Club (sem outcome de recusado, sem taxa
  populacional). Isso é SEÇÃO HONESTA no portfólio, NÃO manchete.
- Processo interno do projeto fica em _processo_interno/ (ignorado, não vai ao GitHub).

### Documentação
- docs/reject_inference_roadmap.md é a fonte única de decisões deste projeto.
- Ao mexer em doc, procurar frases que a mudança tornou MENTIRA (ex.: "not deployed" quando
  já há deploy), não só adicionar o novo.
