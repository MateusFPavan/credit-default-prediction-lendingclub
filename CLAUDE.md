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
