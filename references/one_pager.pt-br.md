# Transformando Aprovação de Crédito em Decisão de Lucro

**Resumo:** construí um modelo de previsão de inadimplência para empréstimos reais
peer-to-peer e o otimizei para maximizar o lucro da carteira, não a acurácia da previsão
— porque, para quem concede crédito, essas duas coisas não são a mesma coisa.

## Impacto Principal

**No conjunto de teste de 2015 (dados que o modelo nunca viu), o modelo recusa apenas
3,8% dos pedidos de empréstimo — evitando US$ 32,2 mi em perdas ao custo de US$ 23,1 mi
em juros que deixou de ganhar, um ganho líquido de US$ 9,0 mi sobre a política de aprovar
todo mundo.** Esse ganho é estatisticamente robusto (IC 95% US$ 7,6 mi–US$ 10,7 mi) e se
sustenta também contra um baseline de regressão logística (+US$ 6,3 mi).

## O Problema

Toda concessão de crédito é uma aposta: um retorno conhecido (os juros) contra uma perda
conhecida (o principal não pago, se o tomador não honrar o pagamento). Aprovar demais
corrói a carteira com inadimplência; recusar demais afasta clientes que pagariam em dia.
Acertar esse corte — e acertar em solicitantes genuinamente novos, não nos dados que o
modelo já conhece — é o problema de negócio inteiro. Havia dados reais disponíveis
(~673 mil empréstimos maduros da Lending Club, 2007–2015, 14,8% de inadimplência), o que
permitiu testar uma política de decisão contra o que de fato aconteceu, não uma hipótese.

## O Que Foi Feito

- Otimizei para o lucro da carteira, ponderando cada empréstimo pelo seu resultado real
  em dólares, em vez de tratar todo erro de previsão do mesmo jeito.
- Validei do jeito que um banco de fato opera: treinei no passado e testei no futuro
  (validação *walk-forward* — janelas temporais crescentes, nunca um embaralhamento
  aleatório dos dados).
- Calibrei as probabilidades do modelo, testei sua robustez, e auditei onde ele é mais
  fraco por perfil de tomador, em vez de reportar só uma nota agregada.

## Resultados

- **Ganho líquido de +US$ 9,0 mi** sobre aprovar todo mundo — escolhido por lucro
  esperado, não por acurácia, e essa escolha importou: o modelo vencedor empata com um
  baseline mais simples de regressão logística em AUC (0,68), mas vence com folga em
  lucro — valor que um teste de acurácia sozinho teria deixado passar.
- **Recusar apenas os 10% mais arriscados evita ~21% de todas as inadimplências** — o
  dobro da eficácia de um corte aleatório do mesmo tamanho.
- **Ressalva honesta, dita de cara:** o modelo é menos confiável exatamente no segmento
  de maior risco e menor renda, e foi treinado só sobre empréstimos aprovados — não
  pontua quem foi recusado (viés de seleção). Não foi construído para decisão de crédito
  ao vivo como está.

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP

## Links

- Repositório: https://github.com/MateusFPavan/credit-default-prediction-lendingclub
- Relatório técnico completo: `docs/technical_report.md`
- Contato: https://www.linkedin.com/in/mateus-fardin-pavan/
