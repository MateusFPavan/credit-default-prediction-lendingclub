# ============================================================================
# Fase 2b - Bloco 1: avaliacao Bayesiana ciente do vies (Kozodoi) + varredura de prior
# ============================================================================
# CONCEITO (material de reaquecimento Bayesiano -- ler antes do codigo)
#
# O problema de fundo (vies de selecao / MNAR): o modelo de credito so ve o desfecho
# (bom/mau pagador) de quem foi APROVADO. A avaliacao padrao (AUC, lucro, threshold
# otimo) roda so nesse subconjunto -- e o subconjunto e enviesado por construcao: a
# propria LendingClub ja filtrou os "piores" antes do modelo os ver. Uma performance boa
# nos aprovados nao garante a mesma performance na populacao "through-the-door" (todo
# mundo que aplicou, aprovado ou nao).
#
# A ideia do Kozodoi (arXiv 2407.13009) pra medir a performance na populacao INTEIRA sem
# ter os labels dos recusados: tratar o label de cada recusado como uma VARIAVEL LATENTE
# (desconhecida, mas com uma distribuicao assumida) em vez de simplesmente ignora-la.
# Bayesiano no sentido classico: comeca de um PRIOR (a taxa de default que vc ACREDITA que
# os recusados teriam, antes de ver qualquer score) e usa isso pra completar a metrica na
# populacao inteira -- e' literalmente "assumir uma crenca sobre o que nao pode ser
# observado, e propagar essa crenca pela conta".
#
# Formalmente (versao didatica, nao o framework Bayesiano completo do paper -- ver NOTA
# de escopo mais abaixo): a metrica esperada na carteira aceita e' uma combinacao de
#   (a) o que sabemos com certeza: o label REAL dos aprovados aceitos.
#   (b) o que so podemos ESPERAR: o label dos recusados aceitos, sob o prior assumido.
# Pergunta que isso responde: "se eu assumir que X% dos recusados dariam default, qual
# seria a performance REAL do meu modelo se ele decidisse sobre TODO MUNDO que aplicou,
# nao so sobre quem a LendingClub deixou ele ver?"
#
# O ponto CRITICO, que e o motivo deste bloco existir: o resultado depende inteiramente
# do PRIOR escolhido. O Kozodoi consegue mostrar um "teto" porque assume um PRIOR
# PERFEITO (ele conhece, ou aproxima muito bem, a taxa real dos recusados, porque o
# dataset dele tem uma amostra de controle -- ver Fase 2a, achado sobre a amostra
# nao-enviesada de 1.967 casos que o Kozodoi tem e o Lending Club nao). No mundo real -- e
# no Lending Club especificamente -- o prior e DESCONHECIDO. Isso nao invalida o metodo
# Bayesiano em si (ele e matematicamente correto dado um prior), mas levanta a pergunta
# que a Fase 2b existe pra responder com NUMERO, nao so citacao:
#
#   Se a estimativa Bayesiana MUDA MUITO conforme o prior assumido, o metodo Bayesiano
#   nao resolveu o problema de nao ter os labels -- ele so TRANSFERIU o problema pra
#   escolha do prior. E como a Fase 2a ja provou (empiricamente, nao por citacao) que o
#   Lending Club NAO tem a taxa populacional real pra escolher o prior certo, a conclusao
#   e que a avaliacao Bayesiana AQUI tambem fica indeterminada -- ela herda a mesma
#   limitacao estrutural que already invalidou a Fase 2a (ausencia de outcome de
#   recusado, ausencia de amostra nao-enviesada).
#
# NOTA de escopo (decidido, nao ambiguidade): implementamos a extensao Bayesiana da TAXA
# DE DEFAULT esperada da carteira sob uma politica de aceitacao -- nao o framework
# Bayesiano completo do paper (que envolve inferencia posterior formal sobre parametros
# do modelo). A versao aqui basta pra testar a tese (a sensibilidade ao prior), e o rigor
# extra da simulacao completa nao teria retorno de portfolio (decisao ja tomada, nao
# reaberta aqui).
#
# Referencias: Kozodoi (arXiv 2407.13009, 1909.06108); Illusion of Improvement (arXiv
# 2606.18479) -- a mesma logica de "sem taxa populacional real, sem validacao honesta"
# que fechou a Fase 2a se aplica aqui, agora testada contra o metodo Bayesiano tambem.

import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# notebooks/17_reject_thin_model.py carregado como modulo (mesmo padrao ja usado em
# notebooks/18_profit_evaluation.py e notebooks/19_illusion_demonstration.py).
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)


def bayesian_expected_default_rate(pd_appr, y_appr, pd_rej, prior_rej_bad_rate,
                                    acceptance_rate):
    """
    Estima a taxa de default ESPERADA da carteira, se aceitassemos os `acceptance_rate`
    de menor risco da POPULACAO INTEIRA (aprovados + recusados), usando:
    - label real dos aprovados aceitos;
    - label ESPERADO dos recusados aceitos, sob o PRIOR (prior_rej_bad_rate).

    IMPORTANTE (integracao): pd_appr e pd_rej precisam estar na MESMA escala de
    probabilidade pra fazer sentido ordena-los juntos (np.argsort sobre a concatenacao).
    So o thin model pontua as duas populacoes na mesma escala (o XGB de 79 features nao
    consegue pontuar recusados -- seria a mesma armadilha de escala que o Bloco 7
    encontrou e corrigiu). Por isso pd_appr aqui vem do THIN MODEL, nao do XGB.

    prior_rej_bad_rate: a taxa de default ASSUMIDA dos recusados (o prior). E' isto que
    varremos -- porque nao sabemos o valor verdadeiro (Fase 2a: sem taxa populacional real).
    """
    all_pd = np.concatenate([pd_appr, pd_rej])
    n_total = len(all_pd)
    n_accept = int(n_total * acceptance_rate)
    order = np.argsort(all_pd)  # aceita menor PD primeiro
    accepted = order[:n_accept]

    is_appr = accepted < len(pd_appr)
    appr_defaults = y_appr[accepted[is_appr]].sum()
    n_rej_accepted = int((~is_appr).sum())
    rej_defaults_expected = n_rej_accepted * prior_rej_bad_rate

    exp_rate = (appr_defaults + rej_defaults_expected) / n_accept
    return exp_rate, n_rej_accepted


def prior_sweep(pd_appr, y_appr, pd_rej,
                 prior_grid=(0.10, 0.20, 0.30, 0.40, 0.50),
                 acceptance_rates=(0.1, 0.3, 0.5)):
    """
    Varre o PRIOR e mostra quanto a taxa de default esperada da carteira muda.
    Se muda MUITO -> a conclusao depende do prior -> Bayesian herda a limitacao do LC.
    Se muda POUCO -> Bayesian seria robusto (improvavel dado o AUC 0.56 do thin model:
    um score fraco deixa a decisao de quem entra dominada pelo prior, nao pelo score).
    """
    appr_base = float(y_appr.mean())
    print(f"[REF] taxa default dos aprovados: {appr_base:.4f}")
    print("[NOTA] prior verdadeiro dos recusados: DESCONHECIDO (Fase 2a). Por isso varremos.\n")
    print(f"{'accept_rate':>12}{'prior':>8}{'taxa_esperada':>16}{'n_rej_aceitos':>16}")
    results = {}
    for ar in acceptance_rates:
        for prior in prior_grid:
            rate, n_rej = bayesian_expected_default_rate(pd_appr, y_appr, pd_rej, prior, ar)
            results[(ar, prior)] = (rate, n_rej)
            print(f"{ar:>12}{prior:>8}{rate:>16.4f}{n_rej:>16,}")
        vals = [results[(ar, p)][0] for p in prior_grid]
        print(f"    -> amplitude sob esta accept_rate: {max(vals) - min(vals):.4f}\n")
    print("[LEITURA] Amplitude grande = a conclusao depende do prior = Bayesian nao resolve")
    print("o problema no LC (so o transfere para a escolha do prior, que nao sabemos).")
    return results


if __name__ == "__main__":
    print("[Fase 2b - Bloco 1] avaliacao Bayesiana + varredura de prior\n")

    print("Carregando aprovados (split 'train')...")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    print(f"  {len(X_appr):,} linhas | bad rate {y_appr.mean()*100:.2f}%")

    print("\nCarregando recusados (Parquet particionado, 27.648.741 linhas esperadas)...")
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    print(f"  {len(X_rej):,} linhas")

    print("\nTreinando thin model (baseline_ignore_rejects, Bloco 10) -- unico modelo que "
          "pontua as duas populacoes na mesma escala...")
    thin = nb17.baseline_ignore_rejects(X_appr, y_appr)
    pd_appr = thin.predict_proba(X_appr)[:, 1]
    pd_rej = thin.predict_proba(X_rej)[:, 1]

    print("\n=== VARREDURA DE PRIOR ===")
    results = prior_sweep(pd_appr, y_appr, pd_rej)

    print("\n[Fase 2b - Bloco 1] concluido.")
