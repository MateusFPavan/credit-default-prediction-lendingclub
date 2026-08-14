# ============================================================================
# Fase 2a - Bloco 7: avaliacao por lucro honesta (formula Kozodoi, LGD varrida)
# ============================================================================
# Formula por emprestimo (Kozodoi, arXiv 2407.13009):
#   profit = PD * A * (1+i) * (1-LGD) + (1-PD) * A * (1+i) - A
# LGD varrida em {0.5..0.9} (P2P sem garantia recupera 10-30% -> LGD 70-90%; 0.5 e piso
# otimista). Varrer premissa em vez de fixar = honesto (mesma logica das outras varreduras
# deste projeto).
#
# [VERIFICAR NO REPO] resolvido:
#   - docs/REFERENCIA_avaliacao_por_lucro.md NAO existe no repo (Glob/find vazio em
#     qualquer local/caixa). A formula, a citacao (Kozodoi arXiv 2407.13009) e a grade de
#     LGD ja vieram completas e inequivocas no proprio pedido -- nao ha nome de
#     funcao/coluna pendendo desse arquivo especifico, entao NAO bloqueei o bloco por
#     causa dele. Reportado ao Mateus como achado, nao como bloqueio (ver mensagem de chat).
#   - src/economics.py: existe, mas calcula lucro REALIZADO (interest/loss reconstruidos
#     de installment/term/loan_amnt/total_rec_prncp -- desfecho ja observado). E' um
#     paradigma DIFERENTE da formula Kozodoi (valor ESPERADO via PD do modelo). Nao da pra
#     reusar directamente (loss=100% do principal implicito la, sem LGD variavel; e nao
#     serve pros recusados, que nunca tem total_rec_prncp por nao terem virado
#     emprestimo). Usado aqui so como REFERENCIA de sanity-check na Etapa 1 (mesmo
#     pd_hat/test set, formula antiga vs nova).
#   - Taxa real (int_rate) e valor (loan_amnt) e label (target) dos aprovados: existem em
#     data/processed/test.parquet (src.data.load_split("test")). int_rate esta em
#     src.data.EXCLUDED (nao e feature do modelo) mas a coluna crua existe e foi
#     verificada: armazenada como PERCENTUAL (ex. 11.27 = 11.27%), nao fracao -- dividida
#     por 100 antes de entrar na formula (unidade confirmada por leitura direta do
#     parquet, nao suposta).
#   - Scoring dos aprovados: reusa o padrao JA usado por src/run_facts.py (que gera
#     docs/FACTS.md, docs comprovadamente corretos) -- build_features -> prepare_X ->
#     reindex nas colunas do booster -> predict_proba. NAO usei src.scoring.score_frame
#     porque ela chama clean_record() antes de build_features, e load_split("test") ja
#     vem limpo (sentinelas do notebook 03 ja aplicadas) -- run_facts.py NAO chama
#     clean_record nesse ponto, entao segui o padrao comprovado, nao o mais generico.
#   - Recusados/thin model: reusa notebooks/17_reject_thin_model.py inteiro (carregado
#     como modulo via importlib, sem rodar o bloco __main__) -- load_approved_shared,
#     load_rejected_shared, train_thin_model, SHARED. Nao duplica logica ja tratada
#     (dti/emp_length/amount por mecanismo, Blocos 1c-6).
#
# Decisoes de escopo (registrar, nao decisao de metodo pendente):
#   - Recusados NAO tem taxa i real (nunca viraram emprestimo). Premissa: i = taxa media
#     de int_rate dos APROVADOS (split 'train', mesma populacao usada pelo thin model) na
#     banda de score equivalente do thin model. Banda fixada em 10 (meio-termo dos
#     Blocos 1d-6; nao resweepada aqui pra manter a grade de combinacoes tratavel).
#   - Recusados NAO tem label real. O PD usado e o score do thin model (Etapa 2, cenario
#     "aceitar todos") ou o PD ajustado pelo parcelling (Etapa 3, base_mult/censored_extra
#     -- mesma logica de notebooks/17, mas AQUI vetorizada: usa a probabilidade continua
#     ajustada por banda diretamente na formula de lucro, em vez de sortear um label 0/1
#     como o parcelling original fazia -- sortear introduziria ruido que a formula de
#     valor esperado nao precisa).
#   - Lucro sobre recusados e CENARIO ("se aceitos"), nunca medida. Explicito em todo
#     print/report.
#
# Bloco 7b (CORRIGE a Etapa 3 original): a primeira versao aplicava o threshold 0.26 do
#   XGB (79 features, aprovados) direto na escala de PD do thin model (3 features,
#   recusados) -- invalido, escalas diferentes. Confirmado empiricamente: a 0.26,
#   100% dos 27,5M recusados eram "aceitos", implausivel dado que a populacao recusada e'
#   objetivamente pior em dti/emp_length (Blocos 1c/4). Corrigido com duas solucoes da
#   literatura (KNIME; Verbraken 2014): threshold otimo NA PROPRIA escala de cada modelo
#   (teto individual) + comparacao a TAXA DE ACEITACAO FIXA entre estrategias (principal,
#   justa entre escalas -- mesma fracao aceita, cada estrategia escolhe QUAIS).

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split, FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.economics import compute_interest_loss, optimal_threshold, profit_at_threshold

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# notebooks/17_reject_thin_model.py carregado como modulo (nome comeca com digito,
# nao importavel via `import`) -- reusa os loaders/thin model ja tratados por mecanismo,
# sem rodar o bloco __main__ dele (guard if __name__ == "__main__" protege).
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)

LGD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
THR_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
N_BANDS = 10  # mesma granularidade "meio-termo" das varreduras dos Blocos 1d-6
BASE_MULT_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)  # faixa corrigida do Bloco 3
CENSORED_EXTRA_GRID = (1.0, 1.25, 1.5)


# --- formula de lucro por emprestimo (Kozodoi) -------------------------------
def expected_profit_per_loan(pd_hat, A, i, lgd):
    """Lucro esperado de UM emprestimo. Vetorizavel (pd_hat, A, i podem ser arrays)."""
    pd_hat = np.asarray(pd_hat, dtype=float)
    A = np.asarray(A, dtype=float)
    i = np.asarray(i, dtype=float)
    return (pd_hat * A * (1 + i) * (1 - lgd)
            + (1 - pd_hat) * A * (1 + i)
            - A)


def portfolio_profit(pd_hat, A, i, lgd, accept_mask=None):
    """Lucro total de uma carteira. accept_mask: quais emprestimos sao aceitos (bool).
    Se None, aceita todos."""
    p = expected_profit_per_loan(pd_hat, A, i, lgd)
    if accept_mask is not None:
        p = p[np.asarray(accept_mask)]
    return float(np.sum(p))


def incremental_profit(baseline_profit, ri_profit):
    """Incremental = lucro com reject inference - lucro do baseline so-aprovados.
    Positivo => RI ajudou."""
    return ri_profit - baseline_profit


# ============================================================================
# ETAPA 1 - VALIDAR a formula nos APROVADOS (PD/taxa/label reais)
# ============================================================================
def validate_on_approved():
    """
    Roda a formula Kozodoi nos aprovados (split 'test', XGB_walkforward v2.0.0) e
    confere coerencia contra a formula REALIZADA ja usada no projeto (src.economics),
    no MESMO pd_hat e MESMO test set -- se o threshold otimo da formula nova divergir
    muito do 0.31/0.26 ja conhecidos, e sinal de erro de unidade/formula.
    """
    raw_test = load_split("test")
    test_feat = build_features(raw_test.copy())

    xgb_model = joblib.load(MODELS_DIR / "xgb_final.joblib")
    trained_cols = list(xgb_model.get_booster().feature_names)
    X = prepare_X(test_feat, FEATURE_SET, CATEGORICAL_COLS).reindex(columns=trained_cols, fill_value=0)
    pd_hat = xgb_model.predict_proba(X)[:, 1]

    A = raw_test["loan_amnt"].to_numpy(dtype=float)
    i = raw_test["int_rate"].to_numpy(dtype=float) / 100.0  # percentual -> fracao
    y = raw_test["target"].to_numpy()

    interest, loss = compute_interest_loss(raw_test)
    interest = interest.to_numpy()
    loss = loss.to_numpy()
    thr_realized, profit_realized = optimal_threshold(y, pd_hat, interest, loss)
    profit_at_031 = profit_at_threshold(y, pd_hat, 0.31, interest, loss)

    print("=== ETAPA 1: lucro nos aprovados (Kozodoi, valor esperado) por LGD ===")
    print(f"  [referencia formula REALIZADA, src.economics, mesmo pd_hat/test set]")
    print(f"    threshold otimo={thr_realized:.2f} | lucro={profit_realized:,.2f}")
    print(f"    lucro @0.31 (operacional)={profit_at_031:,.2f} "
          f"(docs/FACTS.md xgb_profit: 242,230,710.89)")
    print(f"\n{'LGD':>5}{'lucro_total_kozodoi':>22}{'threshold_otimo':>18}")
    results = {}
    for lgd in LGD_GRID:
        profits = np.array([
            portfolio_profit(pd_hat, A, i, lgd, accept_mask=(pd_hat < t)) for t in THR_GRID
        ])
        best_idx = int(np.argmax(profits))
        best_thr, best_profit = float(THR_GRID[best_idx]), float(profits[best_idx])
        results[lgd] = (best_thr, best_profit)
        print(f"{lgd:>5}{best_profit:>22,.0f}{best_thr:>18.3f}")
    return pd_hat, A, i, y, results


# ============================================================================
# ETAPA 2 - CENARIO dos recusados (taxa/label = premissas)
# ============================================================================
def evaluate_rejected_scenario():
    """
    Lucro CENARIO dos recusados se TODOS fossem aceitos (sem threshold ainda -- isso
    vem na Etapa 3). PD = score bruto do thin model; i = premissa (taxa media da banda
    de score equivalente nos aprovados). E' CENARIO, nao medida -- recusados nunca
    viraram emprestimo, nao tem label real nem taxa real.
    """
    print("\n[Carregando thin model e recusados tratados (Blocos 1c-6)...]")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    thin = nb17.train_thin_model(X_appr, y_appr)

    p_appr = thin.predict_proba(X_appr)[:, 1]
    p_rej = thin.predict_proba(X_rej)[:, 1]
    A_rej = X_rej["amount"].to_numpy(dtype=float)

    # premissa de taxa: int_rate media dos aprovados (split 'train') na banda equivalente
    train_raw = load_split("train")
    int_rate_train = train_raw["int_rate"].to_numpy(dtype=float) / 100.0

    edges = np.quantile(p_appr, np.linspace(0, 1, N_BANDS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    appr_band = np.digitize(p_appr, edges[1:-1])
    rej_band = np.digitize(p_rej, edges[1:-1])

    band_rate = {}
    band_bad = {}
    for b in range(N_BANDS):
        m = appr_band == b
        band_rate[b] = float(int_rate_train[m].mean()) if m.any() else float(int_rate_train.mean())
        band_bad[b] = float(y_appr[m].mean()) if m.any() else float(y_appr.mean())
    i_rej = np.array([band_rate[b] for b in rej_band])

    print("\n=== ETAPA 2: lucro-cenario dos recusados por LGD (aceitar TODOS; PREMISSA de taxa) ===")
    print(f"{'LGD':>5}{'lucro_cenario':>20}")
    scenario_results = {}
    for lgd in LGD_GRID:
        lucro = portfolio_profit(p_rej, A_rej, i_rej, lgd)  # aceita todos = cenario
        scenario_results[lgd] = lucro
        print(f"{lgd:>5}{lucro:>20,.0f}")
    print("[NOTA] Lucro dos recusados e CENARIO: taxa (banda equivalente dos aprovados, "
          "split 'train') e PD (thin model, 3 features) sao premissas, nao medida.")

    return {
        "thin": thin, "X_appr": X_appr, "y_appr": y_appr, "X_rej": X_rej,
        "rej_flags": rej_flags, "p_appr": p_appr, "p_rej": p_rej, "edges": edges,
        "appr_band": appr_band, "rej_band": rej_band, "band_rate": band_rate,
        "band_bad": band_bad, "i_rej": i_rej, "A_rej": A_rej,
        "scenario_results": scenario_results,
    }


# ============================================================================
# ETAPA 3 (Bloco 7b, CORRIGE a Etapa 3 original): comparar threshold 0.26 do XGB
# (79 features) direto na escala do thin model (3 features) foi invalido -- escalas
# diferentes (confirmado empiricamente: 100% dos recusados "aceitos" a 0.26, o que e'
# implausivel dado que sabemos que a populacao recusada e' objetivamente pior em dti e
# emp_length). Duas solucoes da literatura (KNIME/Verbraken 2014; comparacao a taxa de
# aceitacao fixa), threshold-por-modelo como teto de cada um, taxa-de-aceitacao-fixa como
# comparacao PRINCIPAL (justa entre modelos: mesma fracao aceita, cada modelo escolhe
# QUAIS -- isola qualidade da decisao do volume de aceitacao).
# ============================================================================
def parcelling_pd_rejected(rej_band, band_bad, rej_flags, base_mult, censored_extra):
    """PD ajustada por banda (mesma logica de nb17.parcelling_labels_grouped), mas
    retorna a PROBABILIDADE continua (band_bad*mult), nao um label sorteado -- a formula
    de valor esperado usa probabilidade, sortear introduziria ruido desnecessario.
    Totalmente vetorizado (sem loop Python linha-a-linha, diferente do parcelling
    original do notebook 17)."""
    band_bad_arr = np.array([band_bad[b] for b in range(len(band_bad))])
    p_bad = band_bad_arr[rej_band] * base_mult
    censored = rej_flags["dti_censored"].to_numpy() == 1
    p_bad = np.where(censored, p_bad * censored_extra, p_bad)
    return np.clip(p_bad, 0.0, 1.0)


def optimal_threshold_ownscale(pd_hat, A, i, lgd, grid=None):
    """SOLUCAO 1: threshold otimo NA ESCALA DO PROPRIO MODELO (maximiza lucro). Nome
    distinto de src.economics.optimal_threshold (formula/paradigma diferente -- realizado
    vs valor esperado -- e nao deve ser confundido com ela)."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_p = None, -np.inf
    for t in grid:
        mask = pd_hat < t
        p = portfolio_profit(pd_hat, A, i, lgd, accept_mask=mask) if mask.any() else 0.0
        if p > best_p:
            best_p, best_t = p, float(t)
    return best_t, best_p


def profit_at_acceptance_rate(pd_hat, A, i, lgd, rate, order=None):
    """SOLUCAO 2: aceita a fracao `rate` de MENOR PD (melhores), pela escala do proprio
    modelo. Comparacao justa entre estrategias: mesma fracao aceita, cada uma escolhe
    quais -- nao depende de nenhum threshold numerico cross-model. `order` (argsort de
    pd_hat) pode ser precomputado e reusado entre LGD/taxa -- nao muda com essas duas,
    so com pd_hat -- pra nao repetir um argsort de 27,5M elementos 60x."""
    if order is None:
        order = np.argsort(pd_hat)
    n_accept = int(len(pd_hat) * rate)
    idx = order[:n_accept]
    return float(np.sum(expected_profit_per_loan(pd_hat[idx], A[idx], i[idx], lgd)))


RATE_GRID = (0.1, 0.2, 0.3, 0.5)
LGD_GRID_ETAPA3 = (0.5, 0.7, 0.9)  # subconjunto representativo (5 LGD x muitas estrategias ficaria denso)


def run_etapa3(etapa2_out):
    """SOLUCAO 1 (teto de cada modelo, escalas proprias) + SOLUCAO 2 (comparacao
    principal, taxa de aceitacao fixa) entre 5 estrategias de pontuacao dos recusados:
    thin model cru, e parcelling com base_mult em {1.5, 2.0} (faixa plausivel do Bloco 3)
    cruzado com censored_extra em {1.0 = sem boost, 1.5 = boost maximo}. Isola o efeito
    do censored_extra: com base_mult uniforme (mesmo fator em toda banda), o RANKING dos
    recusados nao muda (multiplicar tudo pela mesma constante preserva ordem) -- so
    censored_extra, aplicado SELETIVAMENTE aos censored, reordena quem entra nos X%
    aceitos. Por isso a comparacao a taxa fixa e' o teste certo pro criterio (ii)."""
    p_rej = etapa2_out["p_rej"]
    rej_band = etapa2_out["rej_band"]
    band_bad = etapa2_out["band_bad"]
    rej_flags = etapa2_out["rej_flags"]
    A_rej = etapa2_out["A_rej"]
    i_rej = etapa2_out["i_rej"]

    strategies = {"raw_thin (sem parcelling)": p_rej}
    for base_mult in (1.5, 2.0):
        for cx, label in ((1.0, "sem_censored"), (1.5, "com_censored")):
            name = f"base{base_mult}_{label}"
            strategies[name] = parcelling_pd_rejected(rej_band, band_bad, rej_flags, base_mult, cx)

    print("\n=== ETAPA 3a (Solucao 1): threshold otimo NA PROPRIA ESCALA, por estrategia ===")
    print(f"{'estrategia':>28}{'LGD':>6}{'threshold_proprio':>20}{'lucro_no_otimo':>20}")
    for name, pd_hat in strategies.items():
        for lgd in LGD_GRID_ETAPA3:
            t, p = optimal_threshold_ownscale(pd_hat, A_rej, i_rej, lgd)
            print(f"{name:>28}{lgd:>6}{t:>20.3f}{p:>20,.0f}")
    print("  [referencia] threshold otimo do XGB nos aprovados (Etapa 1, LGD=0.5) = 0.260 -- "
          "os valores acima devem ficar bem abaixo disso, coerente com uma escala mais "
          "comprimida (thin model, 3 features fracas vs 79 do XGB).")

    print("\n=== ETAPA 3b (Solucao 2, PRINCIPAL): lucro a TAXA DE ACEITACAO FIXA ===")
    names = list(strategies.keys())
    orders = {n: np.argsort(strategies[n]) for n in names}  # 1 argsort/estrategia, reusado
    header = f"{'LGD':>4}{'taxa':>6}  " + "  ".join(f"{n:>24}" for n in names)
    print(header)
    rows = []
    for lgd in LGD_GRID_ETAPA3:
        for rate in RATE_GRID:
            vals = [profit_at_acceptance_rate(strategies[n], A_rej, i_rej, lgd, rate, order=orders[n])
                    for n in names]
            rows.append((lgd, rate, *vals))
            print(f"{lgd:>4}{rate:>6}  " + "  ".join(f"{v:>24,.0f}" for v in vals))

    df = pd.DataFrame(rows, columns=["lgd", "rate"] + names)

    print("\n[VEREDITO censored_extra] Dentro de cada (LGD, taxa), comparar "
          "base{X}_com_censored vs base{X}_sem_censored (mesmo base_mult, so cx muda). "
          "Se com_censored > sem_censored consistentemente, a flag agrega lucro -> mantida "
          "(fecha o teste ii, Bloco 1d). Se igual ou pior, remover a flag do parcelling.")
    for base_mult in (1.5, 2.0):
        sem = df[f"base{base_mult}_sem_censored"]
        com = df[f"base{base_mult}_com_censored"]
        delta = com - sem
        print(f"  base_mult={base_mult}: com_censored - sem_censored "
              f"varia de {delta.min():,.0f} a {delta.max():,.0f} "
              f"({(delta > 0).sum()}/{len(delta)} combinacoes com melhora)")
    return df


if __name__ == "__main__":
    print("[Fase 2a - Bloco 7/7b] avaliacao por lucro honesta (Kozodoi, LGD varrida)\n")

    pd_hat_appr, A_appr, i_appr, y_appr_test, etapa1_results = validate_on_approved()
    etapa2_out = evaluate_rejected_scenario()
    etapa3_df = run_etapa3(etapa2_out)

    print("\n[Fase 2a - Bloco 7/7b] concluido.")
