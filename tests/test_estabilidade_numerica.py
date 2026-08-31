"""
Estabilidade numerica: NaN e Inf (P-011) e a divergencia de encoding do P-046.

Por que existe, e por que o guard nao e um so:

  build_features divide por annual_inc (tres razoes) e por total_acc (uma). Zero em
  qualquer das duas produz Inf ou NaN. Medido sobre o split de treino congelado em
  2026-08-31: 0 de 90 colunas com NaN ou Inf, 172.988 linhas, e ZERO linhas com
  annual_inc == 0 ou total_acc == 0.

  Ou seja: no caminho de treino o assert nao pode disparar por causa do dado -- o parquet
  esta congelado e foi medido limpo. Ele dispara quando ALGUEM MUDA O CODIGO DE FEATURE.
  E para isso que serve.

  O caminho que precisa de guard de verdade e o de SERVING, onde a entrada nao e
  congelada nem medida. La o guard e sobre a SAIDA (probabilidade nao-finita nunca e
  legitima e nao da pra tratar depois), e a origem provavel -- o contrato da API aceitar
  annual_inc >= 0 e total_acc >= 0, regiao onde o treino tem zero linhas -- e o P-047.

  assert_matriz_finita NAO e chamada dentro de prepare_X de proposito: prepare_X e
  compartilhada pelos dois caminhos, e levantar excecao no serving mataria o monitor de
  drift numa unica linha suja. E o mesmo raciocinio que removeu o aviso do P-043.
"""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_record
from src.data import FEATURE_SET, CATEGORICAL_COLS
from src.features import assert_matriz_finita, build_features, prepare_X
from src.scoring import load_model, score_frame, _normalize_dates


BASE = {
    "loan_amnt": 10000.0, "installment": 325.5, "term": 36,
    "annual_inc": 60000.0, "fico_range_low": 710.0, "dti": 15.2,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": "rent", "purpose": "debt_consolidation",
    "verification_status": "verified", "initial_list_status": "w",
    "application_type": "individual",
    "acc_open_past_24mths": 3.0, "open_acc": 8.0, "total_acc": 20.0,
    "revol_bal": 8500.0, "revol_util": 42.3, "inq_last_6mths": 1.0,
}


def _matriz(rec):
    d = clean_record(_normalize_dates(pd.DataFrame([rec])))
    return prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS, drop_first=False)


# ----------------------------------------------------- o guard fica quieto no normal

def test_matriz_de_um_registro_valido_passa_em_silencio():
    """§12.2: guard que fala no caminho feliz e pior que guard nenhum.

    Se este teste falhar, o assert do P-011 esta gritando lobo e nao deve ser commitado --
    foi exatamente assim que o aviso do P-043 morreu."""
    assert_matriz_finita(_matriz(BASE), "registro valido")


def test_o_guard_nao_esta_dentro_de_prepare_X():
    """Decisao explicita, mantida viva como teste.

    prepare_X e compartilhada por treino e serving. Se alguem mover a checagem para
    dentro dela, uma linha suja passa a derrubar o monitor de drift inteiro. Este teste
    quebra quando isso acontecer."""
    rec = dict(BASE, annual_inc=0.0)
    X = _matriz(rec)                      # nao pode levantar
    assert np.isinf(X["loan_to_income"].iloc[0])


# ------------------------------------------------------------- o guard pega o que deve

def test_guard_pega_NaN_e_nomeia_a_coluna():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan]})
    with pytest.raises(ValueError, match=r"NaN in \['b'\]"):
        assert_matriz_finita(X)


def test_guard_pega_Inf_e_nomeia_a_coluna():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    with pytest.raises(ValueError, match=r"Inf in \['b'\]"):
        assert_matriz_finita(X)


def test_guard_ignora_coluna_bool():
    """As 16 one-hot sao bool. isinf sobre bool levanta TypeError se o filtro numerico
    nao estiver certo -- este teste trava o filtro."""
    X = pd.DataFrame({"num": [1.0], "dummy": [True]})
    assert_matriz_finita(X)


@pytest.mark.parametrize("campo,valor,rotulo", [
    ("annual_inc", 0.0, "renda zero -> Inf em installment_to_income, loan_to_income e revol_bal_to_income"),
    ("total_acc", 0.0, "total_acc zero -> Inf ou NaN em open_acc_ratio"),
])
def test_guard_pega_o_mecanismo_real_e_nao_so_um_NaN_sintetico(campo, valor, rotulo):
    """Liga o guard a causa de verdade.

    O treino tem ZERO linhas nos dois casos (medido: 0 de 172.988), mas o contrato da API
    aceita os dois hoje, porque usa ge=0. Enquanto o P-047 nao estreitar o contrato, este
    e o caminho pelo qual Inf entra no modelo em producao."""
    X = _matriz(dict(BASE, **{campo: valor}))
    with pytest.raises(ValueError):
        assert_matriz_finita(X, rotulo)


# ------------------------------------------------------ o guard do lado de serving

def test_score_de_registro_valido_devolve_probabilidade_finita():
    out = score_frame(pd.DataFrame([BASE]))
    p = out["probability_default"].iloc[0]
    assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_score_frame_levanta_se_a_probabilidade_sair_nao_finita():
    """A unica forma honesta de testar um guard cujo gatilho real e inalcancavel hoje.

    Nao existe entrada conhecida que faca o modelo devolver NaN -- por isso o guard e
    barato e silencioso. Mas 'nao sei como disparar' nao e prova de que ele funciona, e um
    guard nunca exercitado e decoracao. O fake abaixo troca so a saida do predict_proba."""
    class _ModeloQueDevolveNaN:
        def __init__(self, real):
            self._real = real

        def get_booster(self):
            return self._real.get_booster()

        def predict_proba(self, X):
            p = self._real.predict_proba(X)
            p[:, 1] = np.nan
            return p

    with pytest.raises(ValueError, match="nao finito"):
        score_frame(pd.DataFrame([BASE]), model=_ModeloQueDevolveNaN(load_model()))


# ------------------------------------------------------- o mecanismo do P-046

def test_drop_first_sobre_subconjunto_produz_conjunto_de_colunas_diferente():
    """Documenta POR QUE o verify_pipeline precisa do guard do P-046.

    O reindex la e a mesma construcao que foi o bug P-043 no serving. Hoje e seguro por
    propriedade do DADO -- o split inteiro tem todas as categorias --, nao do codigo. Este
    teste mostra a propriedade falhando assim que o frame deixa de cobrir o vocabulario:
    e o cenario exato que o guard passa a recusar em voz alta, em vez de produzir um
    profit plausivel e errado."""
    cols = ["home_ownership"]
    completo = pd.DataFrame({"home_ownership": ["mortgage", "own", "rent"]})
    parcial = pd.DataFrame({"home_ownership": ["own", "rent"]})

    X_completo = prepare_X(completo, cols, cols)          # drop_first=True: base 'mortgage'
    X_parcial = prepare_X(parcial, cols, cols)            # drop_first=True: base 'own'

    assert list(X_completo.columns) == ["home_ownership_own", "home_ownership_rent"]
    assert list(X_parcial.columns) == ["home_ownership_rent"]
    assert set(X_completo.columns) != set(X_parcial.columns), (
        "se estes conjuntos passarem a ser iguais, o guard do P-046 perdeu o motivo"
    )
