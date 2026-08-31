"""
Testes unitarios de src/features.py -- build_features e prepare_X.

Cobre o item Data 7 do ML Test Score ("all input feature code is tested"), que a
medicao de 2026-08-30 encontrou em ZERO: tests/ cobria economics.py e psi.py, mas
nao o codigo de criacao de feature.

O teste mais importante deste arquivo e test_build_features_e_puro_por_linha: ele
transforma em teste executavel a afirmacao do docstring de build_features ("does not
read, receive, or reference any other dataset"). Essa propriedade e a razao de o
pipeline nao ter vazamento de estado entre treino e inferencia; sem teste, e so uma
promessa em prosa.

Os dois testes test_REGRESSAO_* sao o guard do BUG P-043, achado enquanto este arquivo
era escrito e corrigido em 2026-08-30. O bug: com drop_first=True e um lote de 1 linha,
get_dummies produz ZERO colunas categoricas, e o reindex(fill_value=0) de score_frame
preenchia tudo com 0 em silencio -- como a API e single-record, TODA requisicao era
pontuada como se o candidato fosse a categoria-base.

Historico verificado: em 2026-08-30, ANTES da correcao, estes dois testes rodaram
marcados como falha esperada e o pytest reportou "2 xfailed" -- ou seja, falharam de
verdade, reproduzindo o bug. Contra o modelo real, o mesmo registro dava p=0.1341794431
sozinho e p=0.1289446801 em lote. A correcao (drop_first=False na inferencia + reindex)
os fez passar; o marcador foi entao removido e eles viraram testes de regressao normais.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLS, REFERENCE_DATE
from src.features import build_features, prepare_X


def _frame_minimo(n=3):
    """Frame com as colunas que build_features consome, valores distintos por linha."""
    return pd.DataFrame({
        "installment": [300.0, 500.0, 150.0][:n],
        "annual_inc": [60000.0, 120000.0, 30000.0][:n],
        "loan_amnt": [10000.0, 25000.0, 5000.0][:n],
        "revol_bal": [8500.0, 20000.0, 1000.0][:n],
        "open_acc": [8.0, 12.0, 4.0][:n],
        "total_acc": [20.0, 30.0, 10.0][:n],
        "issue_d": pd.to_datetime(["2015-06-01", "2016-01-01", "2014-11-15"][:n]),
        "earliest_cr_line": pd.to_datetime(["2001-08-01", "1998-03-01", "2010-03-15"][:n]),
    })


# --- build_features: pureza por linha (a propriedade que sustenta o pipeline) ---

def test_build_features_e_puro_por_linha():
    """Processar um subconjunto da vez tem que dar exatamente o mesmo resultado que
    processar o frame inteiro e depois fatiar. Se build_features lesse qualquer
    estatistica de outras linhas (media, mediana, min/max), este teste quebraria."""
    df = _frame_minimo(3)
    inteiro = build_features(df)

    derivadas = ["installment_to_income", "loan_to_income", "credit_history_months",
                 "revol_bal_to_income", "open_acc_ratio"]

    for i in range(len(df)):
        uma_linha = build_features(df.iloc[[i]])
        pd.testing.assert_frame_equal(
            uma_linha[derivadas].reset_index(drop=True),
            inteiro[derivadas].iloc[[i]].reset_index(drop=True),
        )


def test_build_features_nao_muta_a_entrada():
    df = _frame_minimo()
    colunas_antes = list(df.columns)
    build_features(df)
    assert list(df.columns) == colunas_antes


# --- build_features: cada derivada, calculada a mao ---

def test_installment_to_income_e_parcela_sobre_renda_mensal():
    df = build_features(_frame_minimo(1))
    assert df["installment_to_income"].iloc[0] == pytest.approx(300.0 / (60000.0 / 12))


def test_loan_to_income_usa_renda_anual_nao_mensal():
    df = build_features(_frame_minimo(1))
    assert df["loan_to_income"].iloc[0] == pytest.approx(10000.0 / 60000.0)


def test_revol_bal_to_income_usa_renda_anual():
    df = build_features(_frame_minimo(1))
    assert df["revol_bal_to_income"].iloc[0] == pytest.approx(8500.0 / 60000.0)


def test_open_acc_ratio():
    df = build_features(_frame_minimo(1))
    assert df["open_acc_ratio"].iloc[0] == pytest.approx(8.0 / 20.0)


def test_credit_history_months_conta_meses_atravessando_o_ano():
    """2001-08 -> 2015-06: 14 anos completos menos 2 meses = 166 meses."""
    df = build_features(_frame_minimo(1))
    assert df["credit_history_months"].iloc[0] == (2015 - 2001) * 12 + (6 - 8)
    assert df["credit_history_months"].iloc[0] == 166


def test_credit_history_months_pode_ser_negativo_se_as_datas_estiverem_invertidas():
    """Documenta o comportamento atual: a funcao nao valida a ordem das datas.
    Um registro com earliest_cr_line depois de issue_d produz valor negativo em vez
    de erro. Quem valida ordem e o schema da API, nao build_features."""
    df = _frame_minimo(1)
    df["earliest_cr_line"] = pd.to_datetime(["2020-01-01"])
    out = build_features(df)
    assert out["credit_history_months"].iloc[0] < 0


# --- build_features: a borda que gera infinito (ligada a P-011) ---

def test_renda_zero_produz_infinito_nas_tres_razoes_de_renda():
    """ACHADO, nao bug corrigido aqui: annual_inc == 0 gera +inf em
    installment_to_income, loan_to_income e revol_bal_to_income (divisao por zero
    em float64 nao levanta, retorna inf).

    build_features NAO trata isso de proposito -- e transformacao pura por linha, sem
    politica. Quem barra o caso e o guard de finitude em src.guards, chamado no
    caminho de inferencia (score_frame). Este teste existe para que a borda fique
    registrada e para quebrar se alguem mudar o comportamento sem querer."""
    df = _frame_minimo(1)
    df["annual_inc"] = [0.0]
    out = build_features(df)
    assert np.isinf(out["installment_to_income"].iloc[0])
    assert np.isinf(out["loan_to_income"].iloc[0])
    assert np.isinf(out["revol_bal_to_income"].iloc[0])


def test_total_acc_zero_produz_nan_em_open_acc_ratio():
    """0/0 em float64 da NaN (nao inf). NaN e entrada legitima pro XGBoost, que tem
    direcao default aprendida -- por isso o guard de finitude barra inf, nao NaN."""
    df = _frame_minimo(1)
    df["open_acc"] = [0.0]
    df["total_acc"] = [0.0]
    out = build_features(df)
    assert np.isnan(out["open_acc_ratio"].iloc[0])


# --- prepare_X: determinismo e ausencia de estado ---

def test_prepare_X_e_deterministico():
    df = build_features(_frame_minimo())
    cols = ["installment", "annual_inc", "issue_d", "earliest_cr_line", "home_ownership"]
    df["home_ownership"] = ["rent", "own", "mortgage"]
    a = prepare_X(df, cols, ["home_ownership"])
    b = prepare_X(df, cols, ["home_ownership"])
    pd.testing.assert_frame_equal(a, b)


def test_prepare_X_nao_guarda_estado_entre_chamadas():
    """Chamar com um frame nao pode influenciar o resultado do frame seguinte.
    Se prepare_X guardasse categorias vistas (como um encoder com fit), quebraria."""
    cols = ["annual_inc", "home_ownership"]
    df1 = pd.DataFrame({"annual_inc": [1.0, 2.0], "home_ownership": ["rent", "own"]})
    df2 = pd.DataFrame({"annual_inc": [3.0], "home_ownership": ["rent"]})

    sozinho = prepare_X(df2, cols, ["home_ownership"])
    prepare_X(df1, cols, ["home_ownership"])
    depois = prepare_X(df2, cols, ["home_ownership"])
    pd.testing.assert_frame_equal(sozinho, depois)


def test_prepare_X_converte_datas_para_dias_desde_a_referencia():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2015-06-01"]),
        "earliest_cr_line": pd.to_datetime(["2001-08-01"]),
    })
    X = prepare_X(df, ["issue_d", "earliest_cr_line"], [])
    assert REFERENCE_DATE == pd.Timestamp("2000-01-01")
    assert X["issue_d"].iloc[0] == 5630
    assert X["earliest_cr_line"].iloc[0] == 578


def test_prepare_X_faz_one_hot_com_drop_first():
    """3 categorias -> 2 colunas (a primeira em ordem alfabetica e a base)."""
    df = pd.DataFrame({"home_ownership": ["rent", "own", "mortgage"]})
    X = prepare_X(df, ["home_ownership"], ["home_ownership"])
    assert "home_ownership_mortgage" not in X.columns
    assert set(X.columns) == {"home_ownership_own", "home_ownership_rent"}


def test_prepare_X_ignora_categorica_que_nao_esta_em_feature_cols():
    """categorical_cols pode listar colunas ausentes sem quebrar (cat_present)."""
    df = pd.DataFrame({"annual_inc": [1.0]})
    X = prepare_X(df, ["annual_inc"], CATEGORICAL_COLS)
    assert list(X.columns) == ["annual_inc"]


# --- prepare_X: o risco que o docstring nomeia, e o bug que ele esconde ---

def test_prepare_X_muda_colunas_com_categorias_diferentes():
    """O docstring de prepare_X avisa: 'Column set/order can differ between two
    different calls if the underlying categorical columns don't share the same
    categories - callers must reindex'.

    Este teste prova que o aviso e real."""
    cols = ["home_ownership"]
    treino = pd.DataFrame({"home_ownership": ["rent", "own", "mortgage"]})
    lote = pd.DataFrame({"home_ownership": ["rent", "own"]})

    X_treino = prepare_X(treino, cols, cols)
    X_lote = prepare_X(lote, cols, cols)

    assert list(X_treino.columns) == ["home_ownership_own", "home_ownership_rent"]
    assert list(X_lote.columns) == ["home_ownership_rent"]  # drop_first removeu 'own'
    assert list(X_treino.columns) != list(X_lote.columns)


def test_REGRESSAO_um_registro_sozinho_mantem_as_categoricas():
    """GUARD do P-043. Um registro sozinho tem que ser codificado corretamente.

    Antes da correcao: drop_first=True removia a unica categoria presente, get_dummies
    produzia ZERO colunas, e o reindex preenchia tudo com 0 -- o candidato virava a
    categoria-base. Como a API e single-record (src/api.py: `def score(req:
    ScoreRequest)`), isso valia para TODA requisicao."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    um = pd.DataFrame({"home_ownership": ["rent"]})
    X = prepare_X(um, cols, cols, drop_first=False)
    X_alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)

    assert X_alinhado["home_ownership_rent"].iloc[0] == 1
    assert X_alinhado["home_ownership_own"].iloc[0] == 0


def test_categoria_base_sozinha_fica_com_todas_as_dummies_em_zero():
    """A outra metade da equivalencia, e a que quase ninguem testa.

    A categoria-base ('mortgage', primeira em ordem alfabetica) nao tem coluna no
    treino. Com drop_first=False ela GANHA uma coluna, que o reindex descarta por nao
    estar na lista treinada -- deixando o grupo todo em zero, que e exatamente como a
    base e representada no treino. Sem este teste, a correcao poderia estar certa para
    3 das 4 categorias e errada justamente para a base."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    um = pd.DataFrame({"home_ownership": ["mortgage"]})
    X = prepare_X(um, cols, cols, drop_first=False)
    assert "home_ownership_mortgage" in X.columns

    X_alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)
    assert list(X_alinhado.columns) == colunas_de_treino
    assert X_alinhado.sum(axis=1).iloc[0] == 0


def test_REGRESSAO_mesmo_registro_independe_do_lote():
    """GUARD do P-043, segunda face e a mais grave.

    Antes da correcao o MESMO registro recebia encoding diferente dependendo de quem
    mais estava no lote, porque drop_first escolhe a base pelas categorias PRESENTES
    naquele lote. Independencia de linha e propriedade nao-negociavel de um scorer."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]
    reg = {"home_ownership": "rent"}
    outro = {"home_ownership": "own"}

    sozinho = prepare_X(pd.DataFrame([reg]), cols, cols, drop_first=False).reindex(
        columns=colunas_de_treino, fill_value=False)
    em_lote = prepare_X(pd.DataFrame([reg, outro]), cols, cols, drop_first=False).reindex(
        columns=colunas_de_treino, fill_value=False)

    # compara VALOR, e tambem dtype -- com fill_value=False (e nao 0) as duas matrizes
    # ficam bool nas duas situacoes. Com fill_value=0 os valores batiam (0 == False) mas
    # o dtype nao (int64 vs bool), e foi assim que este teste falhou na primeira
    # tentativa da correcao, em 2026-08-30.
    assert sozinho.iloc[0].tolist() == em_lote.iloc[0].tolist()
    assert list(sozinho.dtypes) == list(em_lote.dtypes), (
        f"dtype depende do lote: {dict(sozinho.dtypes)} vs {dict(em_lote.dtypes)}"
    )


def test_drop_first_True_na_inferencia_ainda_falha_em_silencio():
    """Por que a correcao teve que ser no CHAMADOR e nao no reindex.

    Este teste mantem vivo o comportamento antigo (drop_first=True numa linha so) para
    documentar POR QUE ele e perigoso: reindex(fill_value=0) preenche coluna AUSENTE
    com 0, que e indistinguivel de 'a categoria-base foi observada'. Nao levanta, nao
    loga, passa no CI -- a resposta continua bem-formada.

    Licao transferivel: valor de preenchimento que coincide com um valor legitimo
    transforma erro em silencio. Verificar que a protecao existe nao e o mesmo que
    verificar o que ela recebe."""
    cols = ["home_ownership"]
    colunas_de_treino = ["home_ownership_own", "home_ownership_rent"]

    X = prepare_X(pd.DataFrame({"home_ownership": ["rent"]}), cols, cols, drop_first=True)
    alinhado = X.reindex(columns=colunas_de_treino, fill_value=0)

    assert list(alinhado.columns) == colunas_de_treino   # forma correta
    assert alinhado.sum(axis=1).iloc[0] == 0             # conteudo silenciosamente errado
