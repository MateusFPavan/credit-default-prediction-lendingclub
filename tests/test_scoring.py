"""
Testes de src/scoring.py -- o caminho de inferencia, contra o modelo real.

Existem por causa do BUG P-043 (2026-08-30): a API aceitava, validava e IGNORAVA
home_ownership, purpose, verification_status e initial_list_status, e o mesmo
candidato recebia score diferente dependendo de quem mais estava no lote.

Por que nenhuma rede existente pegou, e por que estes testes sao a rede certa:
  - verify_pipeline.py roda com train/test INTEIROS -- todas as categorias presentes,
    encoding correto, profit reproduz ao centavo. Nao ve o bug.
  - o smoke test do CI confere que a resposta tem probability_default e decision.
    Ambos existiam. A resposta era sintaticamente perfeita e semanticamente errada.
  - test_api.py::test_score_valid_is_coherent confere faixa 0<=p<=1 e coerencia
    decisao<->threshold. Ambas continuavam validas.

Nenhum deles testava SENSIBILIDADE: que mudar uma feature mude o score. Esse e o unico
teste que pega esta classe de bug, porque o modo de falha produz um valor legal.

Nao precisam de parquet: score_frame depende so de models/xgb_final.joblib e de
src/_cleaning_stats.json, ambos versionados.

Categoria desconhecida continua sendo pontuada em silencio como a categoria-base. Isso
NAO foi resolvido aqui, e o motivo esta em test_categoria_base_e_indistinguivel_de_
desconhecida_no_artefato: os dois casos sao indistinguiveis a partir do .joblib.
Rastreado em P-044.
"""
import pandas as pd
import pytest

from src.scoring import score_frame


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


def _p(rec: dict) -> float:
    return float(score_frame(pd.DataFrame([rec]))["probability_default"].iloc[0])


# --- sensibilidade: o teste que faltava ---

@pytest.mark.parametrize("campo,a,b", [
    ("home_ownership", "rent", "own"),
    ("purpose", "debt_consolidation", "small_business"),
    ("verification_status", "verified", "not verified"),
    ("initial_list_status", "w", "f"),
])
def test_feature_categorica_afeta_o_score(campo, a, b):
    """Mudar a categorica, com todo o resto igual, TEM que mudar a probabilidade.

    Antes do P-043 os quatro campos davam spread=0.0000000000 -- p=0.1341794431 para
    qualquer valor. Se este teste voltar a falhar, o encoding de inferencia regrediu."""
    ra = dict(BASE); ra[campo] = a
    rb = dict(BASE); rb[campo] = b
    assert _p(ra) != _p(rb), (
        f"{campo} nao afeta o score: '{a}' e '{b}' dao a mesma probabilidade. "
        "E o sintoma do P-043."
    )


# --- independencia de linha ---

def test_score_de_um_registro_independe_do_lote():
    """O score de uma linha nao pode depender das outras linhas do frame.

    Antes do P-043: 0.1341794431 sozinho vs 0.1289446801 em lote (diferenca de
    0.0052347630). Um candidato recebia decisao diferente conforme a companhia."""
    outro = dict(BASE)
    outro.update(home_ownership="own", purpose="medical",
                 initial_list_status="f", verification_status="not verified")

    sozinho = _p(BASE)
    em_lote = float(
        score_frame(pd.DataFrame([BASE, outro]))["probability_default"].iloc[0]
    )
    assert sozinho == em_lote, (
        f"mesmo registro: {sozinho} sozinho vs {em_lote} em lote"
    )


def test_lote_inteiro_bate_linha_a_linha():
    """Analogo offline do item Monitor 3 do ML Test Score (train/serving skew).

    Pontuar N registros de uma vez tem que dar exatamente o mesmo que pontuar cada um
    sozinho. E a forma geral do teste acima, e o guard de qualquer regressao futura
    que faca o encoding depender do conteudo do lote."""
    variantes = []
    for ho in ("rent", "own", "mortgage", "other"):
        for pu in ("debt_consolidation", "medical", "car"):
            r = dict(BASE); r["home_ownership"] = ho; r["purpose"] = pu
            variantes.append(r)

    em_lote = score_frame(pd.DataFrame(variantes))["probability_default"].tolist()
    um_a_um = [_p(r) for r in variantes]
    assert em_lote == um_a_um


# --- categoria desconhecida: lacuna conhecida, documentada, NAO resolvida ---

def test_categoria_desconhecida_pontua_como_categoria_base():
    """LACUNA CONHECIDA (P-044), nao um comportamento desejado.

    Uma categoria nunca vista no treino encoda como todas-as-dummies-zero, ou seja, e
    pontuada como a categoria-base -- em silencio. E o comportamento do
    OneHotEncoder(handle_unknown='ignore') do scikit-learn, cujo default e 'error'
    justamente porque silencio e perigoso.

    Por que nao ha aviso: 'categoria desconhecida' e 'categoria-base' sao
    INDISTINGUIVEIS a partir do artefato. A base tambem nao tem coluna (drop_first a
    removeu no treino). Uma primeira tentativa de aviso foi escrita e removida em
    2026-08-30 porque disparava em TODA requisicao -- application_type tem ZERO colunas
    treinadas, entao seu unico valor legitimo ('individual') era sinalizado como
    desconhecido. Distinguir exige o vocabulario de treino congelado em disco, do jeito
    que _cleaning_stats.json ja congela as medianas. Rastreado em P-044.

    Este teste existe para que a lacuna seja EXPLICITA e para quebrar se alguem mudar o
    comportamento sem atualizar a decisao."""
    desconhecida = dict(BASE); desconhecida["home_ownership"] = "categoria_inexistente"
    base = dict(BASE); base["home_ownership"] = "mortgage"  # a categoria-base real

    assert _p(desconhecida) == _p(base)


def test_categoria_base_e_indistinguivel_de_desconhecida_no_artefato():
    """O fato estrutural que justifica P-044, como teste em vez de comentario.

    As colunas treinadas nao contem a categoria-base de nenhuma coluna categorica.
    application_type e o caso extremo: ZERO colunas treinadas, porque no treino havia
    um valor so e drop_first o eliminou. Logo 'individual' -- valor legitimo, presente
    em todo registro -- nao aparece na lista treinada, exatamente como um valor
    inventado nao apareceria."""
    from src.scoring import load_model, _trained_columns
    treinadas = _trained_columns(load_model())

    assert not [c for c in treinadas if c.startswith("application_type_")], (
        "application_type deveria ter zero colunas treinadas"
    )
    assert "home_ownership_mortgage" not in treinadas, (
        "a categoria-base de home_ownership nao tem coluna, por construcao"
    )


# --- sanidade basica do contrato ---

def test_score_e_deterministico():
    assert _p(BASE) == _p(BASE)


def test_decisao_concorda_com_threshold():
    out = score_frame(pd.DataFrame([BASE]))
    p = out["probability_default"].iloc[0]
    d = out["decision"].iloc[0]
    assert (d == "reject") == (p >= 0.31)
