"""
Testes de src/cleaning.py -- o ultimo modulo do caminho de serving sem rede.

Fecha a outra metade do P-042. features.py ganhou 20 testes no P-043; cleaning.py
continuava com ZERO, e ele roda em TODA requisicao: score_frame chama clean_record antes
de build_features, aplicando sentinelas, flags e medianas congeladas.

O que cada grupo protege, em ordem de gravidade:

  1. INDEPENDENCIA DE LOTE. E a propriedade que o P-043 violou na funcao vizinha. Aqui
     ela deveria valer por construcao -- tudo e por linha ou vem do JSON congelado -- mas
     "deveria valer por construcao" foi exatamente o que eu escrevi sobre prepare_X no
     P-013, um item antes de achar o P-043. Agora e assert.

  2. MEDIANA LIDA, NUNCA RECALCULADA. O docstring do modulo chama isso de "a classe de
     erro que este projeto evita": recalcular a mediana a partir do lote faz a imputacao
     de serving divergir da de treino. O teste passa lotes com distribuicoes diferentes e
     exige o MESMO valor imputado.

  3. FLAG ANTES DO PREENCHIMENTO. A ordem importa e e facil de inverter numa refatoracao:
     se alguem preencher a origem antes de calcular a flag, a flag vira 0 para todo mundo
     e a informacao de ausencia -- que e MNAR e o projeto trata como informativa --
     desaparece em silencio, sem quebrar nada.

Nao precisa de parquet nem de modelo: cleaning.py depende so de src/_cleaning_stats.json,
que e versionado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.cleaning as cleaning
from src.cleaning import (
    SENTINEL_999_WITH_FLAG,
    SENTINEL_NEG1_WITH_FLAG,
    SENTINEL_999_ROLLOUT,
    SENTINEL_NEG1_ROLLOUT,
    SPARSE_COLS,
    clean_record,
)


MINIMO = {
    "loan_amnt": 10000.0, "installment": 325.5, "term": 36,
    "annual_inc": 60000.0, "fico_range_low": 710.0,
    "earliest_cr_line": "2001-08-01", "issue_d": "2015-06-01",
    "home_ownership": "rent", "purpose": "debt_consolidation",
    "verification_status": "verified", "initial_list_status": "w",
    "application_type": "individual",
    "open_acc": 8.0, "total_acc": 20.0, "revol_bal": 8500.0,
    "inq_last_6mths": 1.0,
}


# ------------------------------------------------- 1. independencia de lote (a grave)

def test_registro_sozinho_e_em_lote_produzem_a_mesma_linha():
    """A propriedade que o P-043 violou na funcao vizinha.

    Se um dia alguem trocar uma mediana congelada por `df[c].median()`, ou uma flag por
    algo que olhe o lote, este teste quebra. Sem ele, a quebra sairia como um score
    ligeiramente diferente conforme a companhia -- que foi exatamente o modo de falha do
    P-043, e ninguem percebe olhando a resposta."""
    a = dict(MINIMO)
    b = dict(MINIMO, annual_inc=20000.0, revol_util=90.0, dti=44.0, open_acc=2.0)

    sozinho = clean_record(pd.DataFrame([a]))
    em_lote = clean_record(pd.DataFrame([a, b]))

    colunas = sorted(set(sozinho.columns) & set(em_lote.columns))
    assert sorted(sozinho.columns) == sorted(em_lote.columns)
    pd.testing.assert_series_equal(
        sozinho[colunas].iloc[0], em_lote[colunas].iloc[0], check_names=False
    )


def test_a_ordem_das_linhas_no_lote_nao_muda_nenhuma_delas():
    """Complemento do anterior: nao basta ser igual sozinho, tem que ser igual em
    qualquer posicao."""
    a = dict(MINIMO)
    b = dict(MINIMO, annual_inc=20000.0, dti=44.0)

    ab = clean_record(pd.DataFrame([a, b])).reset_index(drop=True)
    ba = clean_record(pd.DataFrame([b, a])).reset_index(drop=True)

    colunas = sorted(ab.columns)
    pd.testing.assert_series_equal(
        ab[colunas].iloc[0], ba[colunas].iloc[1], check_names=False
    )


# ---------------------------------------- 2. mediana congelada, nunca recalculada

def test_mediana_imputada_nao_depende_do_lote():
    """"A classe de erro que este projeto evita", nas palavras do proprio docstring.

    Dois lotes com distribuicoes deliberadamente opostas na mesma coluna. Se a mediana
    viesse do lote, os dois valores imputados seriam diferentes."""
    baixo = [dict(MINIMO, revol_util=v) for v in (1.0, 2.0, 3.0)]
    alto = [dict(MINIMO, revol_util=v) for v in (95.0, 96.0, 97.0)]
    faltante = dict(MINIMO, revol_util=np.nan)

    imputado_baixo = clean_record(pd.DataFrame(baixo + [faltante]))["revol_util"].iloc[-1]
    imputado_alto = clean_record(pd.DataFrame(alto + [faltante]))["revol_util"].iloc[-1]

    assert imputado_baixo == imputado_alto


def test_o_valor_imputado_e_exatamente_o_do_json_congelado():
    """Nao basta ser estavel -- tem que ser o numero do treino."""
    stats = json.loads(
        (Path(cleaning.__file__).parent / "_cleaning_stats.json").read_text()
    )["sparse_medians"]

    df = clean_record(pd.DataFrame([dict(MINIMO)]))  # nenhuma sparse informada
    for c in SPARSE_COLS:
        assert df[c].iloc[0] == stats[c], f"{c}: {df[c].iloc[0]} != {stats[c]}"


# ------------------------------------------- 3. flag calculada ANTES do preenchimento

@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_coluna_ausente_marca_a_flag_e_recebe_a_sentinela(origem, flag):
    """A ordem: a flag e calculada antes de a origem ser preenchida.

    Se alguem inverter numa refatoracao, a flag vira 0 para todo mundo e a ausencia --
    que aqui e MNAR e informativa -- some sem quebrar nada."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df[flag].iloc[0] == 1, f"{flag} deveria marcar ausencia"
    assert df[origem].notna().iloc[0], f"{origem} deveria ter recebido sentinela"


@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_coluna_nula_tambem_marca_a_flag(origem, flag):
    """Presente-mas-nula tem que ser tratada como ausente. Duas portas, um resultado."""
    df = clean_record(pd.DataFrame([dict(MINIMO, **{origem: np.nan})]))
    assert df[flag].iloc[0] == 1


@pytest.mark.parametrize("origem,flag", sorted(
    {**SENTINEL_999_WITH_FLAG, **SENTINEL_NEG1_WITH_FLAG}.items()
))
def test_valor_presente_nao_marca_a_flag_nem_e_sobrescrito(origem, flag):
    """§12.2 aplicada a uma flag: no caminho normal ela tem que ficar quieta.

    E o valor informado nao pode ser trocado pela sentinela -- seria perder dado real."""
    df = clean_record(pd.DataFrame([dict(MINIMO, **{origem: 7.0})]))
    assert df[flag].iloc[0] == 0
    assert df[origem].iloc[0] == 7.0


def test_sentinela_999_e_neg1_vao_para_as_colunas_certas():
    """As duas tabelas de decisao nao podem trocar de lugar: 999 preserva a ordenacao
    'maior = mais seguro'; -1 e para contadores, onde essa ordenacao nao existe."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    for c in list(SENTINEL_999_WITH_FLAG) + SENTINEL_999_ROLLOUT:
        assert df[c].iloc[0] == 999.0, f"{c} deveria ser 999"
    for c in list(SENTINEL_NEG1_WITH_FLAG) + SENTINEL_NEG1_ROLLOUT:
        assert df[c].iloc[0] == -1.0, f"{c} deveria ser -1"


# ----------------------------------------------------------- flag agregada e derivadas

def test_sparse_bureau_missing_e_um_OU_sobre_as_seis():
    todas = {c: 1.0 for c in SPARSE_COLS}
    completo = clean_record(pd.DataFrame([dict(MINIMO, **todas)]))
    assert completo["sparse_bureau_missing"].iloc[0] == 0

    for c in SPARSE_COLS:
        faltando_uma = dict(MINIMO, **todas)
        faltando_uma[c] = np.nan
        df = clean_record(pd.DataFrame([faltando_uma]))
        assert df["sparse_bureau_missing"].iloc[0] == 1, f"{c} sozinha deveria ligar a flag"


def test_era_pre_2012_e_sempre_zero_para_registro_novo():
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["era_pre_2012"].iloc[0] == 0


def test_funded_amnt_e_derivado_de_loan_amnt_quando_ausente():
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["funded_amnt"].iloc[0] == MINIMO["loan_amnt"]


def test_funded_amnt_informado_nao_e_sobrescrito():
    df = clean_record(pd.DataFrame([dict(MINIMO, funded_amnt=9000.0)]))
    assert df["funded_amnt"].iloc[0] == 9000.0


@pytest.mark.parametrize("coluna", ["acc_now_delinq", "delinq_amnt", "delinq_2yrs", "pub_rec"])
def test_contadores_de_evento_raro_caem_para_zero(coluna):
    """Ausencia de evento = 0 e o significado do campo, nao um chute -- e a justificativa
    esta escrita no proprio cleaning.py."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df[coluna].iloc[0] == 0.0


# --------------------------------------------------------------- contrato da funcao

def test_nao_muta_o_dataframe_de_entrada():
    entrada = pd.DataFrame([dict(MINIMO)])
    antes = entrada.copy(deep=True)
    clean_record(entrada)
    pd.testing.assert_frame_equal(entrada, antes)


def test_aceita_dict_e_dataframe_com_o_mesmo_resultado():
    de_dict = clean_record(dict(MINIMO))
    de_frame = clean_record(pd.DataFrame([dict(MINIMO)]))
    colunas = sorted(de_dict.columns)
    assert sorted(de_frame.columns) == colunas
    pd.testing.assert_series_equal(
        de_dict[colunas].iloc[0], de_frame[colunas].iloc[0], check_names=False
    )


def test_e_idempotente():
    """score_frame chama clean_record em toda requisicao, e o monitor de drift pode
    receber lote ja limpo. Passar duas vezes nao pode mudar nada."""
    uma = clean_record(pd.DataFrame([dict(MINIMO)]))
    duas = clean_record(uma)
    colunas = sorted(uma.columns)
    pd.testing.assert_frame_equal(uma[colunas], duas[colunas])


def test_coercao_numerica_nao_toca_categorica_nem_data():
    """A coercao defensiva do passo 9 existe porque um Optional omitido chega como None e
    o pandas cria coluna 'object', que o XGBoost recusa. Ela nao pode passar por cima das
    categoricas nem das datas -- build_features e prepare_X usam .dt nessas duas."""
    df = clean_record(pd.DataFrame([dict(MINIMO)]))
    assert df["home_ownership"].iloc[0] == "rent"
    assert df["purpose"].iloc[0] == "debt_consolidation"
    assert df["earliest_cr_line"].iloc[0] == "2001-08-01"
    assert df["issue_d"].iloc[0] == "2015-06-01"


def test_a_condicao_do_passo_9_nao_pode_ser_dtype_igual_object():
    """GUARD do P-048, e a razao de ele existir como teste separado.

    A condicao original era `df[c].dtype == object`. Estava CERTA quando foi escrita e
    ficou errada sozinha: no pandas >= 3.0 uma coluna de strings recebe a StringDtype
    dedicada (imprime como `str`), nao object -- entao a comparacao dava False e a coercao
    nunca rodava justamente no caso para o qual ela existe. Reproduzido no pandas 3.0.2.

    Este teste nao olha o codigo; olha o COMPORTAMENTO sob a dtype nova. Se alguem voltar a
    condicao para `== object`, ele quebra.

    Licao transferivel, e e diferente das outras tres da familia do §16: ali o contrato
    prometia algo que o codigo nao fazia. Aqui o codigo fazia, e parou de fazer porque uma
    DEPENDENCIA mudou embaixo dele. Proteger contra isso exige teste de comportamento, nao
    revisao de codigo -- revisao nenhuma pega isso, porque no dia em que foi escrito estava
    certo."""
    entrada = pd.DataFrame([dict(MINIMO, revol_bal="8500")])
    assert entrada["revol_bal"].dtype != object, (
        "pandas antigo: este teste perde o sentido, mas nao fica errado"
    )
    df = clean_record(entrada)
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])


def test_coluna_numerica_que_chega_como_texto_vira_numero():
    df = clean_record(pd.DataFrame([dict(MINIMO, revol_bal="8500")]))
    assert pd.api.types.is_numeric_dtype(df["revol_bal"])
    assert df["revol_bal"].iloc[0] == 8500.0


def test_valor_numerico_impossivel_vira_NaN_em_vez_de_explodir():
    """errors='coerce': entrada suja vira NaN (erro limpo mais adiante) em vez de 500."""
    df = clean_record(pd.DataFrame([dict(MINIMO, revol_bal="oito mil")]))
    assert pd.isna(df["revol_bal"].iloc[0])
