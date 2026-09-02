"""
Paridade TREINO <-> INFERENCIA contra o artefato real (P-012 / ML Test Score Monitor 3).

Por que este arquivo existe separado de tests/test_features.py:

  test_features.py prova independencia de linha DENTRO da inferencia -- o mesmo registro
  sozinho vs em lote -- em frames sinteticos de UMA coluna categorica, com a lista de
  colunas treinadas escrita a mao. Isso e necessario e nao e suficiente.

  O Monitor 3 pergunta outra coisa: a matriz que o TREINO montou e a matriz que a
  INFERENCIA monta, para as MESMAS linhas, sao iguais elemento a elemento? Responder isso
  exige as 90 colunas do artefato de verdade, porque o caso patologico so aparece em
  escala real -- nenhum frame sintetico de duas colunas o alcanca.

  O caso que motivou este arquivo era application_type, com ZERO colunas treinadas por ser
  constante no treino. Ele foi REMOVIDO do FEATURE_SET em 2026-08-31 (P-045), depois de uma
  ablacao com diferenca de exatamente $0.00. O motivo do arquivo nao mudou: a categoria-BASE
  de toda coluna categorica continua sem coluna treinada, que e a mesma patologia numa forma
  que nao da para remover.

  Este arquivo nasceu do bloco D do _scratch_p043_prova.py, que fazia exatamente essa
  comparacao e passou -- mas morava num scratch fora do repo, ou seja, a unica prova em
  escala real estava num arquivo destinado a ser apagado. Um teste que so existe uma vez
  nao e uma rede.

Nao precisa de parquet: depende so de models/xgb_final.joblib e src/_cleaning_stats.json,
ambos versionados. O vocabulario de categorias e DERIVADO do artefato, nunca digitado --
se um retreino mudar as categorias, os testes se ajustam sozinhos, menos os dois que
guardam decisoes nomeadas (P-045 e P-044), que devem falhar de proposito.
"""
import pandas as pd
import pytest

from src.cleaning import clean_record
from src.data import FEATURE_SET, CATEGORICAL_COLS
from src.features import build_features, prepare_X
from src.scoring import load_model, score_frame, _normalize_dates, _trained_columns


# Um registro valido qualquer. Os valores numericos nao importam para a paridade de
# encoding -- o que importa e que sejam os MESMOS nas duas matrizes comparadas.
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

# Vale como a categoria-BASE de qualquer coluna: nao e nenhuma categoria treinada, e
# ordena antes de todas elas ('_' = 0x5F < 'a' = 0x61), entao pd.get_dummies(drop_first=
# True) descarta justamente ele -- que e o que o treino fez com a base de verdade.
# O nome real da base NAO e recuperavel do artefato (e a lacuna P-044), e este truque
# existe para nao precisar dele: reproduz a ESTRUTURA do encoding de treino sem digitar
# nenhuma categoria. A ordenacao e verificada em test_o_placeholder_ordena_antes_de_tudo.
PLACEHOLDER = "__base__"


# --------------------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def treinadas():
    """As 90 colunas na ordem exata do treino, lidas do proprio .joblib."""
    return _trained_columns(load_model())


def _onehot(treinadas):
    return [c for c in treinadas if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)]


def _categorias_por_coluna(treinadas):
    """{coluna categorica: [categorias com coluna treinada]}, derivado do artefato.

    A categoria-base de cada coluna NAO aparece aqui: drop_first removeu a coluna dela no
    treino. Uma coluna com lista vazia era constante no treino (ver P-045)."""
    return {
        p: sorted(c[len(p) + 1:] for c in treinadas if c.startswith(p + "_"))
        for p in CATEGORICAL_COLS
    }


def _frame_cobrindo_o_vocabulario(treinadas):
    """Frame que contem TODAS as categorias treinadas, mais a base.

    E a condicao para reproduzir o encoding de treino: get_dummies escolhe a base entre as
    categorias PRESENTES na chamada, entao so um frame que cobre o vocabulario inteiro
    escolhe a mesma base que o treino escolheu. A linha 0 carrega o PLACEHOLDER em todas
    as colunas -- e a linha da categoria-base. As demais percorrem as categorias nomeadas.
    """
    cats = _categorias_por_coluna(treinadas)
    n_linhas = max([len(v) for v in cats.values()] + [0]) + 1
    linhas = []
    for i in range(n_linhas):
        rec = dict(BASE)
        for coluna, valores in cats.items():
            if i == 0 or not valores:
                rec[coluna] = PLACEHOLDER
            else:
                rec[coluna] = valores[(i - 1) % len(valores)]
        linhas.append(rec)
    return pd.DataFrame(linhas)


def _matriz(df, drop_first, treinadas):
    """A matriz que chega ao modelo, pelos mesmos passos de score_frame."""
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS, drop_first=drop_first)
    return X.reindex(columns=treinadas, fill_value=False)


# ------------------------------------------------- pre-condicoes do proprio andaime

def test_nenhuma_categorica_e_prefixo_de_outra():
    """Se fosse, _categorias_por_coluna atribuiria colunas a coluna errada em silencio."""
    for a in CATEGORICAL_COLS:
        for b in CATEGORICAL_COLS:
            assert a == b or not b.startswith(a + "_"), f"{a} e prefixo de {b}"


def test_o_placeholder_ordena_antes_de_tudo(treinadas):
    """O truque do PLACEHOLDER so funciona se ele for a primeira categoria em ordem.

    Se um retreino introduzir categoria que ordene antes de '__base__', este teste falha e
    avisa que o andaime -- nao o codigo -- precisa mudar."""
    for coluna, valores in _categorias_por_coluna(treinadas).items():
        for v in valores:
            assert PLACEHOLDER < v, f"{coluna}: '{v}' ordena antes do placeholder"


def test_o_frame_cobre_exatamente_o_vocabulario_de_treino(treinadas):
    """Prova que o frame reproduz o encoding de treino, e nao um parecido.

    Com drop_first=True (o default, que E o caminho de treino) sobre este frame,
    get_dummies tem que produzir EXATAMENTE o conjunto de colunas one-hot treinadas:
    nenhuma faltando (senao o reindex preencheria, e a comparacao seria contra uma matriz
    inventada) e nenhuma sobrando (senao o frame tem categoria que o treino nao viu)."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    d = clean_record(_normalize_dates(df.copy()))
    X = prepare_X(build_features(d), FEATURE_SET, CATEGORICAL_COLS)  # drop_first=True
    produzidas = {c for c in X.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLS)}
    assert produzidas == set(_onehot(treinadas)), (
        f"faltando: {sorted(set(_onehot(treinadas)) - produzidas)} / "
        f"sobrando: {sorted(produzidas - set(_onehot(treinadas)))}"
    )


# ------------------------------------------------------------------- o teste do P-012

def test_matriz_de_treino_e_de_inferencia_batem_linha_a_linha(treinadas):
    """O assert que o Monitor 3 pede, contra as 90 colunas do artefato.

    Esquerda: o frame INTEIRO por drop_first=True -- como os notebooks 06-13 montaram a
    matriz de treino. Direita: cada linha SOZINHA por drop_first=False + reindex -- como a
    API monta a matriz de uma requisicao.

    Antes do P-043 este teste falharia em toda linha cuja categoria nao fosse a base."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    X_treino = _matriz(df, True, treinadas)

    for i in range(len(df)):
        X_inf = _matriz(df.iloc[[i]], False, treinadas)
        assert X_inf.iloc[0].tolist() == X_treino.iloc[i].tolist(), (
            f"linha {i} difere entre treino e inferencia. "
            f"categoricas da linha: "
            f"{ {c: df.iloc[i][c] for c in CATEGORICAL_COLS} }"
        )


def test_o_bloco_one_hot_tem_o_mesmo_dtype_nas_duas_matrizes(treinadas):
    """A licao do fill_value=False (P-043, 2026-08-30).

    Com fill_value=0 os VALORES batiam (0 == False) e o dtype nao (int64 vs bool). Igual em
    valor e diferente em tipo e uma matriz diferente -- e foi assim que a primeira
    tentativa da correcao passou nos olhos e falhou no teste."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    onehot = _onehot(treinadas)
    X_treino = _matriz(df, True, treinadas)

    for i in range(len(df)):
        X_inf = _matriz(df.iloc[[i]], False, treinadas)
        assert list(X_inf[onehot].dtypes) == list(X_treino[onehot].dtypes), f"linha {i}"


def test_score_de_cada_linha_sozinha_bate_com_o_score_do_lote_inteiro(treinadas):
    """A mesma paridade, um nivel acima: pelo score, nao pela matriz.

    Passa por score_frame, que e o caminho que a API e o monitor de drift usam de verdade.
    Cobre o caso em que a matriz bate e alguma outra coisa no caminho nao."""
    df = _frame_cobrindo_o_vocabulario(treinadas)
    em_lote = score_frame(df)["probability_default"].tolist()

    for i in range(len(df)):
        sozinho = score_frame(df.iloc[[i]])["probability_default"].iloc[0]
        assert sozinho == em_lote[i], f"linha {i}: {sozinho} sozinha vs {em_lote[i]} em lote"


# ------------------------------------- achados nomeados, mantidos vivos como assert

def test_application_type_continua_fora_do_contrato(treinadas):
    """Guarda a DECISAO do P-045. Ate 2026-08-31 guardava o DEFEITO.

    A versao anterior afirmava que application_type tinha ZERO colunas treinadas --
    documentava uma feature inerte que o contrato da API apresentava como viva. Ela
    previa UMA forma de ficar obsoleta ("se um retreino incluir mais de uma categoria"),
    isto e, o achado ser INVALIDADO. Nao previa a forma que de fato aconteceu: o achado
    ser RESOLVIDO. A feature foi removida do FEATURE_SET, do CATEGORICAL_COLS e do
    ScoreRequest, e o `cats["application_type"]` passou a levantar KeyError -- o teste
    virou obstaculo ao proprio conserto que ele existia para motivar.

    Licao registrada em 04_/arquitetura_de_processo.md §21: teste que documenta um achado
    precisa dizer o que fazer quando o achado for CONSERTADO, nao so quando for
    desmentido. O caminho do conserto e justamente o que se esta perseguindo.

    A remocao nao foi por argumento: a ablacao deu diferenca de exatamente $0.00 com IC
    [$0, $0] -- identidade, nao estimativa, porque a feature nunca produziu uma coluna.
    Agora o teste guarda a saida: quem re-adicionar sem ler o motivo quebra aqui."""
    from src.data import FEATURE_SET

    assert "application_type" not in FEATURE_SET, (
        "application_type voltou ao FEATURE_SET. Era constante no treino (zero colunas "
        "one-hot) e foi removida em 2026-08-31 apos ablacao com delta $0.00. "
        "Ver MODEL_CARD §9 e CHANGELOG 3.1.0 antes de reverter."
    )
    assert "application_type" not in CATEGORICAL_COLS
    assert not [c for c in treinadas if c.startswith("application_type_")], (
        "o artefato treinado tem coluna de application_type -- o modelo foi retreinado "
        "com a feature de volta, e a remocao precisa ser reavaliada"
    )


def test_categoria_desconhecida_e_indistinguivel_da_base_no_artefato(treinadas):
    """Documenta o P-044, agora em escala real e nao em frame de duas colunas.

    Uma categoria que o treino nunca viu produz coluna fora da lista treinada, que o
    reindex descarta -- deixando o grupo em zero. A categoria-BASE produz exatamente o
    mesmo zero. Nao ha como separar as duas a partir do .joblib, e foi por isso que o aviso
    escrito em 30/08 disparava em toda requisicao.

    Enquanto o P-044 nao congelar o vocabulario em disco, este teste tem que passar."""
    onehot = _onehot(treinadas)
    coluna = next(c for c, v in _categorias_por_coluna(treinadas).items() if v)

    base = dict(BASE); base[coluna] = PLACEHOLDER
    desconhecida = dict(BASE); desconhecida[coluna] = "categoria_que_nunca_existiu"

    X_base = _matriz(pd.DataFrame([base]), False, treinadas)
    X_desc = _matriz(pd.DataFrame([desconhecida]), False, treinadas)

    grupo = [c for c in onehot if c.startswith(coluna + "_")]
    assert X_base[grupo].sum(axis=1).iloc[0] == 0
    assert X_desc[grupo].sum(axis=1).iloc[0] == 0
    assert X_base.iloc[0].tolist() == X_desc.iloc[0].tolist()
