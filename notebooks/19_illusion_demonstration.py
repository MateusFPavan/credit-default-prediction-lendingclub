# ============================================================================
# Fase 2a - Bloco 11: demonstracao da ilusao (tese critica)
# ============================================================================
# Tese: reject inference NAO e honestamente validavel no Lending Club. Nao e
# "parcelling ganhou/perdeu" -- e DEMONSTRAR, com o proprio dado, por que qualquer
# "melhora" reportada e provavelmente artefato de avaliacao. Tres razoes medidas:
#
#   1. Sinal compartilhado fraco: thin model (features comuns aos dois lados) tem AUC
#      out-of-sample 0,5620 +/- 0,0027 (Bloco 10, 4-fold CV) -- mal acima de 0,5. Qualquer
#      label inferido a partir desse score e quase ruido.
#   2. Sem outcome de recusado: confirmado no repo (auditoria pre-comparacao -- nenhuma
#      coluna de label de recusado em lugar nenhum). Kickout/AUK (metrica padrao de RI)
#      e impossivel -- exige recusados rotulados (ver roadmap, achado registrado).
#   3. Sem amostra nao-enviesada nem taxa populacional real: a metrica que escapa da
#      ilusao (distorcao da taxa de treino vs populacional, Illusion of Improvement,
#      arXiv 2606.18479) exige a taxa populacional VERDADEIRA como referencia. O LC nao a
#      tem (recusados nunca viraram emprestimo). Kozodoi (arXiv 1909.06108) consegue
#      escapar porque tem uma amostra nao-enviesada de 1.967 casos (recusados aceitos de
#      proposito, pra observar outcome); nao temos analogo aqui. Conclusao do Illusion:
#      avaliar RI nos aprovados PRODUZ ilusao de melhora -- extrapolacao supera ate um
#      Oraculo com labels reais, porque avaliar no pool aceito recompensa fronteiras
#      extremas. Qualquer ganho reportado nos aprovados e provavel artefato, nao melhora
#      real.
#
# O que este codigo DEMONSTRA: nao mede "quem ganhou" (parcelling vs baseline). Mede o
# SINTOMA do artefato que o Illusion identifica -- a taxa de default do conjunto de
# treino aumentado (aprovados reais + recusados com label inferido) inflada em relacao a
# taxa real dos aprovados, e como essa inflacao CRESCE com o base_mult (quanto mais
# "agressivo" o RI, mais inflada a taxa, mais forte o sintoma). Sem a taxa populacional
# verdadeira, essa inflacao NAO e interpretavel como correcao de vies -- pode ser
# exatamente isso, ou pode ser um vies novo sendo criado. E por isso que RI nao e
# validavel aqui, nao porque o parcelling "nao funciona": o parcelling foi implementado e
# roda (competencia tecnica, Blocos 1-10), mas a avaliacao honesta dele e impossivel com
# este dado (a contribuicao intelectual deste bloco).
#
# Referencias: Illusion of Improvement (Scarone & Baeza-Yates, arXiv 2606.18479, ECML
# PKDD 2026); Kozodoi (arXiv 1909.06108, 2407.13009); Hugo Lopes 2018.

import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# notebooks/17_reject_thin_model.py carregado como modulo (nome comeca com digito, nao
# importavel via `import`) -- mesmo padrao ja usado em notebooks/18_profit_evaluation.py.
_NB17_PATH = Path(__file__).resolve().parent / "17_reject_thin_model.py"
_spec = importlib.util.spec_from_file_location("nb17_reject_thin_model", _NB17_PATH)
nb17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb17)


def demonstrate_illusion(X_appr, y_appr, X_rej, bands_list=(10,),
                          base_list=(1.0, 1.5, 2.0, 2.5, 3.0)):
    """
    Demonstra os sintomas da ilusao de melhora (Illusion of Improvement, 2026) no LC.
    Metrica-chave: taxa de default do conjunto de TREINO apos RI vs taxa dos aprovados.
    Se o parcelling INFLA a taxa de treino acima da dos aprovados, e' o sintoma do
    artefato -- e essa inflacao crescendo com base_mult e a evidencia de que "RI mais
    agressivo" so aprofunda o artefato, nao uma melhora real.
    """
    base_rate = float(y_appr.mean())
    print(f"[REFERENCIA] taxa de default dos APROVADOS (unico observavel): {base_rate:.4f}")
    print("[LIMITACAO] taxa POPULACIONAL real: DESCONHECIDA (recusados sem outcome).")
    print("            Sem ela, nao ha como dizer se a inflacao corrige ou distorce.\n")

    thin = nb17.baseline_ignore_rejects(X_appr, y_appr)

    print(f"{'n_bands':>8}{'base_mult':>10}{'taxa_treino_pos_RI':>20}{'inflacao_vs_aprovados':>22}")
    results = []
    for nb in bands_list:
        for bm in base_list:
            inferred, _ = nb17.parcelling_labels(thin, X_appr, y_appr, X_rej, nb, bm)
            n_appr, n_rej = len(y_appr), len(inferred)
            train_rate = (y_appr.sum() + inferred.sum()) / (n_appr + n_rej)
            inflacao = train_rate - base_rate
            results.append((nb, bm, train_rate, inflacao))
            print(f"{nb:>8}{bm:>10}{train_rate:>20.4f}{inflacao:>+22.4f}")

    print("\n[LEITURA] Se a taxa de treino SOBE com base_mult (inflacao positiva "
          "crescente), e' o sintoma do artefato do Illusion 2026: RI mais agressivo "
          "infla a taxa de treino, cria fronteira mais extrema, e PARECE melhorar nos "
          "aprovados sem melhora real. Sem a taxa populacional verdadeira, essa inflacao "
          "NAO e interpretavel como correcao.")
    return results


if __name__ == "__main__":
    print("[Fase 2a - Bloco 11] demonstracao da ilusao de melhora (tese critica)\n")

    print("Carregando aprovados (split 'train')...")
    X_appr, y_appr, issue_d_appr = nb17.load_approved_shared()
    print(f"  {len(X_appr):,} linhas | bad rate {y_appr.mean()*100:.2f}%")

    print("\nCarregando recusados (Parquet particionado, 27.648.741 linhas esperadas)...")
    X_rej, rej_flags, rej_counts = nb17.load_rejected_shared()
    print(f"  {len(X_rej):,} linhas")

    print("\n=== TRES PERNAS DA TESE (registro) ===")
    print("  1. Sinal compartilhado fraco: thin model AUC out-of-sample = 0.5620 +/- 0.0027 "
          "(Bloco 10, 4-fold CV) -- mal acima de 0.5.")
    print("  2. Sem outcome de recusado: confirmado no repo (auditoria pre-comparacao) -- "
          "Kickout/AUK impossivel.")
    print("  3. Sem amostra nao-enviesada nem taxa populacional real: sem referencia pra "
          "validar se o RI corrige vies ou cria um novo (Illusion of Improvement, 2026).")

    print("\n=== DEMONSTRACAO: inflacao da taxa de treino vs base_mult ===")
    demonstrate_illusion(X_appr, y_appr, X_rej)

    print("\n[Fase 2a - Bloco 11] concluido.")
