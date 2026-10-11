"""Ferramentas de diagnóstico de causa raiz para uma métrica que piorou entre dois períodos.

Duas perguntas guiam o diagnóstico:
1. A piora veio de **mudança de composição** (mais pedidos para lugares que sempre atrasam mais)
   ou de **piora dentro dos grupos** (os mesmos lugares atrasando mais)?
2. Qual **etapa** do processo explica a piora? Simula-se o período ruim com uma etapa de cada vez
   devolvida ao comportamento normal e mede-se quanto da piora desaparece.
"""
import numpy as np
import pandas as pd


def decompor_mix_taxa(base: pd.DataFrame, pico: pd.DataFrame, grupo: str, metrica: str) -> pd.DataFrame:
    """Decomposição da variação de uma taxa média em efeito taxa e efeito composição (shift-share).

    variação = Σ peso_pico · (taxa_pico − taxa_base)   [efeito taxa: os grupos pioraram]
             + Σ (peso_pico − peso_base) · taxa_base  [efeito composição: mudou o peso dos grupos]
    """
    def resumo(df):
        g = df.groupby(grupo)[metrica].agg(["mean", "size"])
        g["peso"] = g["size"] / g["size"].sum()
        return g

    b, p = resumo(base), resumo(pico)
    t = b[["mean", "peso"]].join(p[["mean", "peso"]], lsuffix="_base", rsuffix="_pico", how="outer").fillna(0)
    t["efeito_taxa"] = t["peso_pico"] * (t["mean_pico"] - t["mean_base"])
    t["efeito_composicao"] = (t["peso_pico"] - t["peso_base"]) * t["mean_base"]
    t["contribuicao"] = t["efeito_taxa"] + t["efeito_composicao"]
    return t.sort_values("contribuicao", ascending=False)


def atraso_simulado(pedidos: pd.DataFrame, fator_postagem=None, fator_transporte=None, fator_prazo=None) -> float:
    """Taxa de atraso se cada etapa fosse multiplicada por um fator (por pedido, ex.: base/pico do seu grupo).

    Etapas: compra → postagem (vendedor despacha) → entrega (transporte). O prazo prometido é contado
    da data da compra. Fatores ausentes deixam a etapa como foi observada.
    """
    um = pd.Series(1.0, index=pedidos.index)
    fp = um if fator_postagem is None else fator_postagem
    ft = um if fator_transporte is None else fator_transporte
    fz = um if fator_prazo is None else fator_prazo
    dias_post = (pedidos["postagem"] - pedidos["compra"].dt.normalize()).dt.total_seconds() / 86400
    dias_transp = (pedidos["entrega"] - pedidos["postagem"]).dt.total_seconds() / 86400
    dias_prazo = (pedidos["prometido"] - pedidos["compra"].dt.normalize()).dt.days + 1  # vale até o fim do dia
    # Atrasado = entregue a partir do dia seguinte ao prometido (mesma regra da coluna observada)
    return float(((dias_post * fp + dias_transp * ft) >= dias_prazo * fz).mean())


def fator_por_grupo(base: pd.DataFrame, pico: pd.DataFrame, grupo: str, coluna: str) -> pd.Series:
    """Razão mediana_base / mediana_pico de uma duração, por grupo, alinhada aos pedidos do pico."""
    razao = base.groupby(grupo)[coluna].median() / pico.groupby(grupo)[coluna].median()
    return pico[grupo].map(razao).fillna(1.0)
