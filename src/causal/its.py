"""Série temporal interrompida (ITS) por regressão segmentada, com erros-padrão robustos (Newey-West).

Modelo:  y_t = a + b·t + c·exposicao_t + d·depois_t + e·depois_t·(t − t0) + sazonalidade + erro

- `b` é a tendência que já existia;
- `c` é o desvio médio durante a janela de exposição (o efeito imediato);
- `d` e `e` medem se, depois da janela, o nível ou a tendência ficaram diferentes.

Séries diárias têm erros correlacionados no tempo, e o erro-padrão comum de mínimos quadrados fica
pequeno demais. Newey-West corrige isso.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Ajuste:
    coef: pd.Series
    erro_padrao: pd.Series
    residuos: np.ndarray

    def intervalo(self, nome, alfa=0.05):
        z = stats.norm.ppf(1 - alfa / 2)
        return self.coef[nome] - z * self.erro_padrao[nome], self.coef[nome] + z * self.erro_padrao[nome]

    def p_valor(self, nome):
        return float(2 * stats.norm.sf(abs(self.coef[nome] / self.erro_padrao[nome])))


def mqo_newey_west(X: pd.DataFrame, y: pd.Series, defasagens: int = 5, pesos=None) -> Ajuste:
    """Mínimos quadrados (ponderados, se `pesos`) com matriz de covariância de Newey-West."""
    Xm = X.to_numpy(float)
    ym = y.to_numpy(float)
    w = np.ones(len(ym)) if pesos is None else np.asarray(pesos, float)
    raiz = np.sqrt(w)
    Xw, yw = Xm * raiz[:, None], ym * raiz
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    pao = np.linalg.pinv(Xw.T @ Xw)
    u = Xw * resid[:, None]
    S = u.T @ u
    for lag in range(1, defasagens + 1):
        peso = 1 - lag / (defasagens + 1)  # núcleo de Bartlett
        G = u[lag:].T @ u[:-lag]
        S += peso * (G + G.T)
    cov = pao @ S @ pao
    return Ajuste(pd.Series(beta, index=X.columns), pd.Series(np.sqrt(np.diag(cov)), index=X.columns), ym - Xm @ beta)


def desenho_segmentado(datas: pd.DatetimeIndex, inicio, fim, dias_semana=True) -> pd.DataFrame:
    """Matriz de regressão da ITS: tendência, janela de exposição [inicio, fim], período depois e dia da semana."""
    datas = pd.DatetimeIndex(datas)
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    t = (datas - datas[0]).days.to_numpy(float)
    t0 = (fim - datas[0]).days
    X = pd.DataFrame({"constante": 1.0, "tendencia": t,
                      "exposicao": ((datas >= inicio) & (datas <= fim)).astype(float),
                      "depois": (datas > fim).astype(float)}, index=datas)
    X["depois_tendencia"] = X["depois"] * np.maximum(t - t0, 0)
    if dias_semana:
        for d in sorted(set(datas.dayofweek))[1:]:  # o primeiro dia fica como referência
            X[f"dia_{d}"] = (datas.dayofweek == d).astype(float)
    return X


def contrafactual(X: pd.DataFrame, ajuste: Ajuste) -> pd.Series:
    """Previsão do modelo com os termos da intervenção zerados: o que se esperaria sem o evento."""
    X0 = X.copy()
    X0[["exposicao", "depois", "depois_tendencia"]] = 0.0
    return pd.Series(X0.to_numpy(float) @ ajuste.coef.to_numpy(), index=X.index)


def estimar_evento(serie: pd.Series, inicio, fim, pre_inicio, pos_fim, tendencia=True, pesos=None,
                   defasagens=5):
    """Ajusta a ITS numa janela [pre_inicio, pos_fim] em torno do evento [inicio, fim].

    Devolve (ajuste, matriz X, série usada, pesos usados). Sem `tendencia`, o contrafactual é o nível
    médio do período anterior ajustado por dia da semana.
    """
    janela = (serie.index >= pd.Timestamp(pre_inicio)) & (serie.index <= pd.Timestamp(pos_fim))
    y = serie[janela]
    w = None if pesos is None else pesos[janela]
    X = desenho_segmentado(y.index, inicio, fim)
    if not tendencia:
        X = X.drop(columns=["tendencia", "depois_tendencia"])
        X["tendencia"] = 0.0
        X["depois_tendencia"] = 0.0
    colunas_validas = [c for c in X.columns if X[c].abs().sum() > 0]
    ajuste = mqo_newey_west(X[colunas_validas], y, defasagens, pesos=w)
    ajuste.coef = ajuste.coef.reindex(X.columns, fill_value=0.0)
    ajuste.erro_padrao = ajuste.erro_padrao.reindex(X.columns, fill_value=0.0)
    return ajuste, X, y, w
