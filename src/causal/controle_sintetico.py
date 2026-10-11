"""Controle sintético (Abadie, Diamond e Hainmueller, 2010), implementado do zero.

Ideia: quando só uma unidade recebe a intervenção (um estado, uma loja, um país), não existe grupo de controle
natural. Constrói-se um: uma média ponderada de unidades não tratadas que reproduz a trajetória da unidade
tratada ANTES da intervenção. Depois da intervenção, a distância entre a unidade real e a sintética é o efeito.

Esta versão ajusta os pesos só pela série do resultado no período anterior (sem covariáveis), a forma mais
transparente do método. Os pesos são não negativos e somam 1: o controle sintético é uma interpolação de
unidades reais, nunca uma extrapolação.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass
class Resultado:
    pesos: pd.Series          # peso de cada unidade doadora
    real: pd.Series           # série da unidade tratada
    sintetico: pd.Series      # série do controle sintético
    inicio: int               # primeiro período tratado

    @property
    def efeito(self) -> pd.Series:
        return self.real - self.sintetico

    def rmspe(self, depois: bool) -> float:
        """Raiz do erro quadrático médio antes (ajuste) ou depois (efeito) da intervenção."""
        e = self.efeito
        e = e[e.index >= self.inicio] if depois else e[e.index < self.inicio]
        return float(np.sqrt((e ** 2).mean()))

    @property
    def razao_rmspe(self) -> float:
        """Efeito depois em relação ao erro de ajuste antes: a estatística dos testes placebo."""
        return self.rmspe(True) / self.rmspe(False)


def pesos_otimos(y_tratado: np.ndarray, Y_doadores: np.ndarray, penalidade: float = 1e4) -> np.ndarray:
    """Pesos w ≥ 0, Σw = 1 que minimizam ||y_tratado − Y_doadores · w||² no período anterior.

    Resolvido como mínimos quadrados não negativos (NNLS), com a restrição Σw = 1 imposta por uma linha
    extra de peso alto. É mais estável que otimizadores genéricos, que podem parar longe do ótimo
    quando há muitos doadores parecidos.
    """
    escala = np.abs(y_tratado).mean()
    A = np.vstack([Y_doadores / escala, penalidade * np.ones(Y_doadores.shape[1])])
    b = np.r_[y_tratado / escala, penalidade]
    w, _ = nnls(A, b, maxiter=10_000)
    return w / w.sum()


def controle_sintetico(painel: pd.DataFrame, tratado, inicio: int, doadores=None, fim_ajuste=None) -> Resultado:
    """Ajusta o controle sintético.

    painel: tabela larga, linhas = períodos, colunas = unidades.
    inicio: primeiro período em que a intervenção vale.
    fim_ajuste: último período usado para ajustar os pesos (padrão: o anterior a `inicio`).
    """
    doadores = [c for c in painel.columns if c != tratado] if doadores is None else list(doadores)
    fim_ajuste = inicio - 1 if fim_ajuste is None else fim_ajuste
    pre = painel.index <= fim_ajuste
    w = pesos_otimos(painel.loc[pre, tratado].to_numpy(float), painel.loc[pre, doadores].to_numpy(float))
    sintetico = pd.Series(painel[doadores].to_numpy(float) @ w, index=painel.index)
    return Resultado(pd.Series(w, index=doadores), painel[tratado], sintetico, inicio)


def placebos_no_espaco(painel: pd.DataFrame, tratado, inicio: int) -> pd.DataFrame:
    """Finge que cada unidade não tratada recebeu a intervenção e mede o 'efeito' que aparece.

    Se o efeito da unidade realmente tratada é grande em relação a esses efeitos de mentira,
    ele dificilmente é obra do acaso. Devolve uma linha por unidade com as estatísticas.
    """
    linhas = []
    controles = [c for c in painel.columns if c != tratado]
    for unidade in [tratado] + controles:
        # O tratado real não pode servir de doador para os placebos: ele foi afetado
        doadores = [c for c in controles if c != unidade]
        r = controle_sintetico(painel, unidade, inicio, doadores)
        linhas.append({"unidade": unidade, "tratado": unidade == tratado, "rmspe_antes": r.rmspe(False),
                       "rmspe_depois": r.rmspe(True), "razao": r.razao_rmspe, "efeito": r.efeito})
    return pd.DataFrame(linhas).set_index("unidade")


def p_valor_permutacao(placebos: pd.DataFrame) -> float:
    """Fração das unidades com razão RMSPE depois/antes pelo menos tão alta quanto a do tratado."""
    razao_tratado = placebos.loc[placebos.tratado, "razao"].iloc[0]
    return float((placebos["razao"] >= razao_tratado).mean())


def diferenca_em_diferencas(painel: pd.DataFrame, tratado, inicio: int) -> float:
    """DiD clássico: (tratado depois − antes) − (média dos controles depois − antes)."""
    depois = painel.index >= inicio
    controles = painel.drop(columns=tratado).mean(axis=1)
    return float((painel.loc[depois, tratado].mean() - painel.loc[~depois, tratado].mean())
                 - (controles[depois].mean() - controles[~depois].mean()))
