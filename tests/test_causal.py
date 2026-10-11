"""Testes das funções de inferência causal. Rodam com `pytest -q tests` a partir da raiz."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal.ab import (correcao_bh, correcao_holm, cuped, poder_simulado, simular_espiadas,  # noqa: E402
                       tamanho_amostra_proporcoes, taxa_falso_positivo_espiando, verificar_srm)
from causal.causa_raiz import decompor_mix_taxa  # noqa: E402
from causal.controle_sintetico import controle_sintetico, pesos_otimos  # noqa: E402
from causal.its import contrafactual, desenho_segmentado, mqo_newey_west  # noqa: E402


# ---------------------------------------------------------------- A/B

def test_tamanho_amostra_entrega_o_poder_prometido():
    n = tamanho_amostra_proporcoes(0.03, 0.10)
    assert 0.77 < poder_simulado(0.03, 0.10, n, simulacoes=10_000) < 0.83


def test_teste_aa_sem_espiar_tem_erro_de_5_porcento():
    z = simular_espiadas(0.05, 1_000, 10, simulacoes=8_000, semente=3)
    assert 0.04 < taxa_falso_positivo_espiando(z, [10]) < 0.06


def test_espiar_infla_o_falso_positivo():
    z = simular_espiadas(0.05, 1_000, 20, simulacoes=4_000, semente=4)
    assert taxa_falso_positivo_espiando(z, range(1, 21)) > 2 * taxa_falso_positivo_espiando(z, [20])


def test_correcoes_conhecidas():
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(correcao_holm(p), [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(correcao_bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_cuped_mantem_a_media_e_reduz_a_variancia():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 20_000)
    y = 0.8 * x + rng.normal(0, 1, 20_000)
    ajustado = cuped(y, x)
    assert abs(ajustado.mean() - y.mean()) < 1e-9
    assert ajustado.var() < 0.3 * y.var()


def test_srm_detecta_divisao_errada():
    assert verificar_srm(50_000, 49_900) > 0.05
    assert verificar_srm(50_000, 48_800) < 0.001


# ---------------------------------------------------------------- controle sintético

def test_pesos_recuperam_combinacao_conhecida():
    rng = np.random.default_rng(1)
    Y = rng.normal(100, 10, (30, 5))
    w_real = np.array([0.5, 0.3, 0.2, 0, 0])
    w = pesos_otimos(Y @ w_real, Y)
    assert np.allclose(w, w_real, atol=1e-3) and np.isclose(w.sum(), 1)


def test_controle_sintetico_acha_efeito_plantado():
    rng = np.random.default_rng(2)
    anos = np.arange(1980, 2000)
    doadores = pd.DataFrame(rng.normal(50, 5, (20, 6)).cumsum(axis=0) / 5 + 50, index=anos)
    tratado = doadores[[0, 1]].mean(axis=1)
    tratado[anos >= 1990] -= 10  # efeito real: −10
    painel = doadores.assign(t=tratado)
    r = controle_sintetico(painel, "t", 1990)
    assert abs(r.efeito.loc[1990:].mean() + 10) < 0.5
    assert r.rmspe(False) < 0.1


# ---------------------------------------------------------------- ITS e causa raiz

def test_its_recupera_salto_conhecido():
    datas = pd.bdate_range("2018-01-01", periods=120)
    rng = np.random.default_rng(5)
    y = 10 + 0.01 * np.arange(120) + rng.normal(0, 0.3, 120)
    y[60:70] += 3  # choque de 3 unidades durante 10 dias
    X = desenho_segmentado(datas, datas[60], datas[69])
    ajuste = mqo_newey_west(X, pd.Series(y, index=datas))
    lo, hi = ajuste.intervalo("exposicao")
    assert lo < 3 < hi
    assert abs(contrafactual(X, ajuste)[datas[65]] - (10 + 0.01 * 65)) < 0.5


def test_decomposicao_soma_a_variacao_total():
    base = pd.DataFrame({"g": ["a"] * 80 + ["b"] * 20, "m": [0] * 76 + [1] * 4 + [0] * 18 + [1] * 2})
    pico = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "m": [0] * 45 + [1] * 5 + [0] * 40 + [1] * 10})
    t = decompor_mix_taxa(base, pico, "g", "m")
    assert np.isclose(t.contribuicao.sum(), pico.m.mean() - base.m.mean())
