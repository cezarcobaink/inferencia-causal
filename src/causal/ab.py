"""Planejamento e análise de testes A/B.

Funções pequenas e testáveis para as perguntas que aparecem antes, durante e depois de um experimento:
quantas pessoas são necessárias, o que acontece se o resultado for espiado antes da hora, como corrigir
várias comparações e como reduzir a variância com dados do período anterior (CUPED).
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- antes do teste

def tamanho_amostra_proporcoes(p_base, efeito_relativo, alfa=0.05, poder=0.80):
    """Pessoas por grupo para detectar uma variação relativa numa taxa (teste bicaudal de duas proporções).

    Exemplo: p_base = 0,03 e efeito_relativo = 0,10 → detectar 3,0% → 3,3%.
    """
    p1 = p_base
    p2 = p_base * (1 + efeito_relativo)
    z_a = stats.norm.ppf(1 - alfa / 2)
    z_b = stats.norm.ppf(poder)
    p_med = (p1 + p2) / 2
    numerador = (z_a * np.sqrt(2 * p_med * (1 - p_med)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(numerador / (p2 - p1) ** 2))


def teste_proporcoes(conv_a, n_a, conv_b, n_b):
    """Teste z de duas proporções (variância agrupada). Devolve a estatística z e o p-valor bicaudal."""
    conv_a, n_a, conv_b, n_b = map(np.asarray, (conv_a, n_a, conv_b, n_b))
    p = (conv_a + conv_b) / (n_a + n_b)
    erro = np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    z = (conv_b / n_b - conv_a / n_a) / erro
    return z, 2 * stats.norm.sf(np.abs(z))


def poder_simulado(p_base, efeito_relativo, n_por_grupo, alfa=0.05, simulacoes=20_000, semente=0):
    """Poder estimado por simulação: em quantos experimentos o efeito real seria detectado."""
    rng = np.random.default_rng(semente)
    a = rng.binomial(n_por_grupo, p_base, simulacoes)
    b = rng.binomial(n_por_grupo, p_base * (1 + efeito_relativo), simulacoes)
    _, p = teste_proporcoes(a, n_por_grupo, b, n_por_grupo)
    return float((p < alfa).mean())


def verificar_srm(n_a, n_b, proporcao_a=0.5):
    """Sample Ratio Mismatch: a divisão observada bate com a planejada? p-valor do qui-quadrado.

    Um p-valor muito baixo (ex.: < 0,001) indica problema na atribuição ou no registro dos dados,
    e o resultado do teste não deve ser lido antes de achar a causa.
    """
    total = n_a + n_b
    esperado = [total * proporcao_a, total * (1 - proporcao_a)]
    return float(stats.chisquare([n_a, n_b], esperado).pvalue)


# ---------------------------------------------------------------- durante o teste

def simular_espiadas(p_base, n_por_dia, dias, simulacoes=5_000, efeito_relativo=0.0, semente=0):
    """Simula experimentos acompanhados dia a dia. Devolve a matriz de estatísticas z (simulações × dias).

    Com efeito_relativo = 0 (teste A/A), toda "vitória" é um falso positivo.
    """
    rng = np.random.default_rng(semente)
    a = rng.binomial(n_por_dia, p_base, (simulacoes, dias)).cumsum(axis=1)
    b = rng.binomial(n_por_dia, p_base * (1 + efeito_relativo), (simulacoes, dias)).cumsum(axis=1)
    n = n_por_dia * np.arange(1, dias + 1)
    z, _ = teste_proporcoes(a, n, b, n)
    return z


def taxa_falso_positivo_espiando(z, dias_olhados, limite=None, alfa=0.05):
    """Fração de testes A/A declarados vencedores se o analista parar no primeiro dia olhado com |z| acima do limite."""
    limite = stats.norm.ppf(1 - alfa / 2) if limite is None else limite
    olhados = np.abs(z[:, np.asarray(dias_olhados) - 1])
    limites = np.broadcast_to(limite, olhados.shape[1:])
    return float((olhados > limites).any(axis=1).mean())


def fronteira_obrien_fleming(z_aa, dias_olhados, alfa=0.05):
    """Limites de O'Brien-Fleming calibrados por simulação A/A.

    O formato é limite_k = c · sqrt(K / k): muito exigente no começo e perto de 1,96 no fim.
    A constante c é ajustada para que a taxa global de falso positivo fique em `alfa`.
    """
    k = np.arange(1, len(dias_olhados) + 1)
    forma = np.sqrt(len(k) / k)
    baixo, alto = 1.5, 4.0
    for _ in range(50):  # bisseção em c
        c = (baixo + alto) / 2
        if taxa_falso_positivo_espiando(z_aa, dias_olhados, c * forma) > alfa:
            baixo = c
        else:
            alto = c
    return alto * forma


# ---------------------------------------------------------------- depois do teste

def correcao_holm(p):
    """p-valores ajustados por Holm (controla a chance de ao menos um falso positivo)."""
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    ajustado = np.empty(m)
    ajustado[ordem] = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[ordem]))
    return ajustado


def correcao_bh(p):
    """p-valores ajustados por Benjamini-Hochberg (controla a fração de falsos positivos entre os achados)."""
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    bruto = p[ordem] * m / np.arange(1, m + 1)
    ajustado = np.empty(m)
    ajustado[ordem] = np.minimum(1, np.minimum.accumulate(bruto[::-1])[::-1])
    return ajustado


def cuped(y, x_pre):
    """CUPED: y - θ·(x_pre - média), com θ = cov(y, x_pre) / var(x_pre), estimado nos dois grupos juntos.

    Remove de y a parte previsível pelo comportamento anterior ao teste. A média não muda (o efeito
    estimado continua sem viés), mas a variância cai na proporção de 1 - ρ².
    """
    y, x_pre = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean())


def diferenca_medias(y_a, y_b, alfa=0.05):
    """Diferença de médias (B − A) com intervalo de confiança de Welch."""
    y_a, y_b = np.asarray(y_a, float), np.asarray(y_b, float)
    dif = y_b.mean() - y_a.mean()
    erro = np.sqrt(y_a.var(ddof=1) / len(y_a) + y_b.var(ddof=1) / len(y_b))
    z = stats.norm.ppf(1 - alfa / 2)
    return dif, dif - z * erro, dif + z * erro
