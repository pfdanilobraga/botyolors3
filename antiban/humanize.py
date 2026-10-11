"""
Aleatoriedade "humana" central do bot: nenhuma ação pode ter tempo fixo nem trajetória 100% previsível.

Tudo que o bot espera, escolhe ou move passa por aqui. Os tempos seguem distribuições assimétricas (a maioria perto
da mediana, algumas respostas bem mais lentas, como uma pessoa de verdade), os pontos de clique caem dentro do alvo
(não sempre no centro) e a "velocidade" geral varia de uma sessão para outra.

O gerador é um `random.Random` próprio (`rng`) para os testes poderem fixar a semente ou trocá-lo por um gerador
neutro (veja `tests/conftest.py`).
"""
import math
import random
from typing import Optional, Sequence, Tuple

rng = random.Random()

# "Ritmo" da sessão: multiplica todas as esperas (0,88 = mais rápido que a mediana, 1,2 = mais devagar)
_pace = 1.0
PACE_RANGE = (0.88, 1.22)
DISTRACTION_CHANCE = 0.015        # chance, a cada alvo clicado, de uma pausa longa (a pessoa se distraiu)
DISTRACTION_SECONDS = (3.0, 11.0)


def seed(value) -> None:
    rng.seed(value)


def new_session() -> float:
    """Sorteia o ritmo desta sessão (chamar no Start). Retorna o fator sorteado."""
    global _pace
    _pace = rng.uniform(*PACE_RANGE)
    return _pace


def pace() -> float:
    return _pace


def uniform(low: float, high: float) -> float:
    return rng.uniform(low, high)


def randint(low: int, high: int) -> int:
    return rng.randint(low, high)


def choice(options: Sequence):
    return rng.choice(options)


def chance(probability: float) -> bool:
    return rng.random() < probability


def jitter(value: float, spread: float = 0.25) -> float:
    """`value` variado em ±spread (uniforme): jitter(8, 0.3) fica entre 5,6 e 10,4."""
    return value * rng.uniform(1.0 - spread, 1.0 + spread)


def human_delay(median: float, sigma: float = 0.35, low: float = 0.45, high: float = 2.6) -> float:
    """
    Tempo de reação/espera em segundos: lognormal em torno de `median` (a maioria perto dela, às vezes bem mais
    lento), limitado a [low, high] x median e multiplicado pelo ritmo da sessão.
    """
    value = median * math.exp(rng.gauss(0.0, sigma))
    return min(max(value, median * low), median * high) * _pace


def interval(median: float, sigma: float = 0.4, low: float = 0.55, high: float = 1.8) -> float:
    """Intervalo longo entre ações periódicas (ex.: ler o XP a cada ~3 min): bem espalhado, nunca "de 3 em 3"."""
    return human_delay(median, sigma, low, high)


def distraction() -> float:
    """Quase sempre 0; raramente uma pausa longa, como quando uma pessoa se distrai por alguns segundos."""
    return rng.uniform(*DISTRACTION_SECONDS) if chance(DISTRACTION_CHANCE) else 0.0


def gauss_offset(sigma_x: float, sigma_y: Optional[float] = None, limit: float = 2.5) -> Tuple[float, float]:
    """Deslocamento (dx, dy) gaussiano, cortado em ±limit desvios (sem outliers absurdos)."""
    sigma_y = sigma_x if sigma_y is None else sigma_y
    dx = max(min(rng.gauss(0.0, sigma_x), limit * sigma_x), -limit * sigma_x)
    dy = max(min(rng.gauss(0.0, sigma_y), limit * sigma_y), -limit * sigma_y)
    return dx, dy


def point_in_box(box: Sequence[float], inner: float = 0.7, sigma: float = 0.2) -> Tuple[int, int]:
    """
    Ponto de clique dentro de uma caixa (x1, y1, x2, y2): gaussiano em torno do centro (desvio = `sigma` x o tamanho),
    sempre dentro da região central `inner` da caixa. Pessoas não clicam no centro exato toda vez.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = max(min(rng.gauss(0.0, w * sigma), w * inner / 2), -w * inner / 2)
    dy = max(min(rng.gauss(0.0, h * sigma), h * inner / 2), -h * inner / 2)
    return int(round(cx + dx)), int(round(cy + dy))


def around(point: Tuple[float, float], sigma: float, bounds: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int]:
    """Ponto gaussiano em torno de `point`, preso aos limites (xmin, ymin, xmax, ymax) se dados."""
    dx, dy = gauss_offset(sigma)
    x, y = point[0] + dx, point[1] + dy
    if bounds:
        x, y = min(max(x, bounds[0]), bounds[2]), min(max(y, bounds[1]), bounds[3])
    return int(round(x)), int(round(y))
