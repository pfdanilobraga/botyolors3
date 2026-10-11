import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _project_root_cwd(monkeypatch):
    """Os módulos abrem assets/ por caminho relativo à raiz do projeto."""
    monkeypatch.chdir(ROOT)


class NeutralRandom:
    """Gerador sem sorte: sempre o valor do meio. Para testes que precisam de tempos exatos."""

    def uniform(self, a, b):
        return (a + b) / 2

    def gauss(self, mu, sigma):
        return mu

    def random(self):
        return 0.5

    def randint(self, a, b):
        return (a + b) // 2

    def choice(self, seq):
        return seq[len(seq) // 2]

    def seed(self, value):
        pass


@pytest.fixture
def neutral(monkeypatch):
    """Troca o gerador do bot pelo neutro (tempos = mediana, pontos = centro) e fixa o ritmo da sessão em 1."""
    from antiban import humanize
    monkeypatch.setattr(humanize, "rng", NeutralRandom())
    monkeypatch.setattr(humanize, "_pace", 1.0)
    return humanize


@pytest.fixture(autouse=True)
def _seeded_humanize(monkeypatch):
    """Por padrão cada teste usa o gerador real, mas com semente fixa (resultados repetíveis) e ritmo 1."""
    from antiban import humanize
    monkeypatch.setattr(humanize, "rng", random.Random(1234))
    monkeypatch.setattr(humanize, "_pace", 1.0)
