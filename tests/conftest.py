import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _project_root_cwd(monkeypatch):
    """Os módulos abrem assets/ por caminho relativo à raiz do projeto."""
    monkeypatch.chdir(ROOT)
