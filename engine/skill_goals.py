"""
Tabela de XP do RuneScape 3, meta de nível por habilidade e persistência do progresso entre sessões.

Nada aqui lê o jogo: o XP chega por leituras (o tooltip da aba Skills) ou digitado pelo usuário, e este módulo
só calcula nível, ritmo e distância até a meta a partir delas.
"""
import json
import math
import os
from typing import Optional, Tuple

MAX_LEVEL = 120

PROGRESS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "progress.json")

def xp_for_level(level: int) -> int:
    """XP total necessário para ALCANÇAR `level` (nível 1 = 0 XP; nível 99 = 13.034.431)."""
    level = min(max(int(level), 1), MAX_LEVEL)
    total = sum(math.floor(lv + 300 * 2 ** (lv / 7)) for lv in range(1, level))
    return total // 4


def level_for_xp(xp: float) -> int:
    """Nível correspondente ao XP total (limitado a MAX_LEVEL)."""
    level = 1
    while level < MAX_LEVEL and xp >= xp_for_level(level + 1):
        level += 1
    return level


class SkillGoal:
    """
    Acompanha o XP de UMA habilidade a partir de leituras (xp, instante) e a distância até o nível-meta.
    O ritmo (XP/hora) vem da primeira à última leitura; entre leituras o XP é só extrapolado (estimated_xp).
    """

    MIN_SPAN_S = 60.0  # intervalo mínimo entre a primeira e a última leitura para falar em XP/hora

    def __init__(self, skill: str, target_level: Optional[int] = None):
        self.skill = skill
        self.target_level = target_level
        self.first: Optional[Tuple[float, float]] = None  # (xp, instante) da leitura-base
        self.last: Optional[Tuple[float, float]] = None   # (xp, instante) da leitura mais recente
        self.readings = 0
        self.manual_seed = False  # a base foi digitada: a primeira leitura do jogo a substitui

    def observe(self, xp: float, now: float, manual: bool = False) -> None:
        """
        Registra o XP total (lido do jogo, ou digitado se manual=True). Uma leitura manual, ou a primeira
        leitura do jogo depois de uma manual, recomeça a medição do ritmo a partir dela.
        """
        xp = max(float(xp), 0.0)
        if self.last is None or manual or self.manual_seed:
            self.first = (xp, now)
        self.last = (xp, now)
        self.manual_seed = manual
        self.readings += 1

    @property
    def has_reading(self) -> bool:
        return self.last is not None

    @property
    def xp(self) -> Optional[float]:
        return self.last[0] if self.last else None

    @property
    def level(self) -> Optional[int]:
        return level_for_xp(self.last[0]) if self.last else None

    @property
    def start_level(self) -> Optional[int]:
        return level_for_xp(self.first[0]) if self.first else None

    @property
    def gained(self) -> float:
        return self.last[0] - self.first[0] if self.last and self.first else 0.0

    @property
    def target_xp(self) -> Optional[float]:
        return None if self.target_level is None else float(xp_for_level(self.target_level))

    @property
    def xp_to_goal(self) -> Optional[float]:
        target = self.target_xp
        return None if target is None or self.last is None else max(target - self.last[0], 0.0)

    @property
    def reached(self) -> bool:
        target = self.target_xp
        return target is not None and self.last is not None and self.last[0] >= target

    @property
    def progress(self) -> Optional[float]:
        """Fração (0-1) do caminho entre a leitura-base e a meta."""
        target = self.target_xp
        if target is None or self.first is None or self.last is None:
            return None
        if target <= self.first[0]:
            return 1.0
        return min(max((self.last[0] - self.first[0]) / (target - self.first[0]), 0.0), 1.0)

    def xp_per_hour(self) -> Optional[float]:
        """Ritmo da primeira à última leitura; None enquanto não há duas leituras separadas o bastante."""
        if not self.first or not self.last:
            return None
        span = self.last[1] - self.first[1]
        if span < self.MIN_SPAN_S or self.last[0] <= self.first[0]:
            return None
        return (self.last[0] - self.first[0]) / span * 3600

    def estimated_xp(self, now: float) -> Optional[float]:
        """XP agora: a última leitura mais o ritmo desde então (só uma estimativa para decidir quando reler)."""
        if not self.last:
            return None
        rate = self.xp_per_hour()
        return self.last[0] + (rate * (now - self.last[1]) / 3600 if rate else 0.0)

    def eta_seconds(self, now: float) -> Optional[float]:
        """Tempo estimado até a meta no ritmo atual; None sem meta ou sem ritmo medido."""
        rate, target = self.xp_per_hour(), self.target_xp
        if rate is None or target is None:
            return None
        return max(target - self.estimated_xp(now), 0.0) / rate * 3600


def _parse_number(text: str, label: str, decimal: bool = False) -> Optional[float]:
    """Número digitado na tela; vazio -> None. Inteiros aceitam separador de milhar (1.234.567)."""
    text = (text or "").strip()
    if not text:
        return None
    cleaned = text.replace(" ", "").replace(",", ".") if decimal else text.replace(" ", "").replace(".", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"{label}: '{text}' não é um número.") from None
    if value < 0 or math.isnan(value) or math.isinf(value):
        raise ValueError(f"{label} não pode ser negativo.")
    return value


def parse_int(text: str, label: str) -> Optional[int]:
    value = _parse_number(text, label)
    return None if value is None else int(value)


def parse_decimal(text: str, label: str) -> Optional[float]:
    return _parse_number(text, label, decimal=True)


class ProgressStore:
    """
    Guarda em config/progress.json o último XP lido de cada habilidade e a meta escolhida (para preencher a
    tela na próxima sessão). Lê o arquivo a cada chamada.
    """

    def __init__(self, path: str = PROGRESS_PATH):
        self.path = path

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def skill(self, skill: str) -> dict:
        """{'xp': float, 'target_level': int|None} da habilidade; {} se nunca foi salva."""
        entry = self._load().get("skills", {}).get(skill)
        return entry if isinstance(entry, dict) else {}

    def save_skill(self, skill: str, xp: Optional[float], target_level: Optional[int]) -> None:
        """Grava o último XP lido e a meta. xp None mantém o XP já salvo (sessão sem nenhuma leitura)."""
        data = self._load()
        skills = data.setdefault("skills", {})
        if xp is None:
            xp = skills.get(skill, {}).get("xp")
        skills[skill] = {"xp": None if xp is None else round(float(xp), 1), "target_level": target_level}
        self._save(data)
