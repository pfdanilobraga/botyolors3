"""
Tabela de XP do RuneScape 3, meta de nível por habilidade e persistência do progresso entre sessões.

Nada aqui lê o jogo: o XP é uma ESTIMATIVA (itens obtidos x XP por item) que o usuário pode corrigir
a qualquer momento informando o nível/XP reais.
"""
import json
import math
import os
from typing import Dict, Optional

MAX_LEVEL = 120

PROGRESS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "progress.json")

# XP base por item coletado, sem bônus de roupa/evento (o usuário edita o valor na tela)
DEFAULT_XP_PER_ITEM: Dict[str, Dict[str, float]] = {
    "Mineração": {"Copper": 17.5, "Iron": 35.0, "Mithril": 80.0},
}


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


def default_xp_per_item(skill: str, target: str) -> float:
    return DEFAULT_XP_PER_ITEM.get(skill, {}).get(target, 0.0)


class SkillGoal:
    """
    Acompanha o XP estimado de UMA habilidade durante a sessão e a distância até o nível-meta.
    O XP ganho é `itens x xp_per_item`; `sync()` corrige o total quando o usuário informa o valor real.
    """

    def __init__(self, skill: str, start_xp: float, xp_per_item: float = 0.0, target_level: Optional[int] = None):
        self.skill = skill
        self.start_xp = max(float(start_xp), 0.0)
        self.xp_per_item = max(float(xp_per_item), 0.0)
        self.target_level = target_level
        self.items = 0

    @property
    def gained(self) -> float:
        return self.items * self.xp_per_item

    @property
    def xp(self) -> float:
        return self.start_xp + self.gained

    @property
    def level(self) -> int:
        return level_for_xp(self.xp)

    @property
    def start_level(self) -> int:
        return level_for_xp(self.start_xp)

    @property
    def target_xp(self) -> Optional[float]:
        return None if self.target_level is None else float(xp_for_level(self.target_level))

    @property
    def xp_to_goal(self) -> Optional[float]:
        target = self.target_xp
        return None if target is None else max(target - self.xp, 0.0)

    @property
    def reached(self) -> bool:
        target = self.target_xp
        return target is not None and self.xp >= target

    @property
    def progress(self) -> Optional[float]:
        """Fração (0-1) do caminho entre o XP do início (ou da última correção) e a meta."""
        target = self.target_xp
        if target is None:
            return None
        if target <= self.start_xp:
            return 1.0
        return min(max((self.xp - self.start_xp) / (target - self.start_xp), 0.0), 1.0)

    def add_items(self, count: int) -> None:
        self.items += max(int(count), 0)

    def sync(self, xp: float) -> None:
        """Corrige o XP total para `xp` (valor real). O ponto de partida passa a ser o valor corrigido
        (menos o já ganho), então a correção não é contada como XP ganho."""
        self.start_xp = max(float(xp) - self.gained, 0.0)

    def xp_per_hour(self, elapsed_s: float) -> float:
        return self.gained / (elapsed_s / 3600) if elapsed_s >= 1 else 0.0

    def eta_seconds(self, elapsed_s: float) -> Optional[float]:
        """Tempo estimado até a meta no ritmo atual; None sem meta ou sem ritmo medido."""
        to_go, rate = self.xp_to_goal, self.xp_per_hour(elapsed_s)
        if to_go is None or rate <= 0:
            return None
        return to_go / rate * 3600


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
    Guarda em config/progress.json o XP estimado de cada habilidade (para a próxima sessão continuar de
    onde esta parou), a meta escolhida e o XP por item de cada alvo. Lê o arquivo a cada chamada.
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

    def save_skill(self, skill: str, xp: float, target_level: Optional[int]) -> None:
        data = self._load()
        data.setdefault("skills", {})[skill] = {"xp": round(float(xp), 1), "target_level": target_level}
        self._save(data)

    def xp_per_item(self, skill: str, target: str) -> float:
        stored = self._load().get("xp_per_item", {}).get(f"{skill}/{target}")
        if isinstance(stored, (int, float)) and stored > 0:
            return float(stored)
        return default_xp_per_item(skill, target)

    def save_xp_per_item(self, skill: str, target: str, value: float) -> None:
        data = self._load()
        data.setdefault("xp_per_item", {})[f"{skill}/{target}"] = float(value)
        self._save(data)
