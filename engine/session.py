"""
Sessão do bot: tempo limite, XP/nível estimados, meta de nível e os eventos que pedem aviso.

O SessionMonitor não conhece o jogo nem a tela: recebe itens contados (add_items) e, a cada ciclo do
bot, devolve os eventos novos em update(). Quem chama decide o que fazer com eles (avisar, parar).
"""
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2

from engine.skill_goals import SkillGoal

TIME_UP = "time_up"
GOAL_REACHED = "goal_reached"
LEVEL_UP = "level_up"


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    title: str
    message: str
    stop: bool = False  # True: o bot deve parar por causa deste evento


def format_duration(seconds: float) -> str:
    """Segundos -> 'HH:MM:SS'."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def format_eta(seconds: Optional[float]) -> str:
    """Segundos -> '5h 48min' / '42min' / '<1min'; '—' sem previsão."""
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    if minutes < 1:
        return "<1min"
    return f"{minutes // 60}h {minutes % 60:02d}min" if minutes >= 60 else f"{minutes}min"


def format_number(value: float) -> str:
    """1234567.8 -> '1.234.568' (milhar com ponto)."""
    return f"{value:,.0f}".replace(",", ".")


class SessionMonitor:
    """Cronômetro com limite opcional + acompanhamento de uma meta de nível. Seguro para uso entre threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._started: Optional[float] = None
        self._ended: Optional[float] = None
        self._running = False
        self._reset(0.0, None, True)

    def _reset(self, duration_s: float, goal: Optional[SkillGoal], stop_on_goal: bool) -> None:
        self._duration = max(float(duration_s), 0.0)
        self._goal = goal
        self._stop_on_goal = stop_on_goal
        self._seen_level = goal.level if goal else 0
        self._goal_fired = False
        self._time_fired = False

    def configure(self, duration_s: float = 0.0, goal: Optional[SkillGoal] = None, stop_on_goal: bool = True) -> None:
        """Define a sessão (duration_s = 0: sem limite de tempo). Só vale a partir do próximo start()."""
        with self._lock:
            self._reset(duration_s, goal, stop_on_goal)
            self._started = self._ended = None
            self._running = False

    def start(self) -> None:
        with self._lock:
            self._started, self._ended, self._running = self._clock(), None, True

    def add_items(self, count: int) -> None:
        with self._lock:
            if self._goal and self._running:
                self._goal.add_items(count)

    def sync(self, xp: float) -> None:
        """Corrige o XP total com o valor real; não gera aviso de nível por causa da correção."""
        with self._lock:
            if self._goal:
                self._goal.sync(xp)
                self._seen_level = self._goal.level

    @property
    def goal(self) -> Optional[SkillGoal]:
        return self._goal

    def _elapsed(self) -> float:
        if self._started is None:
            return 0.0
        return (self._ended if self._ended is not None else self._clock()) - self._started

    def update(self) -> List[SessionEvent]:
        """Eventos novos desde a última chamada: subida de nível, meta atingida, tempo esgotado (cada um uma vez)."""
        events: List[SessionEvent] = []
        with self._lock:
            if not self._running:
                return events

            goal = self._goal
            if goal:
                level = goal.level
                if level > self._seen_level:
                    self._seen_level = level
                    events.append(SessionEvent(
                        LEVEL_UP, f"{goal.skill}: nível {level}",
                        f"{goal.skill} subiu para o nível {level} (estimado)."))
                if goal.reached and not self._goal_fired:
                    self._goal_fired = True
                    events.append(SessionEvent(
                        GOAL_REACHED, "Meta atingida!",
                        f"{goal.skill}: nível {goal.target_level} alcançado (estimado)."
                        + (" O bot será parado." if self._stop_on_goal else ""),
                        stop=self._stop_on_goal))

            elapsed = self._elapsed()
            if self._duration > 0 and elapsed >= self._duration and not self._time_fired:
                self._time_fired = True
                events.append(SessionEvent(
                    TIME_UP, "Tempo esgotado",
                    f"O tempo de bot ({format_duration(self._duration)}) acabou. O bot será parado.", stop=True))
        return events

    def snapshot(self) -> Dict:
        """Estado atual para a tela (valores simples, sem objetos compartilhados)."""
        with self._lock:
            elapsed = self._elapsed()
            snap = {
                "started": self._started is not None,
                "running": self._running,
                "elapsed": elapsed,
                "duration": self._duration,
                "remaining": max(self._duration - elapsed, 0.0) if self._duration > 0 else None,
                "goal": None,
            }
            goal = self._goal
            if goal:
                snap["goal"] = {
                    "skill": goal.skill,
                    "level": goal.level,
                    "start_level": goal.start_level,
                    "target_level": goal.target_level,
                    "xp": goal.xp,
                    "gained": goal.gained,
                    "items": goal.items,
                    "xp_per_item": goal.xp_per_item,
                    "xp_to_goal": goal.xp_to_goal,
                    "xp_per_hour": goal.xp_per_hour(elapsed),
                    "eta": goal.eta_seconds(elapsed),
                    "progress": goal.progress,
                    "reached": goal.reached,
                }
            return snap

    def finish(self) -> Optional[str]:
        """Encerra a sessão e devolve o resumo (None se ela não estava rodando)."""
        with self._lock:
            if not self._running:
                return None
            self._ended, self._running = self._clock(), False
            summary = f"Sessão encerrada após {format_duration(self._elapsed())}."
            goal = self._goal
            if goal and goal.xp_per_item > 0:
                summary += (f" ~{format_number(goal.gained)} XP estimados ({goal.items} itens),"
                            f" nível {goal.start_level} -> {goal.level}.")
            return summary


class ItemGainTracker:
    """
    Conta os itens que entram na mochila (ex.: minérios) pelo ícone do item, o mesmo template do Auto-drop.

    Só conta AUMENTOS acima do maior valor já visto na rodada ("pico"): o que já estava na mochila ao
    começar não vale, e um item escondido por um instante (tooltip, mouse) não conta de novo ao reaparecer.
    Ao entrar em mochila cheia/dropando a contagem é feita uma vez (o último minério); ao sair dela
    (mochila esvaziada) a contagem recomeça do zero.
    """

    FULL_STATES = ("INVENTARIO_CHEIO", "DROPANDO")
    MATCH_THRESHOLD = 0.85
    MIN_SLOT_DISTANCE = 15

    def __init__(self, inventory):
        self.inventory = inventory
        self.template = None
        self.reset()

    def configure(self, item_image: Optional[str]) -> bool:
        """item_image: nome do arquivo em assets/. Retorna True se o ícone foi carregado (contagem ativa)."""
        self.template = None
        self.reset()
        if item_image:
            self.template = cv2.imread(os.path.join("assets", item_image))
        return self.template is not None

    @property
    def enabled(self) -> bool:
        return self.template is not None

    def reset(self) -> None:
        self._peak: Optional[int] = None
        self._state: Optional[str] = None

    def update(self, frame, state: str) -> int:
        """Itens novos vistos neste frame (0 na maioria das chamadas). `state` = estado atual da FSM."""
        if self.template is None:
            return 0
        previous, self._state = self._state, state
        in_full, was_full = state in self.FULL_STATES, previous in self.FULL_STATES
        if in_full and was_full:
            return 0  # dropando/aguardando: a contagem só diminui
        if was_full:
            self._peak = None  # mochila esvaziada: o próximo valor lido é a nova base

        slots = self.inventory.find_item_slots(frame, self.template, self.MATCH_THRESHOLD, self.MIN_SLOT_DISTANCE)
        if slots is None:
            return 0  # mochila não visível agora
        count = len(slots)
        if self._peak is None:
            self._peak = count
            return 0
        gained = max(count - self._peak, 0)
        self._peak = max(count, self._peak)
        return gained
