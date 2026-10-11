"""
Sessão do bot: tempo limite, XP/nível lidos da aba Skills, meta de nível e os eventos que pedem aviso.

O SessionMonitor não conhece o jogo nem a tela: recebe leituras de XP (observe) e, a cada ciclo do bot,
devolve os eventos novos em update() e diz se está na hora de ler de novo em check_due(). Quem chama
decide o que fazer com isso (ler a tela, avisar, parar).
"""
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from antiban import humanize
from engine.skill_goals import SkillGoal

TIME_UP = "time_up"
GOAL_REACHED = "goal_reached"
LEVEL_UP = "level_up"
LOST = "lost"          # bot perdido: muito tempo sem ver rochas
FROZEN = "frozen"      # a tela do jogo não muda (jogo travado, minimizado ou coberto)


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    title: str
    message: str
    stop: bool = False  # True: o bot deve parar por causa deste evento
    logout: bool = False  # True: depois de parar, deslogar do jogo


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

    RETRY_AFTER_FAILURE_S = 45.0   # (mediana) nova tentativa depois de uma leitura que falhou; o tempo é sorteado
    RETRY_AFTER_NUMBERS_S = 20.0   # idem, quando só os números do tooltip não foram decifrados (fonte ainda aprendendo)
    MAX_FAILURES = 4               # falhas seguidas (da tela) até desistir da leitura automática nesta sessão
    MAX_NUMBER_FAILURES = 20       # idem para falhas de números, que são esperadas enquanto a fonte é aprendida
    EARLY_CHECK_SPACING_S = 30.0   # (mediana) releitura antecipada (perto da meta) no mínimo a cada tanto

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._started: Optional[float] = None
        self._ended: Optional[float] = None
        self._running = False
        self._reset(0.0, None, True, 0.0)

    def _reset(self, duration_s: float, goal: Optional[SkillGoal], stop_on_goal: bool, check_interval_s: float):
        self._duration = max(float(duration_s), 0.0)
        self._goal = goal
        self._stop_on_goal = stop_on_goal
        self._interval = max(float(check_interval_s), 0.0)
        self._seen_level: Optional[int] = None
        self._goal_fired = False
        self._time_fired = False
        self._next_check = 0.0
        self._failures = 0
        self._number_failures = 0
        self._last_failure = ""
        self._forced_check = False
        self._forced_at = 0.0          # a partir de quando uma leitura pedida (botão/chat) pode acontecer
        self._early_spacing = self.EARLY_CHECK_SPACING_S

    def configure(self, duration_s: float = 0.0, goal: Optional[SkillGoal] = None, stop_on_goal: bool = True,
                  check_interval_s: float = 0.0) -> None:
        """
        Define a sessão (duration_s = 0: sem limite de tempo; check_interval_s = 0: sem leitura automática).
        Só vale a partir do próximo start().
        """
        with self._lock:
            self._reset(duration_s, goal, stop_on_goal, check_interval_s)
            self._started = self._ended = None
            self._running = False

    def start(self) -> None:
        with self._lock:
            now = self._clock()
            self._started, self._ended, self._running = now, None, True
            # a primeira leitura é perto do início (base do ganho da sessão), mas num instante sorteado
            self._next_check = now + humanize.uniform(1.5, 9.0)

    @property
    def goal(self) -> Optional[SkillGoal]:
        return self._goal

    @property
    def auto_check(self) -> bool:
        return self._interval > 0

    # --- leituras ---

    def observe(self, xp: float, manual: bool = False) -> None:
        """Registra um XP lido do jogo (ou, com manual=True, digitado pelo usuário)."""
        with self._lock:
            if not self._goal:
                return
            now = self._clock()
            self._goal.observe(xp, now, manual)
            if self._seen_level is None or manual:
                self._seen_level = self._goal.level  # base nova: não é subida de nível
            if not manual:
                self._failures = self._number_failures = 0
                self._last_failure = ""
                self._forced_check = False
                self._early_spacing = humanize.human_delay(self.EARLY_CHECK_SPACING_S, 0.35, 0.6, 1.8)
                self._schedule_next(now)

    def check_failed(self, reason: str, numbers_only: bool = False) -> None:
        """
        A leitura falhou: tenta de novo em breve e desiste depois de várias falhas seguidas.
        numbers_only: o tooltip foi visto, só os números não foram decifrados (esperado enquanto a fonte aprende).
        """
        with self._lock:
            if numbers_only:
                self._number_failures += 1
            else:
                self._failures += 1
            self._last_failure = reason
            self._forced_check = False
            delay = self.RETRY_AFTER_NUMBERS_S if numbers_only else self.RETRY_AFTER_FAILURE_S
            self._next_check = self._clock() + humanize.human_delay(delay, 0.35, 0.6, 2.0)

    def request_check(self, delay: float = 0.0) -> None:
        """
        Pede uma leitura (botão da tela, mensagem de nível no chat); também religa a leitura depois de uma desistência.
        `delay`: espera antes de poder ler (uma pessoa leva alguns segundos para notar a mensagem e abrir a aba).
        """
        with self._lock:
            self._forced_check = True
            self._forced_at = self._clock() + max(float(delay), 0.0)
            self._failures = self._number_failures = 0

    def _gave_up(self) -> bool:
        return self._failures >= self.MAX_FAILURES or self._number_failures >= self.MAX_NUMBER_FAILURES

    def _schedule_next(self, now: float) -> None:
        self._next_check = now + humanize.interval(self._interval)   # bem espalhado: nunca "de N em N minutos"

    def check_due(self) -> bool:
        """True quando é hora de ler o XP: intervalo vencido, pedido manual ou meta provavelmente atingida."""
        with self._lock:
            if not self._running or not self._goal:
                return False
            if self._forced_check:
                return self._clock() >= self._forced_at
            if self._interval <= 0 or self._gave_up():
                return False
            now = self._clock()
            if now >= self._next_check:
                return True
            goal = self._goal
            if (goal.target_xp is not None and not goal.reached and goal.last
                    and now - goal.last[1] >= self._early_spacing):
                estimated = goal.estimated_xp(now)
                return estimated is not None and estimated >= goal.target_xp
            return False

    # --- avanço da sessão ---

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
            if goal and goal.has_reading:
                level = goal.level
                if self._seen_level is not None and level > self._seen_level:
                    self._seen_level = level
                    events.append(SessionEvent(LEVEL_UP, f"{goal.skill}: nível {level}",
                                               f"{goal.skill} subiu para o nível {level}."))
                if goal.reached and not self._goal_fired:
                    self._goal_fired = True
                    if goal.readings == 1:
                        message = f"A meta (nível {goal.target_level}) já estava atingida: {goal.skill} está no nível {level}."
                    else:
                        message = f"{goal.skill}: nível {goal.target_level} alcançado."
                    if self._stop_on_goal:
                        message += " O bot será parado."
                    events.append(SessionEvent(GOAL_REACHED, "Meta atingida!", message, stop=self._stop_on_goal))

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
            now = self._clock()
            elapsed = self._elapsed()
            snap = {
                "started": self._started is not None,
                "running": self._running,
                "elapsed": elapsed,
                "duration": self._duration,
                "remaining": max(self._duration - elapsed, 0.0) if self._duration > 0 else None,
                "auto_check": self._interval > 0,
                "check_failures": self._failures + self._number_failures,
                "check_gave_up": self._gave_up(),
                "last_failure": self._last_failure,
                "goal": None,
            }
            goal = self._goal
            if goal:
                snap["goal"] = {
                    "skill": goal.skill,
                    "has_reading": goal.has_reading,
                    "readings": goal.readings,
                    "read_age": now - goal.last[1] if goal.last else None,
                    "level": goal.level,
                    "start_level": goal.start_level,
                    "target_level": goal.target_level,
                    "xp": goal.xp,
                    "gained": goal.gained,
                    "xp_to_goal": goal.xp_to_goal,
                    "xp_per_hour": goal.xp_per_hour(),
                    "eta": goal.eta_seconds(now),
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
            if goal and goal.has_reading:
                summary += f" {goal.skill}: nível {goal.level} ({format_number(goal.xp)} XP)"
                if goal.readings > 1 and goal.gained > 0:
                    summary += f", +{format_number(goal.gained)} XP na sessão (nível {goal.start_level} -> {goal.level})"
                summary += "."
            return summary
