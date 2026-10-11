import math
import time
from typing import Callable, List, Optional, Tuple

import pydirectinput
import pyautogui

from antiban import humanize

Point = Tuple[float, float]
# Um trecho do movimento: (x, y, espera depois de chegar nesse ponto, em segundos)
Step = Tuple[int, int, float]


def _min_jerk(t: float) -> float:
    """Progresso 0-1 com velocidade em sino: começa e termina devagar, acelera no meio (como a mão humana)."""
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u ** 2 * t * p1[0] + 3 * u * t ** 2 * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u ** 2 * t * p1[1] + 3 * u * t ** 2 * p2[1] + t ** 3 * p3[1])


def _curve_steps(start: Point, end: Point, duration: float) -> List[Step]:
    """Trecho de `start` a `end` em curva cúbica aleatória, velocidade em sino e tremor suave."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return [(int(round(end[0])), int(round(end[1])), 0.0)]
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    side = 1 if humanize.chance(0.5) else -1
    side2 = side if humanize.chance(0.65) else -side            # às vezes uma curva em "S"
    a, b = humanize.uniform(0.18, 0.45), humanize.uniform(0.55, 0.85)
    o1, o2 = humanize.uniform(0.04, 0.22) * dist * side, humanize.uniform(0.02, 0.16) * dist * side2
    p1 = (start[0] + ux * a * dist + nx * o1, start[1] + uy * a * dist + ny * o1)
    p2 = (start[0] + ux * b * dist + nx * o2, start[1] + uy * b * dist + ny * o2)

    count = max(10, int(duration * 70))
    # tremor: ruído gaussiano suavizado (média móvel), que some nas pontas para chegar exatamente ao alvo
    raw = [humanize.gauss_offset(0.9, 0.9) for _ in range(count + 4)]
    steps: List[Step] = []
    for i in range(1, count + 1):
        t = i / count
        x, y = _cubic(start, p1, p2, end, _min_jerk(t))
        window = raw[i:i + 4]
        fade = math.sin(math.pi * t) if t < 1.0 else 0.0
        x += sum(w[0] for w in window) / 4 * fade
        y += sum(w[1] for w in window) / 4 * fade
        steps.append((int(round(x)), int(round(y)), duration / count * humanize.uniform(0.8, 1.25)))
    steps[-1] = (int(round(end[0])), int(round(end[1])), steps[-1][2])
    return steps


def plan_move(start: Point, end: Point, duration: Optional[float] = None) -> List[Step]:
    """
    Plano de um movimento do mouse de `start` até `end`: lista de (x, y, espera). Função pura (não mexe no
    mouse), para poder ser testada. Nunca repete a mesma trajetória: curva, duração, velocidade, tremor e, em
    movimentos longos, uma ultrapassagem seguida de correção, tudo sorteado a cada vez.
    """
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    if dist < 1.0:
        return [(int(round(end[0])), int(round(end[1])), 0.0)]
    if duration is None:
        duration = humanize.human_delay(0.2 + dist / 1500, sigma=0.3, low=0.6, high=2.0)

    if dist > 220 and humanize.chance(0.2):
        # passa um pouco do alvo e corrige (hesitação comum em movimentos longos)
        ux, uy = (end[0] - start[0]) / dist, (end[1] - start[1]) / dist
        over = humanize.uniform(6, 22)
        lateral = humanize.uniform(-8, 8)
        overshoot = (end[0] + ux * over - uy * lateral, end[1] + uy * over + ux * lateral)
        first = _curve_steps(start, overshoot, duration * 0.85)
        first[-1] = (first[-1][0], first[-1][1], humanize.human_delay(0.05, 0.4, 0.5, 2.5))
        return first + _curve_steps(overshoot, end, humanize.human_delay(0.14, 0.3, 0.6, 2.0))
    return _curve_steps(start, end, duration)


class MouseActuator:
    """
    Simulates human-like mouse movements: random cubic curves, bell-shaped speed, smooth tremor and occasional
    overshoot-and-correct, with click points that are never exactly the same.
    """

    def __init__(self, focus_callback: Optional[Callable[[], bool]] = None):
        # Chamado antes de cada clique: garante o foco na janela do jogo (retorna True/False)
        self.focus_callback = focus_callback
        # Disable pyautogui's fail-safe for more control, but keep it in mind
        pyautogui.FAILSAFE = True
        # Set pause between actions to 0 to manage it manually
        pyautogui.PAUSE = 0
        pydirectinput.PAUSE = 0

    def position(self) -> Tuple[int, int]:
        """Posição atual do cursor na tela (x, y)."""
        x, y = pyautogui.position()
        return int(x), int(y)

    def move_to(self, x: int, y: int, duration: float = None):
        """
        Moves the mouse to (x, y) along a freshly randomized human-like path.
        """
        start = pyautogui.position()
        for px, py, wait in plan_move((start[0], start[1]), (x, y), duration):
            pydirectinput.moveTo(px, py)
            if wait > 0:
                time.sleep(wait)

    def click(self, button: str = 'left', delay_range: Optional[Tuple[float, float]] = None):
        """
        Simulates a human click; the hold time is random (a few tens of milliseconds up to ~0.2 s).
        """
        hold = humanize.uniform(*delay_range) if delay_range else humanize.human_delay(0.09, 0.3, 0.55, 2.3)
        pydirectinput.mouseDown(button=button)
        time.sleep(hold)
        pydirectinput.mouseUp(button=button)

    def human_click_at(self, x: int, y: int, button: str = 'left'):
        """
        Adds Gaussian noise to the final target, moves along a random path and clicks.
        """
        if self.focus_callback and not self.focus_callback():
            print(f"[{time.strftime('%X')}] Mouse: janela do jogo sem foco, clique em ({x}, {y}) ignorado")
            return

        # O ponto recebido já é um ponto sorteado dentro do alvo; este ruído extra evita repetir o mesmo pixel
        dx, dy = humanize.gauss_offset(1.6, 1.6, limit=2.0)
        self.move_to(int(round(x + dx)), int(round(y + dy)))
        time.sleep(humanize.human_delay(0.11, 0.4, 0.4, 2.5))
        self.click(button)

if __name__ == "__main__":
    # Test block
    print("Testing human-like mouse movement...")
    actuator = MouseActuator()
    actuator.move_to(500, 500)
    print("Test complete.")
