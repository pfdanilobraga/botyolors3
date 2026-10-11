import math
import statistics

import pytest

from actuators import mouse as mouse_module
from actuators.mouse import MouseActuator, plan_move


def path_points(start, end, duration=None):
    steps = plan_move(start, end, duration)
    return steps, [(s[0], s[1]) for s in steps]


def total_time(steps):
    return sum(s[2] for s in steps)


def test_path_always_ends_exactly_on_the_target_even_with_overshoot():
    for i in range(300):
        _, points = path_points((100 + i, 700), (1300, 150 + i % 40))
        assert points[-1] == (1300, 150 + i % 40)


def test_path_never_repeats_itself():
    paths = {tuple(path_points((200, 600), (900, 300))[1]) for _ in range(60)}
    assert len(paths) == 60
    lengths = {len(p) for p in paths}
    assert len(lengths) > 10                                    # número de passos (duração) também varia


def test_duration_varies_and_grows_with_the_distance():
    short = [total_time(plan_move((500, 500), (580, 540))) for _ in range(300)]
    far = [total_time(plan_move((100, 100), (1250, 800))) for _ in range(300)]
    assert statistics.median(far) > statistics.median(short) * 1.8
    assert statistics.pstdev(far) / statistics.mean(far) > 0.12      # nunca a mesma duração


def test_speed_is_bell_shaped_slow_at_the_ends_fast_in_the_middle():
    start, end = (100.0, 400.0), (1100.0, 400.0)
    ratios = []
    for _ in range(80):
        steps = plan_move(start, end, duration=0.7)
        xs = [start[0]] + [s[0] for s in steps]
        n = len(steps)
        total = abs(xs[-1] - xs[0])
        first = abs(xs[n // 7] - xs[0]) / total
        mid = abs(xs[n * 4 // 7] - xs[n * 3 // 7]) / total
        ratios.append((first, mid))
    assert statistics.mean(f for f, _ in ratios) < 0.08         # o 1/7 inicial do tempo cobre pouco do caminho
    assert statistics.mean(m for _, m in ratios) > 0.22         # o miolo (1/7 do tempo) cobre bem mais


def test_path_is_a_curve_not_a_straight_line_and_bends_to_both_sides():
    start, end = (200.0, 800.0), (1200.0, 200.0)
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    ux, uy = (end[0] - start[0]) / dist, (end[1] - start[1]) / dist
    sides, deviations = [], []
    for _ in range(200):
        _, points = path_points(start, end, 0.8)
        offsets = [(p[0] - start[0]) * -uy + (p[1] - start[1]) * ux for p in points]
        peak = max(offsets, key=abs)
        deviations.append(abs(peak) / dist)
        sides.append(peak > 0)
    assert statistics.median(deviations) > 0.02
    assert 0.3 < sum(sides) / len(sides) < 0.7                  # ora curva para um lado, ora para o outro


def test_long_moves_sometimes_overshoot_and_short_ones_never_do():
    def overshoots(start, end, runs):
        count = 0
        ux, uy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(ux, uy)
        for _ in range(runs):
            _, points = path_points(start, end, 0.6)
            beyond = max(((p[0] - end[0]) * ux + (p[1] - end[1]) * uy) / norm for p in points)
            count += beyond > 4
        return count / runs

    assert 0.08 < overshoots((100, 500), (1200, 520), 400) < 0.4
    assert overshoots((500, 500), (560, 520), 400) == 0


def test_tremor_is_small_and_smooth_not_spiky():
    _, points = path_points((100, 300), (900, 300), 1.0)
    ys = [p[1] for p in points]
    steps = [abs(b - a) for a, b in zip(ys, ys[1:])]
    assert max(ys) - 300 < 80 and max(steps) <= 6              # sem saltos de pixel


def test_zero_length_move_does_nothing():
    assert plan_move((10, 10), (10, 10)) == [(10, 10, 0.0)]


# ---------------------------------------------------------------- clique com os dispositivos de entrada substituídos

class Recorder:
    def __init__(self):
        self.moves, self.downs, self.ups = [], [], []
        self.sleeps = []
        self.position_xy = (400, 400)

    def moveTo(self, x, y):
        self.moves.append((x, y))
        self.position_xy = (x, y)

    def mouseDown(self, button="left"):
        self.downs.append(button)

    def mouseUp(self, button="left"):
        self.ups.append(button)


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setattr(mouse_module.pydirectinput, "moveTo", r.moveTo)
    monkeypatch.setattr(mouse_module.pydirectinput, "mouseDown", r.mouseDown)
    monkeypatch.setattr(mouse_module.pydirectinput, "mouseUp", r.mouseUp)
    monkeypatch.setattr(mouse_module.pyautogui, "position", lambda: r.position_xy)
    monkeypatch.setattr(mouse_module.time, "sleep", lambda s: r.sleeps.append(s))
    return r


def test_clicks_land_around_the_target_never_always_on_the_same_pixel(rec):
    actuator = MouseActuator()
    landings = []
    for _ in range(120):
        rec.position_xy = (300, 300)
        actuator.human_click_at(800, 500)
        landings.append(rec.position_xy)
    assert len(set(landings)) > 25
    assert all(abs(x - 800) <= 4 and abs(y - 500) <= 4 for x, y in landings)
    assert abs(statistics.mean(x for x, _ in landings) - 800) < 1.0
    assert len(rec.downs) == len(rec.ups) == 120


def test_click_hold_time_varies(rec):
    actuator = MouseActuator()
    for _ in range(100):
        actuator.click()
    holds = [s for s in rec.sleeps]
    assert 0.04 <= min(holds) and max(holds) <= 0.24 and statistics.pstdev(holds) > 0.01


def test_unfocused_game_window_blocks_the_click(rec):
    actuator = MouseActuator(focus_callback=lambda: False)
    actuator.human_click_at(800, 500)
    assert rec.downs == [] and rec.moves == []
