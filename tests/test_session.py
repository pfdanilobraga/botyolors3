import numpy as np

from engine.session import (GOAL_REACHED, LEVEL_UP, TIME_UP, ItemGainTracker, SessionMonitor, format_duration,
                            format_eta, format_number)
from engine.skill_goals import SkillGoal, xp_for_level


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(duration=0, goal=None, stop_on_goal=True):
    clock = Clock()
    monitor = SessionMonitor(clock)
    monitor.configure(duration, goal, stop_on_goal)
    monitor.start()
    return monitor, clock


def kinds(events):
    return [e.kind for e in events]


def test_time_limit_fires_once_and_stops():
    monitor, clock = make(duration=3600)
    clock.now += 3599
    assert monitor.update() == []
    clock.now += 2
    events = monitor.update()
    assert kinds(events) == [TIME_UP] and events[0].stop
    assert monitor.update() == []


def test_no_limit_never_expires():
    monitor, clock = make(duration=0)
    clock.now += 10 ** 6
    assert monitor.update() == []
    assert monitor.snapshot()["remaining"] is None


def test_level_up_and_goal_events():
    goal = SkillGoal("Mineração", xp_for_level(58), xp_per_item=100, target_level=60)
    monitor, _ = make(goal=goal)
    assert monitor.update() == []

    monitor.add_items((xp_for_level(59) - xp_for_level(58)) // 100 + 1)
    events = monitor.update()
    assert kinds(events) == [LEVEL_UP] and "59" in events[0].message and not events[0].stop
    assert monitor.update() == []                       # cada nível avisa uma vez

    monitor.add_items((xp_for_level(60) - xp_for_level(59)) // 100 + 1)
    events = monitor.update()
    assert kinds(events) == [LEVEL_UP, GOAL_REACHED] and events[1].stop
    assert monitor.update() == []
    assert monitor.snapshot()["goal"]["reached"]


def test_goal_does_not_stop_when_disabled():
    goal = SkillGoal("Mineração", xp_for_level(59), xp_per_item=10 ** 6, target_level=60)
    monitor, _ = make(goal=goal, stop_on_goal=False)
    monitor.add_items(1)
    events = monitor.update()
    assert GOAL_REACHED in kinds(events) and not any(e.stop for e in events)


def test_sync_does_not_raise_level_up():
    goal = SkillGoal("Mineração", xp_for_level(50), xp_per_item=35, target_level=70)
    monitor, _ = make(goal=goal)
    monitor.sync(xp_for_level(55))
    assert monitor.update() == []
    assert monitor.snapshot()["goal"]["level"] == 55


def test_items_ignored_before_start_and_after_finish():
    goal = SkillGoal("Mineração", 0, xp_per_item=10, target_level=5)
    monitor = SessionMonitor(Clock())
    monitor.configure(0, goal)
    monitor.add_items(3)
    assert goal.items == 0
    monitor.start()
    monitor.add_items(3)
    monitor.finish()
    monitor.add_items(3)
    assert goal.items == 3


def test_snapshot_and_summary():
    goal = SkillGoal("Mineração", xp_for_level(50), xp_per_item=35, target_level=51)
    monitor, clock = make(duration=7200, goal=goal)
    monitor.add_items(100)
    clock.now += 3600
    snap = monitor.snapshot()
    assert snap["running"] and snap["elapsed"] == 3600 and snap["remaining"] == 3600
    assert snap["goal"]["xp_per_hour"] == 3500 and snap["goal"]["items"] == 100

    summary = monitor.finish()
    assert "01:00:00" in summary and "3.500 XP" in summary and "100 itens" in summary
    assert monitor.finish() is None                      # só encerra uma vez
    clock.now += 500
    assert monitor.snapshot()["elapsed"] == 3600         # congela ao encerrar
    assert not monitor.snapshot()["running"]


def test_formatters():
    assert format_duration(3725) == "01:02:05"
    assert format_eta(None) == "—" and format_eta(30) == "<1min"
    assert format_eta(42 * 60) == "42min" and format_eta(5 * 3600 + 48 * 60) == "5h 48min"
    assert format_number(1234567.4) == "1.234.567"


class FakeInventory:
    """find_item_slots devolve `count` itens (None = mochila fechada); o teste define `count` a cada passo."""

    count = 0

    def find_item_slots(self, frame, template, threshold, min_distance):
        return None if self.count is None else [(i, 0) for i in range(self.count)]


def run_tracker(steps):
    """steps: [(contagem lida, estado da FSM)] -> lista de itens ganhos em cada passo."""
    inventory = FakeInventory()
    tracker = ItemGainTracker(inventory)
    tracker.template = np.zeros((1, 1, 3), np.uint8)
    gains = []
    for count, state in steps:
        inventory.count = count
        gains.append(tracker.update(None, state))
    return gains


def test_tracker_ignores_items_already_in_bag_and_counts_increases():
    steps = [(9, "PROCURANDO_ALVO"), (9, "INTERAGINDO"), (10, "INTERAGINDO"), (10, "INTERAGINDO"),
             (12, "INTERAGINDO")]
    assert run_tracker(steps) == [0, 0, 1, 0, 2]


def test_tracker_does_not_recount_item_hidden_for_a_moment():
    steps = [(5, "INTERAGINDO"), (6, "INTERAGINDO"), (5, "INTERAGINDO"), (6, "INTERAGINDO"), (7, "INTERAGINDO")]
    assert run_tracker(steps) == [0, 1, 0, 0, 1]


def test_tracker_full_bag_round_trip():
    steps = [(0, "PROCURANDO_ALVO"), (26, "INTERAGINDO"), (27, "DROPANDO"),   # entrou em cheio: conta o último
             (20, "DROPANDO"), (0, "DROPANDO"),                               # dropando: só diminui
             (0, "PROCURANDO_ALVO"),                                          # esvaziou: nova base
             (1, "INTERAGINDO")]
    gains = run_tracker(steps)
    assert gains == [0, 26, 1, 0, 0, 0, 1]
    assert sum(gains) == 28


def test_tracker_manual_empty_while_waiting_resets_baseline():
    steps = [(0, "PROCURANDO_ALVO"), (27, "INVENTARIO_CHEIO"), (27, "INVENTARIO_CHEIO"),
             (3, "PROCURANDO_ALVO"), (4, "INTERAGINDO")]
    assert run_tracker(steps) == [0, 27, 0, 0, 1]


def test_tracker_ignores_closed_bag_and_missing_icon():
    assert run_tracker([(5, "INTERAGINDO"), (None, "INTERAGINDO"), (6, "INTERAGINDO")]) == [0, 0, 1]
    tracker = ItemGainTracker(FakeInventory())
    assert not tracker.enabled and tracker.update(None, "INTERAGINDO") == 0
