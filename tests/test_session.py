import statistics

import pytest

from antiban import humanize

from engine.session import (GOAL_REACHED, LEVEL_UP, TIME_UP, SessionMonitor, format_duration, format_eta,
                            format_number)
from engine.skill_goals import SkillGoal, xp_for_level


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def _exact_times(neutral):
    """Estes testes conferem instantes exatos: gerador neutro (tempos = mediana). A aleatoriedade tem testes próprios."""


def make(duration=0, goal=None, stop_on_goal=True, interval=0):
    clock = Clock()
    monitor = SessionMonitor(clock)
    monitor.configure(duration, goal, stop_on_goal, interval)
    monitor.start()
    return monitor, clock


def first_check_window(clock):
    """A 1ª leitura é num instante sorteado entre 1,5 e 9 s depois do início."""
    clock.now += 9.1


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


def test_first_reading_is_not_a_level_up_but_later_ones_are():
    goal = SkillGoal("Mineração", 42)
    monitor, clock = make(goal=goal)
    assert monitor.update() == []                       # sem leitura ainda
    monitor.observe(40337)
    assert monitor.update() == []                       # base: nível 40, sem aviso
    clock.now += 120
    monitor.observe(41171)
    events = monitor.update()
    assert kinds(events) == [LEVEL_UP] and "41" in events[0].message and not events[0].stop
    assert monitor.update() == []                       # cada nível avisa uma vez


def test_goal_reached_stops_the_bot():
    goal = SkillGoal("Mineração", 41)
    monitor, clock = make(goal=goal)
    monitor.observe(40337)
    monitor.update()
    clock.now += 300
    monitor.observe(41300)
    events = monitor.update()
    assert kinds(events) == [LEVEL_UP, GOAL_REACHED] and events[1].stop and "41" in events[1].message
    assert monitor.update() == []
    assert monitor.snapshot()["goal"]["reached"]


def test_goal_already_reached_at_the_first_reading():
    goal = SkillGoal("Mineração", 30)
    monitor, _ = make(goal=goal)
    monitor.observe(40337)
    events = monitor.update()
    assert kinds(events) == [GOAL_REACHED] and "já estava atingida" in events[0].message and events[0].stop


def test_goal_does_not_stop_when_disabled():
    goal = SkillGoal("Mineração", 41)
    monitor, clock = make(goal=goal, stop_on_goal=False)
    monitor.observe(40337)
    monitor.update()
    clock.now += 60
    monitor.observe(41200)
    events = monitor.update()
    assert GOAL_REACHED in kinds(events) and not any(e.stop for e in events)


def test_manual_value_never_raises_level_up():
    goal = SkillGoal("Mineração", 70)
    monitor, _ = make(goal=goal)
    monitor.observe(xp_for_level(50), manual=True)
    monitor.observe(xp_for_level(55), manual=True)
    assert monitor.update() == []
    assert monitor.snapshot()["goal"]["level"] == 55


def test_checks_are_scheduled_with_the_interval():
    goal = SkillGoal("Mineração", 60)
    monitor, clock = make(goal=goal, interval=180)
    assert not monitor.check_due()                      # nem no segundo zero: a 1ª leitura tem hora sorteada
    first_check_window(clock)
    assert monitor.check_due()
    monitor.observe(40000)
    assert not monitor.check_due()
    clock.now += 179
    assert not monitor.check_due()
    clock.now += 2
    assert monitor.check_due()


def test_no_automatic_checks_without_interval_but_button_forces_one():
    goal = SkillGoal("Mineração", 60)
    monitor, clock = make(goal=goal, interval=0)
    clock.now += 10 ** 5
    assert not monitor.check_due()
    monitor.request_check()
    assert monitor.check_due()
    monitor.observe(40000)
    assert not monitor.check_due()


def test_failed_check_retries_then_gives_up_until_the_button_is_pressed():
    goal = SkillGoal("Mineração", 60)
    monitor, clock = make(goal=goal, interval=180)
    first_check_window(clock)
    for attempt in range(SessionMonitor.MAX_FAILURES):
        assert monitor.check_due()
        monitor.check_failed("tooltip não encontrado")
        assert not monitor.check_due()                  # espera antes de tentar de novo
        clock.now += SessionMonitor.RETRY_AFTER_FAILURE_S + 1
    assert not monitor.check_due()                      # desistiu
    snap = monitor.snapshot()
    assert snap["check_gave_up"] and snap["last_failure"] == "tooltip não encontrado"

    monitor.request_check()
    assert monitor.check_due()
    monitor.observe(40000)
    snap = monitor.snapshot()
    assert not snap["check_gave_up"] and snap["check_failures"] == 0 and snap["last_failure"] == ""


def test_early_check_when_the_goal_is_probably_reached():
    goal = SkillGoal("Mineração", 41)
    monitor, clock = make(goal=goal, interval=600)
    monitor.observe(40337)
    clock.now += 120
    monitor.observe(40337 + 600)                        # 18.000 XP/h; faltam 234 XP (~47 s)
    assert not monitor.check_due()
    clock.now += 20
    assert not monitor.check_due()                      # ainda cedo, e a estimativa não chegou
    clock.now += 40                                     # estimativa passa da meta, e já faz 60 s da última leitura
    assert monitor.check_due()
    monitor.observe(40337 + 700)                        # a leitura real ainda não chegou na meta
    assert not monitor.check_due()


def test_snapshot_and_summary():
    goal = SkillGoal("Mineração", 41)
    monitor, clock = make(duration=7200, goal=goal, interval=300)
    monitor.observe(40000)
    clock.now += 600
    monitor.observe(40600)
    snap = monitor.snapshot()
    assert snap["running"] and snap["elapsed"] == 600 and snap["remaining"] == 6600
    g = snap["goal"]
    assert g["xp"] == 40600 and g["gained"] == 600 and g["readings"] == 2 and g["read_age"] == 0
    assert g["xp_per_hour"] == pytest.approx(3600) and g["level"] == 40 and g["target_level"] == 41

    summary = monitor.finish()
    assert "00:10:00" in summary and "40.600 XP" in summary and "+600 XP" in summary
    assert monitor.finish() is None                      # só encerra uma vez
    clock.now += 500
    assert monitor.snapshot()["elapsed"] == 600          # congela ao encerrar
    assert not monitor.snapshot()["running"]


def test_summary_without_any_reading_and_without_goal():
    monitor, clock = make(goal=SkillGoal("Mineração", 41), interval=60)
    clock.now += 5
    assert monitor.finish() == "Sessão encerrada após 00:00:05."
    monitor, clock = make()
    monitor.observe(5)                                   # sem meta/acompanhamento: ignorado
    assert monitor.snapshot()["goal"] is None and not monitor.check_due()


def test_formatters():
    assert format_duration(3725) == "01:02:05"
    assert format_eta(None) == "—" and format_eta(30) == "<1min"
    assert format_eta(42 * 60) == "42min" and format_eta(5 * 3600 + 48 * 60) == "5h 48min"
    assert format_number(1234567.4) == "1.234.567"


def test_number_failures_do_not_make_the_monitor_give_up_early():
    monitor, clock = make(goal=SkillGoal("Mineração", 60), interval=180)
    first_check_window(clock)
    for _ in range(SessionMonitor.MAX_FAILURES + 5):
        assert monitor.check_due()
        monitor.check_failed("dígitos ainda não aprendidos", numbers_only=True)
        assert not monitor.check_due()
        clock.now += SessionMonitor.RETRY_AFTER_NUMBERS_S + 1   # volta a tentar mais cedo que numa falha de tela
    assert not monitor.snapshot()["check_gave_up"]
    for _ in range(SessionMonitor.MAX_NUMBER_FAILURES):          # mas não para sempre
        monitor.check_failed("dígitos ainda não aprendidos", numbers_only=True)
    assert monitor.snapshot()["check_gave_up"]


def test_no_endless_early_checks_after_the_goal_was_reached_and_the_bot_keeps_running():
    goal = SkillGoal("Mineração", 41)
    monitor, clock = make(goal=goal, stop_on_goal=False, interval=600)
    monitor.observe(40337)
    clock.now += 120
    monitor.observe(41300)                                  # meta atingida
    monitor.update()
    clock.now += 40                                         # passou do espaçamento da releitura antecipada
    assert not monitor.check_due()                          # mas a meta já foi atingida: só no intervalo normal
    clock.now += 600
    assert monitor.check_due()


# ---------------------------------------------------------------- nada de "a cada 3 minutos"

def test_gaps_between_xp_reads_are_widely_spread_and_never_a_fixed_period(monkeypatch):
    monkeypatch.undo()                                       # volta ao gerador real (com a semente do conftest)
    humanize.seed(7)
    gaps = []
    for _ in range(300):
        monitor, clock = make(goal=SkillGoal("Mineração", 60), interval=180)
        clock.now += 9.1
        monitor.observe(1000)
        step = 0.0
        while not monitor.check_due():
            clock.now += 1
            step += 1
        gaps.append(step)
    assert statistics.pstdev(gaps) / statistics.mean(gaps) > 0.2
    assert min(gaps) >= 180 * 0.55 - 1 and max(gaps) <= 180 * 1.8 + 1
    assert sum(abs(g - 180) <= 9 for g in gaps) / len(gaps) < 0.25         # no máximo 1/4 perto de "3:00"
    assert max(gaps) > 260 and min(gaps) < 120                             # há leituras bem cedo e bem tarde


def test_the_first_read_happens_at_a_random_moment_not_at_second_zero(monkeypatch):
    monkeypatch.undo()
    humanize.seed(3)
    firsts = []
    for _ in range(200):
        monitor, clock = make(goal=SkillGoal("Mineração", 60), interval=180)
        waited = 0.0
        while not monitor.check_due():
            clock.now += 0.25
            waited += 0.25
        firsts.append(waited)
    assert min(firsts) >= 1.5 and max(firsts) <= 9.5 and len({round(f, 2) for f in firsts}) > 15


def test_retry_after_a_failure_is_drawn_not_fixed(monkeypatch):
    monkeypatch.undo()
    humanize.seed(5)
    waits = []
    for _ in range(150):
        monitor, clock = make(goal=SkillGoal("Mineração", 60), interval=180)
        first_check_window(clock)
        monitor.check_failed("tooltip não encontrado")
        waited = 0
        while not monitor.check_due():
            clock.now += 1
            waited += 1
        waits.append(waited)
    assert statistics.pstdev(waits) > 5 and min(waits) >= 45 * 0.6 - 1 and max(waits) <= 45 * 2.0 + 1


def test_a_requested_read_waits_the_given_human_delay():
    goal = SkillGoal("Mineração", 60)
    monitor, clock = make(goal=goal, interval=0)
    monitor.request_check(delay=4.0)                         # ex.: mensagem de nível no chat: alguns segundos até agir
    assert not monitor.check_due()
    clock.now += 3.9
    assert not monitor.check_due()
    clock.now += 0.2
    assert monitor.check_due()
    monitor.request_check()                                  # o botão da tela: na hora
    assert monitor.check_due()
