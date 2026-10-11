import pytest

from engine.skill_goals import (ProgressStore, SkillGoal, level_for_xp, parse_decimal, parse_int, xp_for_level)


@pytest.mark.parametrize("level, xp", [(1, 0), (2, 83), (10, 1154), (41, 41171), (50, 101333), (60, 273742),
                                       (99, 13034431), (120, 104273167)])
def test_xp_table_matches_runescape(level, xp):
    assert xp_for_level(level) == xp


def test_level_for_xp_boundaries():
    assert level_for_xp(0) == 1
    assert level_for_xp(82.9) == 1
    assert level_for_xp(83) == 2
    assert level_for_xp(40337) == 40          # o exemplo do tooltip do jogo
    assert level_for_xp(13034430) == 98
    assert level_for_xp(13034431) == 99
    assert level_for_xp(10 ** 12) == 120


def test_goal_without_readings_knows_nothing():
    goal = SkillGoal("Mineração", 60)
    assert not goal.has_reading and goal.xp is None and goal.level is None
    assert goal.xp_to_goal is None and not goal.reached and goal.progress is None
    assert goal.xp_per_hour() is None and goal.eta_seconds(100) is None


def test_goal_tracks_readings():
    goal = SkillGoal("Mineração", 41)
    goal.observe(40337, now=0)
    assert goal.level == 40 and goal.xp_to_goal == 834 and not goal.reached
    assert goal.progress == 0 and goal.gained == 0
    assert goal.xp_per_hour() is None                      # uma leitura só: sem ritmo

    goal.observe(40337 + 500, now=30)
    assert goal.xp_per_hour() is None                      # menos de 1 min entre leituras

    goal.observe(40337 + 600, now=120)                     # 600 XP em 2 min = 18.000/h
    assert goal.xp_per_hour() == pytest.approx(18000)
    assert goal.gained == 600 and goal.level == 40
    assert goal.progress == pytest.approx(600 / 834)
    assert goal.eta_seconds(120) == pytest.approx(234 / 18000 * 3600)

    goal.observe(41171, now=200)
    assert goal.reached and goal.level == 41 and goal.xp_to_goal == 0 and goal.progress == 1


def test_estimate_between_readings_follows_the_rate():
    goal = SkillGoal("Mineração", 60)
    goal.observe(100000, now=0)
    goal.observe(110000, now=3600)                         # 10.000 XP/h
    assert goal.estimated_xp(3600) == 110000
    assert goal.estimated_xp(3600 + 1800) == pytest.approx(115000)
    assert goal.eta_seconds(3600 + 1800) == pytest.approx((xp_for_level(60) - 115000) / 10000 * 3600)


def test_goal_without_target():
    goal = SkillGoal("Mineração")
    goal.observe(5000, now=0)
    goal.observe(6000, now=600)
    assert goal.target_xp is None and goal.xp_to_goal is None and goal.progress is None and not goal.reached
    assert goal.xp_per_hour() == pytest.approx(6000)       # o ritmo existe mesmo sem meta


def test_manual_value_restarts_the_baseline_and_is_replaced_by_the_first_game_reading():
    goal = SkillGoal("Mineração", 60)
    goal.observe(1000, now=0, manual=True)                 # ponto de partida digitado
    goal.observe(41000, now=10)                            # 1ª leitura do jogo: vira a base
    assert goal.first == (41000, 10) and goal.gained == 0
    goal.observe(42000, now=130)
    assert goal.gained == 1000 and goal.xp_per_hour() == pytest.approx(30000)

    goal.observe(60000, now=200, manual=True)              # correção à mão: recomeça a medição
    assert goal.first == (60000, 200) and goal.gained == 0 and goal.xp_per_hour() is None


def test_parse_numbers():
    assert parse_int("", "x") is None
    assert parse_int(" 1.234.567 ", "x") == 1234567
    assert parse_int("60", "x") == 60
    assert parse_decimal("2,5", "x") == 2.5
    assert parse_decimal("2.5", "x") == 2.5
    with pytest.raises(ValueError, match="Nível"):
        parse_int("abc", "Nível")
    with pytest.raises(ValueError):
        parse_int("-3", "x")


def test_progress_store_roundtrip(tmp_path):
    store = ProgressStore(str(tmp_path / "progress.json"))
    assert store.skill("Mineração") == {}

    store.save_skill("Mineração", 123456.78, 70)
    assert store.skill("Mineração") == {"xp": 123456.8, "target_level": 70}
    store.save_skill("Mineração", None, 75)                    # sessão sem leitura: mantém o XP salvo
    assert store.skill("Mineração") == {"xp": 123456.8, "target_level": 75}
    store.save_skill("Mineração", 200000, None)
    assert store.skill("Mineração") == {"xp": 200000.0, "target_level": None}
    assert store.skill("Pesca") == {}


def test_progress_store_ignores_corrupt_file(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{ não é json", encoding="utf-8")
    store = ProgressStore(str(path))
    assert store.skill("Mineração") == {}
    store.save_skill("Mineração", 10, 5)                      # sobrescreve o arquivo quebrado
    assert store.skill("Mineração")["xp"] == 10
