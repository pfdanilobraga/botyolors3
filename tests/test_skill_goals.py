import pytest

from engine.skill_goals import (ProgressStore, SkillGoal, level_for_xp, parse_decimal, parse_int, xp_for_level)


@pytest.mark.parametrize("level, xp", [(1, 0), (2, 83), (10, 1154), (50, 101333), (60, 273742), (99, 13034431),
                                       (120, 104273167)])
def test_xp_table_matches_runescape(level, xp):
    assert xp_for_level(level) == xp


def test_level_for_xp_boundaries():
    assert level_for_xp(0) == 1
    assert level_for_xp(82.9) == 1
    assert level_for_xp(83) == 2
    assert level_for_xp(13034430) == 98
    assert level_for_xp(13034431) == 99
    assert level_for_xp(10 ** 12) == 120


def test_goal_estimates_xp_level_and_distance():
    goal = SkillGoal("Mineração", xp_for_level(59), xp_per_item=80, target_level=60)
    assert goal.level == 59 and not goal.reached and goal.progress == 0
    to_go = xp_for_level(60) - xp_for_level(59)
    assert goal.xp_to_goal == to_go

    goal.add_items(10)
    assert goal.gained == 800 and goal.xp == xp_for_level(59) + 800
    assert 0 < goal.progress < 1

    goal.add_items(int(to_go // 80) + 1)
    assert goal.reached and goal.level == 60 and goal.xp_to_goal == 0 and goal.progress == 1


def test_goal_rate_and_eta():
    goal = SkillGoal("Mineração", xp_for_level(50), xp_per_item=35, target_level=51)
    assert goal.eta_seconds(3600) is None            # ainda sem ganho: sem previsão
    goal.add_items(100)                              # 3.500 XP em 1 h
    assert goal.xp_per_hour(3600) == 3500
    assert goal.eta_seconds(3600) == pytest.approx(goal.xp_to_goal / 3500 * 3600)
    assert goal.xp_per_hour(0) == 0


def test_goal_without_target_or_without_rate():
    goal = SkillGoal("Mineração", 1000)
    assert goal.target_xp is None and goal.xp_to_goal is None and goal.progress is None and not goal.reached
    goal.add_items(5)
    assert goal.gained == 0                          # xp_per_item desconhecido: nada a estimar


def test_sync_corrects_total_but_keeps_gain():
    goal = SkillGoal("Mineração", 1000, xp_per_item=35, target_level=30)
    goal.add_items(10)
    goal.sync(50000)
    assert goal.xp == 50000 and goal.gained == 350
    assert goal.start_level == level_for_xp(50000 - 350)    # a correção não vira "nível ganho"


def test_parse_numbers():
    assert parse_int("", "x") is None
    assert parse_int(" 1.234.567 ", "x") == 1234567
    assert parse_int("60", "x") == 60
    assert parse_decimal("17,5", "x") == 17.5
    assert parse_decimal("17.5", "x") == 17.5
    with pytest.raises(ValueError, match="Nível"):
        parse_int("abc", "Nível")
    with pytest.raises(ValueError):
        parse_int("-3", "x")


def test_progress_store_roundtrip(tmp_path):
    store = ProgressStore(str(tmp_path / "progress.json"))
    assert store.skill("Mineração") == {}
    assert store.xp_per_item("Mineração", "Iron") == 35.0     # padrão
    assert store.xp_per_item("Mineração", "Desconhecido") == 0.0

    store.save_skill("Mineração", 123456.78, 70)
    store.save_xp_per_item("Mineração", "Iron", 37.1)
    assert store.skill("Mineração") == {"xp": 123456.8, "target_level": 70}
    assert store.xp_per_item("Mineração", "Iron") == 37.1
    store.save_skill("Mineração", 200000, None)
    assert store.skill("Mineração")["target_level"] is None
    assert store.xp_per_item("Mineração", "Iron") == 37.1     # preservado


def test_progress_store_ignores_corrupt_file(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{ não é json", encoding="utf-8")
    store = ProgressStore(str(path))
    assert store.skill("Mineração") == {}
    store.save_skill("Mineração", 10, 5)                      # sobrescreve o arquivo quebrado
    assert store.skill("Mineração")["xp"] == 10
