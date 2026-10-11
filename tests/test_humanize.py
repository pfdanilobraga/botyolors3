import statistics

import pytest

from antiban import humanize


def draws(fn, n=2000):
    return [fn() for _ in range(n)]


def test_human_delay_is_spread_around_the_median_and_bounded():
    values = draws(lambda: humanize.human_delay(2.0))
    assert min(values) >= 2.0 * 0.45 - 1e-9 and max(values) <= 2.0 * 2.6 + 1e-9
    assert 1.8 < statistics.median(values) < 2.2
    assert statistics.pstdev(values) > 0.3                      # nunca um valor fixo
    assert max(values) > 3.0 and min(values) < 1.5              # com cauda: respostas rápidas e outras bem lentas
    assert len({round(v, 3) for v in values}) > 1000          # (os valores cortados nos limites se repetem)


def test_interval_is_wide_not_a_fixed_period():
    values = draws(lambda: humanize.interval(180.0))            # "a cada 3 min"
    assert min(values) >= 180 * 0.55 - 1e-9 and max(values) <= 180 * 1.8 + 1e-9
    assert statistics.pstdev(values) / statistics.mean(values) > 0.2
    assert sum(abs(v - 180) < 9 for v in values) / len(values) < 0.25      # menos de 1/4 cai a ±5% de 3:00


def test_jitter_stays_in_range_and_varies():
    values = draws(lambda: humanize.jitter(8.0, 0.3))
    assert min(values) >= 5.6 and max(values) <= 10.4 and statistics.pstdev(values) > 0.5


def test_point_in_box_stays_inside_the_central_region_and_is_not_always_the_center():
    box = (100, 200, 180, 260)
    points = draws(lambda: humanize.point_in_box(box))
    assert all(100 + 40 - 28 <= x <= 100 + 40 + 28 and 200 + 30 - 21 <= y <= 200 + 30 + 21 for x, y in points)
    assert len(set(points)) > 200                               # pontos diferentes a cada clique
    assert sum(p == (140, 230) for p in points) / len(points) < 0.05
    xs = [x for x, _ in points]
    assert abs(statistics.mean(xs) - 140) < 2                   # centrado, sem viés para um lado


def test_around_is_clamped_to_the_bounds():
    for _ in range(300):
        x, y = humanize.around((10, 10), 50, (0, 0, 20, 20))
        assert 0 <= x <= 20 and 0 <= y <= 20


def test_session_pace_changes_between_sessions_within_its_range():
    paces = {round(humanize.new_session(), 4) for _ in range(50)}
    assert len(paces) > 40 and min(paces) >= humanize.PACE_RANGE[0] and max(paces) <= humanize.PACE_RANGE[1]


def test_pace_scales_every_delay():
    humanize._pace = 1.2
    slow = statistics.median(draws(lambda: humanize.human_delay(2.0)))
    humanize._pace = 0.9
    fast = statistics.median(draws(lambda: humanize.human_delay(2.0)))
    assert slow > fast * 1.25


def test_distraction_is_rare_but_real():
    values = draws(humanize.distraction, 20000)
    hits = [v for v in values if v > 0]
    assert 0.005 < len(hits) / len(values) < 0.03
    assert all(3.0 <= v <= 11.0 for v in hits)


def test_neutral_generator_gives_exact_medians(neutral):
    assert neutral.human_delay(2.0) == 2.0
    assert neutral.interval(180) == 180
    assert neutral.point_in_box((0, 0, 10, 10)) == (5, 5)
    assert neutral.jitter(10, 0.2) == 10
