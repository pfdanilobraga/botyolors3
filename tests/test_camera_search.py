import numpy as np
import pytest

from engine.state_seeker import TargetSeekerFSM


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Keyboard:
    def __init__(self):
        self.pressed = []
        self.times = []
        self.clock = None

    def press(self, key, duration=None, settle=(0.1, 0.2)):
        self.pressed.append(key)
        self.times.append(self.clock.now if self.clock else None)


class Inventory:
    bag_open = True

    def is_open(self, frame):
        return self.bag_open

    def is_full(self, frame):
        return False

    def get_slots_roi(self, frame):
        return None


class Detector:
    def __init__(self):
        self.rocks = []

    def detect(self, frame, conf_threshold=0.5):
        return list(self.rocks)


class Screen:
    window_region = {"left": 0, "top": 0, "width": 100, "height": 100}


def make_world(seed=None):
    from antiban import humanize
    if seed is not None:
        humanize.seed(seed)

    class World:
        pass

    w = World()
    w.clock, w.keyboard, w.inventory, w.detector = Clock(), Keyboard(), Inventory(), Detector()
    w.keyboard.clock = w.clock
    w.logs = []
    w.sleeps = []
    w.fsm = TargetSeekerFSM(w.inventory, w.detector, mouse=None, keyboard=w.keyboard, screencap=Screen(),
                            target_class="Iron_rock", logger_func=w.logs.append)
    w.fsm.park_mouse_after_click = False
    w.fsm.clock = w.clock
    w.fsm.sleep = w.sleeps.append               # as esperas de reação não esperam de verdade nos testes
    w.fsm.camera_search_enabled = True
    w.fsm.camera_search_after = 10.0
    w.frame = np.zeros((100, 100, 3), np.uint8)

    def tick(seconds=0.1, times=1):
        for _ in range(times):
            w.clock.now += seconds
            w.fsm.update(w.frame)

    w.tick = tick
    w.fsm.update(w.frame)                       # IDLE -> PROCURANDO_ALVO
    assert w.fsm.state == "PROCURANDO_ALVO"
    return w


@pytest.fixture
def world():
    return make_world()


def test_does_nothing_until_the_wait_is_over(world):
    world.tick(0.5, times=19)                   # ~9,5 s sem ver rochas
    assert world.keyboard.pressed == []
    world.tick(0.5, times=3)
    assert world.keyboard.pressed and set(world.keyboard.pressed) <= {"left", "right"}
    assert any("girando a câmera" in line for line in world.logs)


def test_disabled_never_touches_the_camera(world):
    world.fsm.camera_search_enabled = False
    world.tick(1.0, times=120)
    assert world.keyboard.pressed == []


def test_rotates_in_separate_steps_and_tilts_the_camera_after_each_turn(world):
    world.tick(0.5, times=21)                   # passa dos 10 s
    world.keyboard.pressed.clear()
    world.keyboard.times.clear()
    world.tick(0.5, times=120)                  # 60 s de busca
    keys = world.keyboard.pressed
    assert set(keys) <= {"left", "right", "up", "down"}
    assert 18 <= len(keys) <= 45                # um giro a cada ~1-3 s, nunca um por ciclo de 0,5 s
    assert {"up", "down"} & set(keys)           # deu a volta: inclinou a câmera


def test_camera_search_is_never_the_same_pattern_twice():
    """Esperas, sentidos, tamanho das voltas e inclinações são sorteados: cada busca é diferente."""
    patterns, gap_sets = set(), []
    for seed in range(14):
        world = make_world(seed)
        world.tick(0.5, times=21)
        world.keyboard.pressed.clear()
        world.keyboard.times.clear()
        world.tick(0.25, times=240)             # 60 s
        patterns.add(tuple(world.keyboard.pressed[:14]))
        times = world.keyboard.times
        gap_sets.append([round(b - a, 2) for a, b in zip(times, times[1:])])
    assert len(patterns) >= 12                  # sequências de teclas diferentes
    gaps = [g for gs in gap_sets for g in gs]
    assert min(gaps) < 1.3 and max(gaps) > 2.0 and len({g for g in gaps}) > 6      # esperas variadas, sem período fixo


def test_camera_picks_both_tilt_directions_and_both_turn_directions():
    tilts, turns = set(), set()
    for seed in range(20):
        world = make_world(seed)
        world.tick(0.5, times=21)
        world.tick(0.5, times=200)
        tilts |= {k for k in world.keyboard.pressed if k in ("up", "down")}
        turns |= {k for k in world.keyboard.pressed if k in ("left", "right")}
    assert tilts == {"up", "down"} and turns == {"left", "right"}


def test_stops_as_soon_as_a_rock_shows_up_and_waits_again_afterwards(world):
    world.tick(0.5, times=30)
    assert world.keyboard.pressed
    world.detector.rocks = [{"class": "Iron_rock", "x": 50, "y": 50, "conf": 0.9}]
    world.fsm.update(world.frame)               # achou: segue para o clique
    assert world.fsm.state == "MOVENDO_PARA_ALVO"
    world.keyboard.pressed.clear()

    # perdeu as rochas de novo: precisa esperar outros 10 s antes de girar
    world.detector.rocks = []
    world.fsm.state = "PROCURANDO_ALVO"
    world.tick(0.5, times=19)
    assert world.keyboard.pressed == []
    world.tick(0.5, times=5)
    assert world.keyboard.pressed


def test_does_not_search_while_the_bag_is_not_visible(world):
    world.inventory.bag_open = False
    world.tick(1.0, times=60)
    assert world.keyboard.pressed == []


# ---------------------------------------------------------------- estacionar o mouse depois do clique

def angle_of(point, w=1335, h=913):
    import math
    return math.atan2(point[1] - h // 2, point[0] - w // 2)


def test_parked_mouse_goes_in_every_direction_not_always_the_same_side(world):
    import math
    points = [world.fsm._pick_park_point(1335, 913) for _ in range(400)]
    sectors = [0, 0, 0, 0]                                      # volta dividida em 4 setores de 90 graus
    for point in points:
        sectors[int((math.degrees(angle_of(point)) % 360) // 90)] += 1
    assert all(0.15 < count / len(points) < 0.35 for count in sectors), sectors


def test_parked_mouse_stays_inside_the_window_and_away_from_the_character(world):
    import math
    w, h = 1335, 913
    for _ in range(500):
        x, y = world.fsm._pick_park_point(w, h)
        assert 40 <= x <= w - 40 and 80 <= y <= h - 40
        assert math.hypot(x - w // 2, y - h // 2) >= TargetSeekerFSM.PARK_MIN_DISTANCE - 1


def test_each_parking_changes_direction_from_the_previous_one(world):
    import math
    angles = [angle_of(world.fsm._pick_park_point(1335, 913)) for _ in range(300)]
    for a, b in zip(angles, angles[1:]):
        turn = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
        assert turn >= math.radians(TargetSeekerFSM.PARK_MIN_TURN) - 1e-6


def test_park_point_in_a_small_window_still_returns_a_valid_point(world):
    x, y = world.fsm._pick_park_point(500, 400)
    assert 0 <= x < 500 and 0 <= y < 400


# ---------------------------------------------------------------- INTERAGINDO sem rocha à vista (caso do log das 15:14)

class FakeStamina:
    """A barra aparece alta (90%) acima do personagem, no centro da janela."""

    def read(self, frame):
        return 0.9, (50, 30)


class Mouse:
    def __init__(self):
        self.clicks, self.moves = [], []

    def human_click_at(self, x, y):
        self.clicks.append((x, y))

    def move_to(self, x, y):
        self.moves.append((x, y))


@pytest.fixture
def mining(world, tmp_path):
    """Já minerando (INTERAGINDO) com o reclique ligado e a barra acima do limite."""
    fsm = world.fsm
    world.mouse = Mouse()
    fsm.mouse = world.mouse
    fsm.stamina = FakeStamina()
    fsm.reclick_enabled = True
    fsm.reclick_threshold = 0.7
    fsm.DEBUG_DIR = str(tmp_path / "debug")
    fsm.state = "INTERAGINDO"
    fsm.current_class = "Iron_rock_buff"
    fsm.interaction_start = world.clock.now
    fsm.last_xp_time = world.clock.now
    fsm.validator = None
    fsm.park_mouse_after_click = False
    return world


def keep_xp_flowing(world):
    world.fsm.last_xp_time = world.clock.now      # o indicador de XP continua aparecendo: segue em INTERAGINDO


def test_rocks_not_visible_for_long_with_the_bar_full_sends_the_bot_back_to_searching(mining):
    world = mining
    for _ in range(80):                                        # 40 s: a barra pede reclique e não há rocha
        keep_xp_flowing(world)
        world.tick(0.5)
        if world.fsm.state == "PROCURANDO_ALVO":
            break
    assert world.fsm.state == "PROCURANDO_ALVO"
    assert any("procurando rochas com a câmera" in line for line in world.logs)
    for _ in range(12):                                        # e a câmera passa a girar
        world.tick(0.5)
    assert set(world.keyboard.pressed) & {"left", "right"}


def test_a_debug_photo_of_what_the_detector_saw_is_saved_once_per_episode(mining, tmp_path):
    world = mining
    world.detector.rocks = []
    world.fsm._lost_since = None
    for _ in range(40):
        keep_xp_flowing(world)
        world.tick(0.5)
    photos = list((tmp_path / "debug").glob("sem_rocha_*.jpg"))
    assert len(photos) == 1                                    # uma foto no início do episódio, não uma por aviso


def test_reclick_accepts_any_rock_of_the_target_family(mining):
    """Minerando 'Iron_rock_buff' (que o YOLO não vê agora), ainda reclica numa 'Iron_rock' ao lado."""
    world = mining
    world.detector.rocks = [{"class": "Iron_rock", "x": 80, "y": 60, "conf": 0.8, "box": [60, 40, 100, 80]}]
    for _ in range(30):
        keep_xp_flowing(world)
        world.tick(0.5)
        if world.mouse.clicks:
            break
    assert len(world.mouse.clicks) == 1
    assert world.fsm._lost_since is None
    assert not any("procurando rochas com a câmera" in line for line in world.logs)


def test_camera_is_not_used_while_mining_normally_with_the_bar_low(mining):
    world = mining
    world.fsm.stamina = type("Low", (), {"read": lambda self, frame: (0.3, (50, 30))})()
    for _ in range(80):
        keep_xp_flowing(world)
        world.tick(0.5)
    assert world.fsm.state == "INTERAGINDO" and world.keyboard.pressed == []


def test_camera_disabled_never_leaves_mining_because_of_missing_rocks(mining):
    world = mining
    world.fsm.camera_search_enabled = False
    for _ in range(80):
        keep_xp_flowing(world)
        world.tick(0.5)
    assert world.fsm.state == "INTERAGINDO" and world.keyboard.pressed == []


# ---------------------------------------------------------------- tempo procurando sem ver rochas (segurança)

def test_searching_for_counts_up_while_no_rock_is_seen_even_without_the_camera(world):
    world.fsm.camera_search_enabled = False
    assert world.fsm.searching_for() == 0.0
    world.tick(1.0, times=1)
    world.tick(1.0, times=30)
    assert 29 <= world.fsm.searching_for() <= 32 and world.keyboard.pressed == []


def test_searching_for_resets_when_a_rock_is_found(world):
    world.tick(1.0, times=20)
    assert world.fsm.searching_for() > 15
    world.detector.rocks = [{"class": "Iron_rock", "x": 50, "y": 50, "conf": 0.9}]
    world.fsm.update(world.frame)
    assert world.fsm.searching_for() == 0.0


def test_lost_while_mining_counts_from_when_the_rocks_disappeared(mining):
    world = mining
    for _ in range(80):
        keep_xp_flowing(world)
        world.tick(0.5)
        if world.fsm.state == "PROCURANDO_ALVO":
            break
    assert world.fsm.state == "PROCURANDO_ALVO"
    assert world.fsm.searching_for() >= world.fsm.camera_search_after      # já contava desde que as rochas sumiram


# ---------------------------------------------------------------- nada de tempo fixo nem de clique no mesmo pixel

def test_click_point_is_inside_the_box_and_never_always_the_center(world):
    detection = {"class": "Iron_rock", "x": 150, "y": 250, "box": [100, 200, 200, 300]}
    points = [world.fsm._click_point(detection) for _ in range(500)]
    assert all(100 + 15 <= x <= 200 - 15 and 200 + 15 <= y <= 300 - 15 for x, y in points)      # região central 70%
    assert len(set(points)) > 150
    assert sum(p == (150, 250) for p in points) / len(points) < 0.05
    assert world.fsm._click_point({"class": "x", "x": 7, "y": 9}) == (7, 9)       # sem caixa: o centro


def test_reclick_limits_and_intervals_are_redrawn_each_time(world):
    world.fsm.reclick_threshold = 0.7
    limits, cooldowns = [], []
    for _ in range(300):
        world.fsm._new_reclick_params()
        limits.append(world.fsm._effective_threshold())
        cooldowns.append(world.fsm._reclick_cooldown)
    assert min(limits) >= 0.7 - 0.08 - 1e-9 and max(limits) <= 0.7 + 0.08 + 1e-9
    assert len({round(v, 3) for v in limits}) > 100 and len({round(v, 2) for v in cooldowns}) > 100
    assert min(cooldowns) >= 8.0 * 0.7 - 1e-9 and max(cooldowns) <= 8.0 * 1.5 + 1e-9


def test_changing_the_configured_threshold_redraws_it(world):
    world.fsm.reclick_threshold = 0.5
    first = world.fsm._effective_threshold()
    world.fsm.reclick_threshold = 0.9
    assert abs(world.fsm._effective_threshold() - 0.9) <= 0.08 and abs(first - 0.5) <= 0.08


def test_interaction_limits_are_drawn_for_each_interaction(world):
    gaps, mins, timeouts, no_bar = set(), set(), set(), set()
    for _ in range(200):
        world.fsm._new_interaction()
        gaps.add(round(world.fsm._xp_gap_limit, 3))
        mins.add(round(world.fsm._min_action, 3))
        timeouts.add(round(world.fsm._timeout_limit, 3))
        no_bar.add(round(world.fsm._no_bar_after, 2))
        assert 2.0 <= world.fsm._xp_gap_limit <= 3.6 and 3.2 <= world.fsm._min_action <= 5.0
        assert 4.5 <= world.fsm._timeout_limit <= 7.0 and 25 * 0.7 <= world.fsm._no_bar_after <= 25 * 1.6
    assert min(map(len, (gaps, mins, timeouts, no_bar))) > 150


def test_the_click_comes_after_a_reaction_time_not_right_after_detecting(world):
    world.detector.rocks = [{"class": "Iron_rock", "x": 50, "y": 50, "conf": 0.9, "box": [30, 30, 70, 70]}]
    world.fsm.mouse = Mouse()
    world.fsm.update(world.frame)               # achou -> MOVENDO_PARA_ALVO
    assert world.fsm.state == "MOVENDO_PARA_ALVO" and world.sleeps == []
    world.fsm.update(world.frame)               # clica
    assert world.fsm.mouse.clicks and world.sleeps
    assert 0.1 <= world.sleeps[0] <= 3.0 * 1.0 + 11.0        # reação (e, raramente, uma distração)


def test_reaction_times_before_clicks_vary_a_lot():
    delays = []
    for seed in range(200):
        world = make_world(seed)
        world.detector.rocks = [{"class": "Iron_rock", "x": 50, "y": 50, "conf": 0.9, "box": [30, 30, 70, 70]}]
        world.fsm.mouse = Mouse()
        world.fsm.update(world.frame)
        world.fsm.update(world.frame)
        delays.append(sum(world.sleeps))
    assert len({round(d, 3) for d in delays}) > 120            # (os que batem nos limites se repetem)
    assert max(delays) > 2 * min(delays)


def test_a_reclick_waits_a_reaction_time_and_clicks_inside_the_rock(mining):
    world = mining
    world.detector.rocks = [{"class": "Iron_rock_buff", "x": 80, "y": 60, "conf": 0.8, "box": [60, 40, 100, 80]}]
    for _ in range(80):
        keep_xp_flowing(world)
        world.tick(0.5)
        if world.mouse.clicks:
            break
    assert len(world.mouse.clicks) == 1 and world.sleeps
    x, y = world.mouse.clicks[0]
    assert 60 <= x <= 100 and 40 <= y <= 80


def test_target_choice_varies_among_near_equal_rocks_but_never_picks_a_weak_one_or_a_lower_class(world):
    fsm = world.fsm
    rocks = [{"class": "Iron_rock", "x": 10, "y": 10, "conf": 0.90},
             {"class": "Iron_rock", "x": 20, "y": 20, "conf": 0.86},
             {"class": "Iron_rock", "x": 30, "y": 30, "conf": 0.55}]
    picks = [fsm._pick_target(rocks)["x"] for _ in range(400)]
    assert set(picks) == {10, 20}                                  # as duas boas, nunca a fraca
    assert 0.3 < picks.count(10) / len(picks) < 0.7
    assert fsm._pick_target([rocks[2]])["x"] == 30                 # só há uma: é ela
    assert fsm._pick_target([]) is None

    fsm.priority_classes = ["Iron_rock_buff"]
    mixed = rocks + [{"class": "Iron_rock_buff", "x": 99, "y": 99, "conf": 0.6}]
    assert all(fsm._pick_target(mixed)["x"] == 99 for _ in range(100))   # a classe prioritária sempre vence
