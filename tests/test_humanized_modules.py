import statistics
import time

import numpy as np
import pytest

from engine.drop_module import InventoryDropper
from engine.survival_modules import CrystalMaskModule, PotionModule, PrayerModule


class Keyboard:
    def __init__(self):
        self.pressed = []
        self.holds = []

    def press(self, key, duration=None, settle=(0.1, 0.2)):
        self.pressed.append(key)
        self.holds.append(duration)


FRAME = np.zeros((50, 50, 3), np.uint8)


# ---------------------------------------------------------------- máscara de cristal

def test_crystal_mask_renewal_is_drawn_each_time_never_exactly_290s():
    keyboard, logs, sleeps = Keyboard(), [], []
    module = CrystalMaskModule(keyboard, logs.append)
    module.sleep = sleeps.append
    module.configure(True, "1")
    intervals = []
    for _ in range(150):
        intervals.append(module.interval)
        module.last_activation = time.time() - module.interval - 1          # venceu
        before = len(keyboard.pressed)
        module.check(FRAME)
        assert len(keyboard.pressed) == before + 1 and sleeps                # renovou, com um tempo de reação
    assert all(266.0 <= i <= 292.0 for i in intervals)                       # sempre antes de a máscara de 5 min acabar
    assert len({round(i, 2) for i in intervals}) > 120 and statistics.pstdev(intervals) > 4


def test_crystal_mask_does_not_renew_before_its_drawn_time():
    keyboard = Keyboard()
    module = CrystalMaskModule(keyboard, lambda m: None)
    module.sleep = lambda s: None
    module.configure(True, "1")
    module.last_activation = time.time() - (module.interval - 5)
    module.check(FRAME)
    assert keyboard.pressed == []


# ---------------------------------------------------------------- poção

class Visible:
    def __init__(self, value=True):
        self.value = value

    def is_visible(self, frame, threshold=0.9):
        return self.value


def test_potion_waits_between_doses_with_varying_times_and_redraws_its_cooldown():
    gaps, cooldowns = [], set()
    for _ in range(80):
        keyboard, sleeps = Keyboard(), []
        module = PotionModule(keyboard, lambda m: None)
        module.sleep = sleeps.append
        module.enabled, module.pot_key, module.validator = True, "2", Visible(True)
        module.last_action_time = 0
        module.check(FRAME)
        assert keyboard.pressed == ["2", "2"] and len(sleeps) == 3           # reação + espera depois de cada dose
        gaps.append(sleeps[1])
        cooldowns.add(round(module._cooldown, 2))
    assert len({round(g, 3) for g in gaps}) > 60 and min(gaps) >= 1.6 * 0.7 and max(gaps) <= 1.6 * 1.8
    assert len(cooldowns) > 60


# ---------------------------------------------------------------- oração

class DimIcon:
    def get_match_data(self, frame, threshold=0.8, scales=None):
        return 0.95, np.zeros((6, 6, 3), np.uint8)                          # ícone apagado (brilho 0 < 82)


def test_prayer_reacts_after_a_human_delay_and_with_a_drawn_cooldown():
    cooldowns, delays = set(), []
    for _ in range(80):
        keyboard, sleeps = Keyboard(), []
        module = PrayerModule(keyboard, lambda m: None)
        module.sleep = sleeps.append
        module.enabled, module.pray_key, module.validator = True, "3", DimIcon()
        module.check(FRAME)
        assert keyboard.pressed == ["3"] and len(sleeps) == 1
        delays.append(sleeps[0])
        cooldowns.add(round(module._cooldown, 2))
        keyboard.pressed.clear()
        module.check(FRAME)                                                  # logo depois: ainda no cooldown
        assert keyboard.pressed == []
    assert min(delays) >= 0.55 * 0.4 and max(delays) <= 0.55 * 3.0 and len({round(d, 3) for d in delays}) > 60
    assert min(cooldowns) >= 2.4 and max(cooldowns) <= 4.6 and len(cooldowns) > 60


# ---------------------------------------------------------------- drop

class FakeInventory:
    pass


def make_dropper(hotkey="-"):
    dropper = InventoryDropper(FakeInventory(), None, Keyboard(), None, lambda m: None)
    dropper.hotkey = hotkey
    return dropper


def test_drop_key_presses_have_varying_intervals_and_the_occasional_hesitation():
    dropper = make_dropper()
    dropper.keyboard = Keyboard()
    dropper._mode = "hotkey"
    slots = [(i, 0) for i in range(10)]
    gaps = []
    now = 1000.0
    for _ in range(600):
        dropper._next_press = 0.0
        dropper._step_hotkey(slots, now)
        gaps.append(dropper._next_press - now)
        dropper._presses = 0                                                 # não deixa o "travou" interferir
        dropper._press_started = None
    assert min(gaps) >= 0.32 * 0.6 - 1e-9
    assert statistics.pstdev(gaps) > 0.05 and len({round(g, 3) for g in gaps}) > 250
    assert 0.02 < sum(g > 1.0 for g in gaps) / len(gaps) < 0.12              # hesitações (~6%)
    holds = dropper.keyboard.holds
    assert min(holds) >= 0.065 * 0.55 - 1e-9 and max(holds) <= 0.065 * 2.2 + 1e-9 and statistics.pstdev(holds) > 0.01


def test_drop_start_waits_a_random_time_before_the_first_press():
    delays = set()
    for _ in range(50):
        dropper = make_dropper()
        before = time.time()
        dropper.start()
        delays.add(round(dropper._next_press - before, 2))
    assert len(delays) > 30 and min(delays) >= 0.25


def test_shift_click_order_is_not_always_reading_order():
    firsts = []
    for _ in range(400):
        dropper = make_dropper(hotkey=None)
        dropper._mode = "shift"
        dropper._initial_count = 6
        clicked = []
        dropper._drop_slot = lambda pos, clicked=clicked: clicked.append(pos) or True
        slots = [(100 + i * 50, 100) for i in range(6)]
        dropper._step_shift(slots, now=1000.0)
        firsts.append(clicked[0])
    first_share = sum(p == (100, 100) for p in firsts) / len(firsts)
    assert 0.5 < first_share < 0.85                                          # quase sempre o 1º, mas nem sempre
    assert len(set(firsts)) >= 3
