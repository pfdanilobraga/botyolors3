import glob
import os

import cv2
import numpy as np
import pytest

from engine.drop_module import InventoryDropper
from engine.session import ItemGainTracker
from vision.inventory import InventoryChecker

ANCHOR = "assets/mochila_aberta_ancora.png"
EMPTY_SLOT = "assets/slot_vazio_mochila.png"
ORE = "assets/iron_ore.png"


@pytest.fixture
def inventory():
    return InventoryChecker(ANCHOR, EMPTY_SLOT)


def synthetic_frame(ore_positions, with_anchor=True):
    """Frame 'de jogo' com a âncora da mochila e o ícone do minério colado nas posições dadas (relativas à âncora)."""
    rng = np.random.default_rng(7)
    frame = rng.integers(20, 60, (913, 1335, 3), dtype=np.uint8)
    ax, ay = 1015, 474
    if with_anchor:
        anchor = cv2.imread(ANCHOR)
        frame[ay:ay + anchor.shape[0], ax:ax + anchor.shape[1]] = anchor
    ore = cv2.imread(ORE)
    for dx, dy in ore_positions:
        frame[ay + dy:ay + dy + ore.shape[0], ax + dx:ax + dx + ore.shape[1]] = ore
    return frame


def test_find_item_slots_counts_icons_in_reading_order(inventory):
    ore = cv2.imread(ORE)
    frame = synthetic_frame([(10, 40), (60, 40), (10, 80)])
    slots = inventory.find_item_slots(frame, ore)
    assert slots is not None and len(slots) == 3
    assert slots == sorted(slots, key=lambda c: (round(c[1] / 20), c[0]))


def test_find_item_slots_none_when_bag_closed_or_empty(inventory):
    ore = cv2.imread(ORE)
    assert inventory.find_item_slots(synthetic_frame([(10, 40)], with_anchor=False), ore) is None
    assert inventory.find_item_slots(synthetic_frame([]), ore) == []


def test_dropper_find_slots_delegates_to_inventory(inventory):
    dropper = InventoryDropper(inventory, None, None, None, print)
    assert dropper.find_slots(synthetic_frame([(10, 40)])) is None      # sem template carregado
    dropper.template = cv2.imread(ORE)
    assert len(dropper.find_slots(synthetic_frame([(10, 40), (60, 40)]))) == 2


MITHRIL_FRAMES = sorted(glob.glob("dataset/mithril/images/*.jpg"))


@pytest.mark.skipif(len(MITHRIL_FRAMES) < 10, reason="capturas reais (dataset/) não estão nesta máquina")
def test_real_mining_session_counts_each_new_ore(inventory):
    """Nas capturas reais a mochila vai de 9 para 15 minérios: a sessão ganhou exatamente 6."""
    tracker = ItemGainTracker(inventory)
    assert tracker.configure("mithril_ore.png")
    counts = [len(inventory.find_item_slots(cv2.imread(f), tracker.template, 0.85, 15) or []) for f in MITHRIL_FRAMES]
    assert counts[0] == 9 and counts[-1] == 15

    gained = sum(tracker.update(cv2.imread(f), "INTERAGINDO") for f in MITHRIL_FRAMES)
    assert gained == counts[-1] - counts[0]
