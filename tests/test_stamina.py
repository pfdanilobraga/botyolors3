import glob

import cv2
import numpy as np
import pytest

from vision.stamina import StaminaBarReader

BANNER = "tests/fixtures/critical_swing_banner.png"      # banner real "Critical swing!" (translúcido, ciano)
H, W = 913, 1335
CENTER = (W // 2, H // 2)


def background(seed=2):
    rng = np.random.default_rng(seed)
    base = np.array([60, 150, 190], np.uint8)            # areia
    return np.clip(base + rng.integers(-12, 12, (H, W, 3)), 0, 255).astype(np.uint8)


def bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_bar(frame, fraction, dx=0, dy=-30, width=54):
    """Fio ciano (3 px) sobre trilha verde-azulada escura + barra azul-marinho/laranja logo abaixo, como no jogo."""
    x, y = CENTER[0] - width // 2 + dx, CENTER[1] + dy
    fill = round(width * fraction)
    frame[y:y + 3, x:x + fill] = bgr(90, 120, 250)
    frame[y:y + 3, x + fill:x + width] = (50, 50, 20)
    frame[y + 6:y + 11, x:x + width] = (130, 0, 9)
    frame[y + 6:y + 11, x:x + width // 3] = bgr(15, 220, 230)
    return frame


def bright_banner(width):
    """O banner no pico da animação (ciano com brilho >= 238), na largura pedida."""
    banner = cv2.imread(BANNER)
    hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
    cyan = (hsv[:, :, 0] >= 86) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 70)
    hsv[:, :, 2] = np.where(cyan, np.clip(hsv[:, :, 2].astype(int) + 60, 0, 255), hsv[:, :, 2]).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (width, 28), interpolation=cv2.INTER_NEAREST)


def paste(frame, image, dx=0, dy=0):
    h, w = image.shape[:2]
    x, y = CENTER[0] - w // 2 + dx, CENTER[1] + dy
    frame[y:y + h, x:x + w] = image
    return frame


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
def test_reads_the_bar_fraction(fraction):
    reading = StaminaBarReader().read(draw_bar(background(), fraction))
    assert reading is not None
    assert reading[0] == pytest.approx(fraction, abs=0.05)
    x, y = reading[1]
    assert abs(x - CENTER[0]) < 30 and abs(y - (CENTER[1] - 30)) <= 3      # o centro da barra, perto do personagem


def test_no_bar_returns_none():
    assert StaminaBarReader().read(background()) is None
    assert StaminaBarReader().read(None) is None


@pytest.mark.parametrize("width", [40, 60, 90, 110, 155, 190])
def test_a_bright_banner_is_not_mistaken_for_a_bar(width):
    """O "Critical swing!"/"x4 damage!" é ciano e largo: o leitor antigo o lia como uma barra quase cheia (71-88%)."""
    frame = paste(background(), bright_banner(width), dy=-20)
    assert StaminaBarReader().read(frame) is None


def test_the_real_bar_is_still_read_while_a_banner_is_on_screen():
    frame = draw_bar(background(), 0.5)
    paste(frame, bright_banner(110), dy=15)                  # o banner aparece logo abaixo das barras
    reading = StaminaBarReader().read(frame)
    assert reading is not None and reading[0] == pytest.approx(0.5, abs=0.05)


def test_scattered_cyan_specks_are_not_a_bar():
    frame = background()
    rng = np.random.default_rng(5)
    for _ in range(40):                                      # brilhos/pontos ciano espalhados na área do personagem
        x, y = CENTER[0] + rng.integers(-90, 90), CENTER[1] + rng.integers(-55, 25)
        frame[y:y + 2, x:x + 2] = bgr(90, 120, 250)
    assert StaminaBarReader().read(frame) is None


REAL = sorted(glob.glob("dataset/*/images/*.jpg"))


@pytest.mark.skipif(len(REAL) < 50, reason="capturas reais (dataset/) não estão nesta máquina")
def test_reads_most_real_frames_that_show_the_bar():
    """Nas capturas reais, 77 de 197 mostram a barra; o leitor deve continuar lendo quase todas."""
    reader = StaminaBarReader()
    readings = [reader.read(cv2.imread(f)) for f in REAL]
    found = [r for r in readings if r]
    assert len(found) >= 70
    assert all(0.0 <= r[0] <= 1.0 for r in found)
