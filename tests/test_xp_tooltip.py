import json
import os
import random

import cv2
import numpy as np
import pytest

from engine.skill_goals import xp_for_level
from vision import xp_tooltip as xt
from vision.xp_tooltip import GlyphFont, XpReading, XpTooltipReader, glyph_key, key_to_bitmap, split_glyphs

FIXTURE = "tests/fixtures/skills_mining_tooltip.png"
TEXT_BGR = (180, 215, 225)
BACKGROUND_BGR = (12, 14, 15)

# Bitmaps inventados para os dígitos que a biblioteca ainda não conhece (a forma real só aparece no jogo)
FAKE_GLYPHS = {
    "2": ".###./#...#/....#/...#./..#../.#.../#..../#...#/#####",
    "5": "#####/#..../####./....#/....#/....#/....#/#...#/.###.",
    "6": "..##./.#.../#..../####./#...#/#...#/#...#/#...#/.###.",
    "9": ".###./#...#/#...#/#...#/.####/....#/....#/...#./.##..",
}


def real_glyphs():
    with open("assets/skills/fonte_xp.json", encoding="utf-8") as f:
        data = json.load(f)
    return {char: key_to_bitmap(keys[0]) for char, keys in data.items()}


def render_tooltip(level, xp, origin=(300, 700), size=(913, 1335)):
    """Frame de jogo com um tooltip de habilidade mostrando o XP dado (rótulos reais, números desenhados)."""
    glyphs = {**real_glyphs(), **{char: key_to_bitmap(key) for char, key in FAKE_GLYPHS.items()}}
    shot = cv2.imread(FIXTURE)
    box = shot[60:142, 148:273].copy()
    for y0, y1 in ((83, 96), (98, 110), (113, 125)):   # apaga os números originais (x >= 222 no print)
        box[y0 - 60:y1 - 60, 74:119] = BACKGROUND_BGR
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (*size, 3), dtype=np.uint8)

    def draw_number(top, text, right):
        x = right
        for char in reversed(text):
            bitmap = glyphs[char]
            h, w = bitmap.shape
            y = top + (7 if char == "," else 0)
            x -= w
            box[y:y + h, x:x + w][bitmap] = TEXT_BGR
            x -= 1

    nxt = xp_for_level(level + 1)
    for top, value in ((84 - 60, xp), (99 - 60, nxt), (114 - 60, nxt - xp)):
        draw_number(top, f"{value:,}", 118)
    oy, ox = origin
    frame[oy:oy + box.shape[0], ox:ox + box.shape[1]] = box
    return frame


@pytest.fixture
def isolated_font(tmp_path):
    """Biblioteca real do projeto, mas aprendendo num arquivo temporário (não suja config/)."""
    return GlyphFont(learned_path=str(tmp_path / "aprendida.json"))


def test_reads_the_real_screenshot(isolated_font):
    reader = XpTooltipReader(isolated_font)
    assert reader.ready
    assert reader.read(cv2.imread(FIXTURE)) == XpReading(xp=40337, next_xp=41171, remainder=834, level=40)


def test_reads_tooltip_anywhere_on_screen(isolated_font):
    frame = np.random.default_rng(5).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
    shot = cv2.imread(FIXTURE)
    frame[500:500 + shot.shape[0], 900:900 + shot.shape[1]] = shot
    reading = XpTooltipReader(isolated_font).read(frame)
    assert reading and reading.xp == 40337 and reading.level == 40


def test_no_tooltip_returns_none_with_reason(isolated_font):
    reader = XpTooltipReader(isolated_font)
    noise = np.random.default_rng(9).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
    assert reader.read(noise) is None and "não encontrado" in reader.last_failure
    assert reader.read(None) is None


def test_glyph_roundtrip_and_split():
    gray = np.zeros((12, 20), np.uint8)
    for x0, key in ((1, ".##/###/..#"), (8, "#.#/.#./#.#")):
        bitmap = key_to_bitmap(key)
        gray[2:2 + bitmap.shape[0], x0:x0 + bitmap.shape[1]][bitmap] = 200
    glyphs = split_glyphs(gray)
    assert [glyph_key(g) for g in glyphs] == [".##/###/..#", "#.#/.#./#.#"]


def test_font_tolerates_one_pixel_noise(isolated_font):
    key = next(k for k, c in isolated_font.glyphs.items() if c == "8")
    noisy = key_to_bitmap(key).copy()
    noisy[4, 2] = not noisy[4, 2]
    assert isolated_font.classify(noisy) == "8"
    noisy[0, 0] = not noisy[0, 0]
    noisy[8, 4] = not noisy[8, 4]
    assert isolated_font.classify(noisy) is None


def test_solver_rejects_incoherent_numbers():
    # Current + Remainder != Next level
    assert xt.solve([list("40,337"), list("41,171"), list("835")])[0] is None
    # Next level que não existe na tabela de XP
    assert xt.solve([list("40,337"), list("41,172"), list("835")])[0] is None
    assert xt.solve([list("40,337"), list("41,171"), list("834")])[0].level == 40


def test_solver_deduces_unique_unknown_digit():
    # '?' é um glifo desconhecido: 40,3?7 + 834 = 41,171  ->  ? = 3
    reading, possible = xt.solve([list("40,") + ["#a", "3", "7"], list("41,171"), list("834")])
    assert reading and reading.xp == 40337 and possible == {"#a": {"3"}}


def test_solver_reports_ambiguity_instead_of_guessing():
    lines = [list("40,") + ["#x", "#x", "7"], list("41,171"), ["#y", "#x", "4"]]
    reading, possible = xt.solve(lines)
    assert reading is None
    assert possible["#x"] == {"3", "8"} and possible["#y"] == {"8", "2"}


def test_learns_missing_digits_from_successive_tooltips(isolated_font):
    """Com os dígitos 2, 5, 6 e 9 desconhecidos, leituras sucessivas (XP diferente) ensinam cada um deles."""
    assert isolated_font.digits_missing == "2569"
    reader = XpTooltipReader(isolated_font)
    rng = random.Random(11)
    results = []
    for _ in range(40):
        level = rng.randint(8, 48)      # números mais longos não cabem na área do desenho sintético
        xp = rng.randint(xp_for_level(level), xp_for_level(level + 1) - 1)
        reading = reader.read(render_tooltip(level, xp))
        results.append((reading, xp, level))
        if not isolated_font.digits_missing:
            break

    assert isolated_font.digits_missing == ""
    for char, key in FAKE_GLYPHS.items():
        assert isolated_font.glyphs[key] == char
    # nunca devolveu um número errado: só acertos ou None enquanto aprendia
    assert all(r is None or (r.xp == xp and r.level == level) for r, xp, level in results)
    # e, com a biblioteca completa, lê qualquer número
    for level in (15, 28, 43):
        xp = xp_for_level(level) + (xp_for_level(level + 1) - xp_for_level(level)) * 3 // 7
        assert reader.read(render_tooltip(level, xp)).xp == xp


def test_learned_glyphs_persist_between_runs(tmp_path):
    path = str(tmp_path / "aprendida.json")
    font = GlyphFont(learned_path=path)
    font.add("5", key_to_bitmap(FAKE_GLYPHS["5"]))
    again = GlyphFont(learned_path=path)
    assert again.glyphs[FAKE_GLYPHS["5"]] == "5" and "5" not in again.digits_missing
    assert os.path.exists(path)
