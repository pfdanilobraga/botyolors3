import cv2
import numpy as np
import pytest

from vision.skills_panel import SkillsPanel, SkillsWindow, skill_cell

FIXTURE = "tests/fixtures/skills_mining_tooltip.png"


@pytest.mark.parametrize("name, cell", [("Mining", (2, 0)), ("Mineração", (2, 0)), ("mineracao", (2, 0)),
                                        ("Pesca", (2, 2)), ("Attack", (0, 0)), ("Woodcutting", (2, 5)),
                                        ("Necromancy", (1, 9))])
def test_skill_cells_follow_the_game_layout(name, cell):
    assert skill_cell(name) == cell


def test_unknown_and_elite_skills_have_no_cell():
    assert skill_cell("Culinária Lunar") is None
    assert skill_cell("Invention") is None       # curva de XP própria: sem leitura automática


def test_locates_the_window_anywhere_and_measures_its_width():
    panel = SkillsPanel()
    assert panel.ready
    shot = cv2.imread(FIXTURE)
    assert panel.locate(shot) == SkillsWindow(0, 0, 216)

    for x, y in ((900, 123), (1040, 500), (20, 600), (1060, 520)):          # em qualquer canto da tela
        frame = np.random.default_rng(2).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
        frame[y:y + shot.shape[0], x:x + shot.shape[1]] = shot[:min(shot.shape[0], 913 - y)]
        window = panel.locate(frame)
        assert window == SkillsWindow(x, y, 216), (x, y)
    assert panel.cell_center(SkillsWindow(900, 123, 216), skill_cell("Mining")) == (900 + 180, 123 + 48)
    assert panel.locate(np.zeros((913, 1335, 3), np.uint8)) is None            # janela fechada
    assert panel.last_score < 0.85


def test_a_narrower_window_from_the_live_game_is_found_and_its_columns_follow_the_width():
    """Captura ao vivo do jogo do usuário: a janela tem 208 px (o print inicial tinha 216) e as colunas se ajustam."""
    panel = SkillsPanel()
    live = cv2.imread("tests/fixtures/skills_window_narrow.png")            # recorte: a janela começa em (2, 2)
    window = panel.locate(live)
    assert window == SkillsWindow(2, 2, 208)
    # centros das 3 colunas medidos na captura (x relativo à âncora): ~36,5 / 104 / 172,5
    centers = [panel.cell_center(window, (col, 0))[0] - window.x for col in range(3)]
    assert [abs(c - m) <= 3 for c, m in zip(centers, (36.5, 104, 172.5))] == [True] * 3


def test_window_is_still_found_when_the_background_behind_it_changes():
    """A âncora não pode depender do cenário atrás da janela (o ícone do canto fica por cima do mundo)."""
    panel = SkillsPanel()
    shot = cv2.imread(FIXTURE)
    for seed in range(5):
        frame = np.random.default_rng(seed).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
        frame[300:300 + shot.shape[0], 500:500 + shot.shape[1]] = shot
        assert panel.locate(frame) is not None


def test_failed_search_keeps_the_best_score_for_diagnosis():
    panel = SkillsPanel()
    assert panel.locate(np.zeros((913, 1335, 3), np.uint8)) is None
    assert 0.0 <= panel.last_score < 0.85


CELLS = {216: [(10, 65), (80, 136), (152, 207)],    # janela do print inicial
         208: [(9, 64), (75, 132), (145, 200)]}     # janela ao vivo (medidas nas capturas reais)


@pytest.mark.parametrize("width", [216, 208])
def test_cell_centers_sit_in_the_middle_of_the_measured_cells(width):
    """Linhas a cada 27 px a partir de y=36 (altura 24); colunas conforme a largura da janela."""
    window = SkillsWindow(0, 0, width)
    for col, (x0, x1) in enumerate(CELLS[width]):
        for row in range(10):
            x, y = SkillsPanel.cell_center(window, (col, row))
            y0 = 36 + 27 * row
            assert x0 + 8 <= x <= x1 - 8, (width, col, row)
            assert y0 + 8 <= y <= y0 + 24 - 8, (width, col, row)
