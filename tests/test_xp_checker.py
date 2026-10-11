import cv2
import numpy as np
import pytest

from engine.xp_checker import XpChecker
from vision.skills_panel import SkillsPanel, SkillsWindow, skill_cell
from vision.xp_tooltip import GlyphFont, XpReading, XpTooltipReader

WINDOW_AT = (900, 123)         # canto da janela Skills no frame do "jogo"
WINDOW = SkillsWindow(900, 123, 216)
SCREEN_OFFSET = (10, 20)       # onde o frame fica na tela (window_region)
HOTKEY = "h"


def near_start(point, origin=(500, 500)):
    """O mouse volta para PERTO de onde estava (nunca ao pixel exato: seria um padrão fixo)."""
    return ((point[0] - origin[0]) ** 2 + (point[1] - origin[1]) ** 2) ** 0.5 <= 130


class FakeGame:
    """O jogo, em miniatura: a janela Skills abre/fecha e o tooltip aparece com o mouse sobre o ícone."""

    def __init__(self, tmp_path, opens=True, tooltip=True, hotkey_closes=True, already_open=False, close_delay=0):
        shot = cv2.imread("tests/fixtures/skills_mining_tooltip.png")
        self.with_tooltip = shot
        self.window_only = shot.copy()
        self.window_only[58:145, 146:273] = (60, 60, 60)          # sem o tooltip
        self.noise = np.random.default_rng(4).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
        self.open = already_open
        self.opens, self.tooltip, self.hotkey_closes = opens, tooltip, hotkey_closes
        self.close_delay, self.pending_close = close_delay, None   # frames até a janela sumir de vez
        self.mouse = (500, 500)
        self.presses, self.moves, self.clicks = [], [], []
        self.window_region = {"left": SCREEN_OFFSET[0], "top": SCREEN_OFFSET[1], "width": 1335, "height": 913}

    # --- capture ---
    def get_frame(self):
        if self.pending_close is not None:
            self.pending_close -= 1
            if self.pending_close < 0:
                self.open, self.pending_close = False, None
        frame = self.noise.copy()
        if self.open:
            image = self.with_tooltip if self.tooltip and self._hovering_mining() else self.window_only
            x, y = WINDOW_AT
            frame[y:y + image.shape[0], x:x + image.shape[1]] = image
        return frame

    def _hovering_mining(self):
        cx, cy = SkillsPanel.cell_center(WINDOW, skill_cell("Mining"))
        mx, my = self.mouse[0] - SCREEN_OFFSET[0], self.mouse[1] - SCREEN_OFFSET[1]
        return abs(mx - cx) <= 28 and abs(my - cy) <= 11

    # --- keyboard ---
    def press(self, key, **_):
        self.presses.append(key)
        if key == HOTKEY:
            if not self.open and self.opens:
                self.open = True
            elif self.open and self.hotkey_closes:
                if self.close_delay:
                    self.pending_close = self.close_delay
                else:
                    self.open = False

    # --- mouse ---
    def position(self):
        return self.mouse

    def move_to(self, x, y, duration=None):
        self.mouse = (x, y)
        self.moves.append((x, y))

    def human_click_at(self, x, y, button="left"):
        self.clicks.append((x, y))


@pytest.fixture
def make(tmp_path):
    def build(**kwargs):
        game = FakeGame(tmp_path, **kwargs)
        reader = XpTooltipReader(GlyphFont(learned_path=str(tmp_path / "aprendida.json")))
        logs = []
        checker = XpChecker(game, game, game, logs.append, reader=reader, sleep=lambda s: None,
                            debug_dir=str(tmp_path / "debug"))
        return game, checker, logs
    return build


def test_opens_reads_closes_and_restores_the_mouse(make):
    game, checker, logs = make()
    reading = checker.check("Mineração", HOTKEY)

    assert reading == XpReading(xp=40337, next_xp=41171, remainder=834, level=40)
    assert game.presses == [HOTKEY, HOTKEY]                      # abriu e fechou
    assert not game.open
    assert near_start(game.mouse)                                # voltou para perto de onde estava
    hover = game.moves[0]
    cx, cy = SkillsPanel.cell_center(WINDOW, (2, 0))
    assert abs(hover[0] - SCREEN_OFFSET[0] - cx) <= 16 and abs(hover[1] - SCREEN_OFFSET[1] - cy) <= 9   # dentro da célula
    assert any("40.337 XP" in line for line in logs)


def test_leaves_the_window_open_if_it_was_already_open(make):
    game, checker, _ = make(already_open=True)
    assert checker.check("Mining", HOTKEY).xp == 40337
    assert game.presses == [] and game.open
    assert near_start(game.mouse)


def test_fails_cleanly_when_the_hotkey_does_not_open_the_window(make, tmp_path):
    game, checker, logs = make(opens=False)
    assert checker.check("Mineração", HOTKEY) is None
    assert "não foi reconhecida" in checker.last_failure and "melhor nota" in checker.last_failure
    assert game.presses == [HOTKEY] and game.moves == []         # não mexeu o mouse
    assert len(list((tmp_path / "debug").glob("skills_nao_reconhecida_*.jpg"))) == 1   # guardou uma foto da tela
    assert any("falhou" in line for line in logs)


def test_fails_cleanly_when_no_tooltip_shows_up(make):
    game, checker, _ = make(tooltip=False)
    assert checker.check("Mineração", HOTKEY) is None
    assert "tooltip" in checker.last_failure
    assert not game.open and near_start(game.mouse)              # ainda fecha a janela e devolve o mouse


def test_never_clicks_to_close_the_window_even_if_the_hotkey_does_not_close_it(make):
    """Um clique no X poderia cair no mundo do jogo (a janela some antes do clique) e o personagem sairia andando."""
    game, checker, logs = make(hotkey_closes=False)
    assert checker.check("Mineração", HOTKEY).xp == 40337
    assert game.clicks == []
    assert game.open                                              # ficou aberta
    assert any("não clica" in line for line in logs)
    assert near_start(game.mouse)


def test_waits_for_a_slowly_closing_window_without_clicking(make):
    game, checker, logs = make(close_delay=3)                     # a janela leva alguns quadros para sumir
    assert checker.check("Mineração", HOTKEY).xp == 40337
    assert game.clicks == [] and not game.open
    assert not any("não fechou" in line for line in logs)


def test_unknown_skill_does_nothing(make):
    game, checker, _ = make()
    assert checker.check("Culinária Lunar", HOTKEY) is None
    assert "não está mapeada" in checker.last_failure
    assert game.presses == [] and game.moves == []


def test_refuses_when_game_window_is_not_focused(tmp_path):
    game = FakeGame(tmp_path)
    checker = XpChecker(game, game, game, lambda m: None, reader=XpTooltipReader(GlyphFont(learned_path=str(tmp_path / "a.json"))),
                        focus_callback=lambda: False, sleep=lambda s: None)
    assert checker.check("Mineração", HOTKEY) is None
    assert "foco" in checker.last_failure and game.presses == []


class StubReader:
    ready = True

    def __init__(self, kind):
        self.last_failure, self.last_failure_kind, self.calls = "falha", kind, 0

    def read(self, frame):
        self.calls += 1
        return None


@pytest.mark.parametrize("kind, expected_reads", [("ui", XpChecker.TOOLTIP_TRIES), ("numbers", XpChecker.NUMBER_TRIES)])
def test_rereads_more_when_only_the_numbers_are_unreadable(tmp_path, kind, expected_reads):
    game = FakeGame(tmp_path)
    reader = StubReader(kind)
    checker = XpChecker(game, game, game, lambda m: None, reader=reader, sleep=lambda s: None)
    assert checker.check("Mineração", HOTKEY) is None
    assert reader.calls == expected_reads and checker.last_failure_kind == kind
    assert not game.open and near_start(game.mouse)


def test_the_mouse_never_returns_to_the_exact_same_pixel_and_hover_points_vary(tmp_path):
    from antiban import humanize
    humanize.seed(21)
    returns, hovers = set(), set()
    for _ in range(12):
        game = FakeGame(tmp_path)
        reader = XpTooltipReader(GlyphFont(learned_path=str(tmp_path / "aprendida.json")))
        checker = XpChecker(game, game, game, lambda m: None, reader=reader, sleep=lambda s: None,
                            debug_dir=str(tmp_path / "debug"))
        assert checker.check("Mineração", HOTKEY) is not None
        returns.add(game.mouse)
        hovers.add(game.moves[0])
    assert len(returns) > 9 and len(hovers) > 9
    assert all(near_start(point) for point in returns)
    cx, cy = SkillsPanel.cell_center(WINDOW, (2, 0))
    assert all(abs(x - SCREEN_OFFSET[0] - cx) <= 16 and abs(y - SCREEN_OFFSET[1] - cy) <= 9 for x, y in hovers)   # dentro da célula
