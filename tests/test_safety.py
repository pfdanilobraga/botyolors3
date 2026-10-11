import cv2
import numpy as np
import pytest

from engine.logout import LogoutAction
from engine.safety import SafetyGuard
from engine.session import FROZEN, LOST


class FakeFSM:
    def __init__(self):
        self.state = "PROCURANDO_ALVO"
        self.seconds = 0.0

    def searching_for(self):
        return self.seconds


# ---------------------------------------------------------------- SafetyGuard

def test_lost_event_after_the_timeout_and_only_once():
    fsm = FakeFSM()
    guard = SafetyGuard(fsm)
    guard.configure(True, 300, logout=True)
    fsm.seconds = 299
    assert guard.check() is None
    fsm.seconds = 301
    event = guard.check()
    assert event.kind == LOST and event.stop and event.logout and "00:05:01" in event.message
    assert guard.check() is None                         # só dispara uma vez


def test_lost_without_logout_flag():
    fsm = FakeFSM()
    guard = SafetyGuard(fsm)
    guard.configure(True, 60, logout=False)
    fsm.seconds = 61
    event = guard.check()
    assert event.kind == LOST and event.stop and not event.logout


def test_only_counts_while_searching_for_rocks():
    fsm = FakeFSM()
    guard = SafetyGuard(fsm)
    guard.configure(True, 60, logout=True)
    fsm.seconds = 999
    for state in ("INTERAGINDO", "MOVENDO_PARA_ALVO", "DROPANDO", "INVENTARIO_CHEIO", "IDLE"):
        fsm.state = state
        assert guard.check() is None, state


def test_disabled_never_fires():
    fsm = FakeFSM()
    guard = SafetyGuard(fsm)
    guard.configure(False, 60, logout=True)
    fsm.seconds = 9999
    assert guard.check() is None


def test_reset_rearms_the_guard():
    fsm = FakeFSM()
    guard = SafetyGuard(fsm)
    guard.configure(True, 60, logout=True)
    fsm.seconds = 61
    assert guard.check()
    assert guard.check() is None
    guard.reset()
    assert guard.check() is not None


# ---------------------------------------------------------------- LogoutAction (jogo simulado)

BUTTON = np.zeros((26, 90, 3), np.uint8)
BUTTON[:] = (40, 60, 90)
cv2.putText(BUTTON, "LOG OUT", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
BUTTON_AT = (500, 400)                                    # canto do botão no frame
OFFSET = (10, 20)


class FakeGame:
    """Esc abre o menu com o botão; clicar no botão desloga (a menos que o jogo recuse as primeiras vezes)."""

    def __init__(self, refuse_first=0, menu_has_button=True):
        self.menu = False
        self.logged_out = False
        self.refuse_first = refuse_first
        self.menu_has_button = menu_has_button
        self.presses, self.clicks = [], []
        self.window_region = {"left": OFFSET[0], "top": OFFSET[1], "width": 1335, "height": 913}
        self.base = np.random.default_rng(3).integers(0, 255, (913, 1335, 3), dtype=np.uint8)

    def get_frame(self):
        frame = self.base.copy()
        if self.menu and self.menu_has_button and not self.logged_out:
            h, w = BUTTON.shape[:2]
            frame[BUTTON_AT[1]:BUTTON_AT[1] + h, BUTTON_AT[0]:BUTTON_AT[0] + w] = BUTTON
        return frame

    def press(self, key, **_):
        self.presses.append(key)
        if key == "esc":
            self.menu = not self.menu

    def human_click_at(self, x, y):
        self.clicks.append((x, y))
        h, w = BUTTON.shape[:2]
        fx, fy = x - OFFSET[0], y - OFFSET[1]
        if self.menu and BUTTON_AT[0] <= fx <= BUTTON_AT[0] + w and BUTTON_AT[1] <= fy <= BUTTON_AT[1] + h:
            if self.refuse_first > 0:
                self.refuse_first -= 1
            else:
                self.logged_out = True


@pytest.fixture
def make_logout(tmp_path):
    template = tmp_path / "logout_botao.png"
    cv2.imwrite(str(template), BUTTON)

    def build(game, with_template=True):
        logs = []
        action = LogoutAction(game, game, game, logs.append, is_ingame=lambda frame: not game.logged_out,
                              template_path=str(template if with_template else tmp_path / "nada.png"),
                              sleep=lambda s: None)
        return action, logs
    return build


def test_logout_opens_the_menu_finds_the_button_and_clicks_it(make_logout):
    game = FakeGame()
    action, logs = make_logout(game)
    assert action.ready and action.run()
    assert game.presses == ["esc"] and len(game.clicks) == 1 and game.logged_out
    cx, cy = game.clicks[0]
    assert abs(cx - OFFSET[0] - (BUTTON_AT[0] + 45)) <= 2 and abs(cy - OFFSET[1] - (BUTTON_AT[1] + 13)) <= 2
    assert any("Deslogado" in line for line in logs)


def test_logout_retries_when_the_game_refuses_at_first(make_logout):
    """Ex.: o jogo recusa deslogar logo depois de um combate: tenta de novo sem alternar o menu com o Esc."""
    game = FakeGame(refuse_first=2)
    action, logs = make_logout(game)
    assert action.run()
    assert len(game.clicks) == 3 and game.presses == ["esc"]          # o menu continuou aberto: sem Esc extra


def test_logout_gives_up_after_the_attempts_and_says_so(make_logout):
    game = FakeGame(refuse_first=99)
    action, logs = make_logout(game)
    assert not action.run()
    assert len(game.clicks) == LogoutAction.ATTEMPTS
    assert any("manualmente" in line for line in logs)


def test_logout_without_the_template_clicks_nothing_and_explains(make_logout):
    game = FakeGame()
    action, logs = make_logout(game, with_template=False)
    assert not action.ready and not action.run()
    assert game.presses == [] and game.clicks == []
    assert any("assets/logout_botao.png" in line for line in logs)


def test_logout_button_not_found_never_clicks_blindly(make_logout):
    game = FakeGame(menu_has_button=False)
    action, logs = make_logout(game)
    assert not action.run()
    assert game.clicks == []


# ---------------------------------------------------------------- com a imagem REAL do menu do jogo (Esc -> Options Menu)

MENU = "tests/fixtures/menu_opcoes.png"          # recorte do print real: janela "OPTIONS MENU" com Hop Worlds/Exit to Lobby/Logout


class RealMenuGame:
    """O jogo com o menu real: Esc o abre em qualquer posição; só o clique no botão Logout desloga."""

    def __init__(self, menu_at=(300, 200)):
        self.menu_image = cv2.imread(MENU)
        self.menu_at = menu_at
        self.menu = False
        self.logged_out = False
        self.presses, self.clicks = [], []
        self.window_region = {"left": 10, "top": 20, "width": 1335, "height": 913}
        self.base = np.random.default_rng(8).integers(0, 255, (913, 1335, 3), dtype=np.uint8)

    def get_frame(self):
        frame = self.base.copy()
        if self.menu and not self.logged_out:
            h, w = self.menu_image.shape[:2]
            frame[self.menu_at[1]:self.menu_at[1] + h, self.menu_at[0]:self.menu_at[0] + w] = self.menu_image
        return frame

    def press(self, key, **_):
        self.presses.append(key)
        if key == "esc":
            self.menu = not self.menu

    def human_click_at(self, x, y):
        self.clicks.append((x, y))
        fx, fy = x - 10 - self.menu_at[0], y - 20 - self.menu_at[1]        # em relação ao recorte do menu
        # botão Logout no recorte: x 244-461, y 201-225 (o print original tem x 667-884, y 547-571 e o recorte parte de 423,346)
        if self.menu and 244 <= fx <= 461 and 201 <= fy <= 225:
            self.logged_out = True


@pytest.mark.parametrize("menu_at", [(300, 200), (20, 600), (700, 60), (850, 640)])
def test_the_real_logout_button_is_found_wherever_the_menu_opens_and_only_it_is_clicked(menu_at):
    game = RealMenuGame(menu_at)
    logs = []
    action = LogoutAction(game, game, game, logs.append, is_ingame=lambda frame: not game.logged_out,
                          sleep=lambda s: None)                         # template padrão: assets/logout_botao.png
    assert action.ready
    assert action.run() and game.logged_out
    assert game.presses == ["esc"] and len(game.clicks) == 1
    cx, cy = game.clicks[0]
    # centro do botão Logout (x ~ 352, y ~ 213 no recorte do menu)
    assert abs(cx - 10 - menu_at[0] - 352) <= 6 and abs(cy - 20 - menu_at[1] - 213) <= 4


def test_the_neighbour_buttons_of_the_real_menu_are_never_clicked():
    """Hop Worlds e Exit to Lobby têm o mesmo formato do Logout: a nota do template neles tem que ficar bem abaixo do mínimo."""
    menu = cv2.imread(MENU)
    template = cv2.imread("assets/logout_botao.png")
    scores = cv2.matchTemplate(menu, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(scores >= LogoutAction.BUTTON_MATCH)
    assert len(ys) >= 1 and all(abs(y - 202) <= 3 for y in ys)           # só na altura do botão Logout (y ~ 202)
    # fora do botão certo, a melhor nota é bem menor que o mínimo
    scores[190:215, :] = -1
    assert scores.max() < LogoutAction.BUTTON_MATCH - 0.2


# ---------------------------------------------------------------- tela congelada (jogo travado)

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def moving_frame(seed):
    """Um frame de jogo 'vivo': muita coisa muda de um quadro para o outro (água, personagens, relógio)."""
    return np.random.default_rng(seed).integers(0, 255, (300, 400, 3), dtype=np.uint8)


def make_frozen_guard(frozen_after=60.0):
    clock = Clock()
    guard = SafetyGuard(FakeFSM(), clock=clock)
    guard.configure(True, 300, logout=True)
    guard.frozen_after_s = frozen_after
    return guard, clock


def feed(guard, clock, frames, step=1.0):
    """Entrega um frame por `step` segundos; devolve o primeiro evento (ou None)."""
    for frame in frames:
        clock.now += step
        event = guard.check(frame)
        if event:
            return event
    return None


def test_identical_screen_for_a_minute_is_a_frozen_game():
    guard, clock = make_frozen_guard()
    still = moving_frame(1)
    assert feed(guard, clock, [still] * 58) is None                      # quase 1 min: ainda não
    event = feed(guard, clock, [still] * 10)
    assert event is not None and event.kind == FROZEN and event.stop and not event.logout    # sem logout: o cliente está travado
    assert "não muda" in event.message and guard.last_frozen_frame is still
    assert guard.check(still) is None                                    # só dispara uma vez


def test_a_living_screen_never_triggers_it_even_with_tiny_changes():
    guard, clock = make_frozen_guard()
    frames = []
    base = moving_frame(2).copy()
    for i in range(200):                                                 # quase parada, mas o relógio do minimapa anda
        frame = base.copy()
        frame[10:30, 10:60] = (i * 7) % 255                              # ~1% da imagem muda a cada segundo
        frames.append(frame)
    assert feed(guard, clock, frames) is None


def test_any_movement_resets_the_frozen_timer():
    guard, clock = make_frozen_guard()
    still = moving_frame(3)
    assert feed(guard, clock, [still] * 50) is None
    assert feed(guard, clock, [moving_frame(4)]) is None                 # a tela mexeu: recomeça a contagem
    assert feed(guard, clock, [moving_frame(4)] * 50) is None
    assert feed(guard, clock, [moving_frame(4)] * 15) is not None


def test_missing_frames_and_fast_calls_do_not_count():
    guard, clock = make_frozen_guard()
    still = moving_frame(5)
    for _ in range(500):                                                 # 500 chamadas no mesmo segundo: 1 amostra só
        assert guard.check(still) is None
    assert feed(guard, clock, [None] * 100) is None                      # sem frame (janela sumiu): não conta como travado
    assert feed(guard, clock, [still] * 30) is None


def test_frozen_guard_is_off_when_safety_is_disabled_and_rearms_on_reset():
    guard, clock = make_frozen_guard()
    guard.configure(False, 300, logout=True)
    assert feed(guard, clock, [moving_frame(6)] * 120) is None

    guard, clock = make_frozen_guard()
    still = moving_frame(7)
    assert feed(guard, clock, [still] * 70) is not None
    guard.reset()
    assert feed(guard, clock, [still] * 30) is None                      # nova sessão: contagem do zero
    assert feed(guard, clock, [still] * 40) is not None


def test_lost_still_fires_with_a_moving_screen():
    guard, clock = make_frozen_guard()
    guard.fsm.seconds = 400
    event = guard.check(moving_frame(8))
    assert event.kind == LOST and event.logout
