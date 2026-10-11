import glob

import cv2
import numpy as np
import pytest

from vision.chat_levelup import ChatLevelUpWatcher

FIXTURE = "tests/fixtures/chat_levelup_mining.png"   # print real do chat: a mensagem de nível 41 + o prompt do chat


def game_frame(messages_at=(), seed=6):
    """Frame de jogo (ruído) com a mensagem real do chat colada no canto inferior esquerdo, em cada y dado."""
    frame = np.random.default_rng(seed).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
    shot = cv2.imread(FIXTURE)[0:22]                  # só a linha da mensagem (sem o prompt do chat)
    for y in messages_at:
        frame[y:y + shot.shape[0], 20:20 + shot.shape[1]] = shot
    return frame


def test_counts_the_levelup_messages_visible_in_the_chat():
    watcher = ChatLevelUpWatcher()
    assert watcher.ready
    assert watcher.count(game_frame([])) == 0
    assert watcher.count(game_frame([700])) == 1
    assert watcher.count(game_frame([700, 740])) == 2


def test_message_is_found_wherever_the_chat_window_is():
    """O chat pode ser movido para qualquer lugar da tela, inclusive o lado direito e o topo."""
    watcher = ChatLevelUpWatcher()
    shot = cv2.imread(FIXTURE)[0:22]
    for x, y in ((20, 880), (20, 420), (700, 790), (990, 100), (1000, 450), (400, 40)):
        frame = np.random.default_rng(6).integers(0, 255, (913, 1335, 3), dtype=np.uint8)
        frame[y:y + shot.shape[0], x:x + shot.shape[1]] = shot
        assert watcher.count(frame) == 1, (x, y)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_the_full_screen_scan_is_throttled():
    clock = Clock()
    watcher = ChatLevelUpWatcher(clock=clock)
    assert not watcher.poll(game_frame([]))                # varre e registra o que já está no chat
    clock.now += 0.3
    assert not watcher.poll(game_frame([800]))             # cedo demais: nem olha (barato)
    clock.now += 0.8
    assert watcher.poll(game_frame([800]))                 # agora sim, e a mensagem é nova


def test_poll_fires_once_per_new_message():
    watcher = ChatLevelUpWatcher(clock=lambda: 0.0)
    watcher.SCAN_INTERVAL = 0.0
    assert not watcher.poll(game_frame([]))               # 1ª chamada só registra o que já está no chat
    assert not watcher.poll(game_frame([]))
    assert watcher.poll(game_frame([800]))                # mensagem nova
    assert not watcher.poll(game_frame([800]))            # continua na tela: não repete
    assert not watcher.poll(game_frame([787]))            # subiu uma linha (chegou outra mensagem do chat)
    assert not watcher.poll(game_frame([]))               # saiu do chat
    assert watcher.poll(game_frame([800]))                # outra subida de nível


def test_second_message_while_the_first_is_still_visible_counts():
    watcher = ChatLevelUpWatcher(clock=lambda: 0.0)
    watcher.SCAN_INTERVAL = 0.0
    watcher.poll(game_frame([]))
    assert watcher.poll(game_frame([760]))
    assert watcher.poll(game_frame([747, 800]))
    assert not watcher.poll(game_frame([734, 787]))


def test_message_already_in_the_chat_at_start_is_not_an_event():
    watcher = ChatLevelUpWatcher(clock=lambda: 0.0)
    watcher.SCAN_INTERVAL = 0.0
    assert not watcher.poll(game_frame([760]))            # prime: já estava lá
    assert not watcher.poll(game_frame([747]))
    watcher.reset()                                       # novo Start
    assert not watcher.poll(game_frame([700]))
    assert watcher.poll(game_frame([700, 760]))


def test_missing_frame_does_not_break_the_state():
    watcher = ChatLevelUpWatcher(clock=lambda: 0.0)
    watcher.SCAN_INTERVAL = 0.0
    watcher.poll(game_frame([]))
    assert not watcher.poll(None)
    assert watcher.poll(game_frame([800]))


REAL_FRAMES = sorted(glob.glob("dataset/*/images/*.jpg"))
OLD_SESSIONS = [f for f in REAL_FRAMES if "_20261010_06" in f or "_20261010_09" in f]   # sem nenhuma subida de nível
LEVEL_UP_SESSION = [f for f in REAL_FRAMES if "_20261010_14" in f]                        # chat com "reached level 42"


@pytest.mark.skipif(len(OLD_SESSIONS) < 10, reason="capturas reais (dataset/) não estão nesta máquina")
def test_no_false_positive_on_real_frames_with_a_busy_chat():
    """Sessões reais cujo chat é só 'While mining, you find a geode!' e afins: nunca acusa subida de nível."""
    watcher = ChatLevelUpWatcher()
    assert sum(watcher.count(cv2.imread(f)) for f in OLD_SESSIONS[::3]) == 0


@pytest.mark.skipif(len(LEVEL_UP_SESSION) < 10, reason="capturas reais (dataset/) não estão nesta máquina")
def test_finds_the_real_levelup_message_in_real_game_frames():
    """Nas capturas reais do nível 42 a mensagem aparece no chat (janela 'Private Chat'): uma única, sempre."""
    watcher = ChatLevelUpWatcher()
    counts = [watcher.count(cv2.imread(f)) for f in LEVEL_UP_SESSION[::3]]
    assert max(counts) == 1 and sum(c == 1 for c in counts) >= 5


def test_templates_load_safely_even_when_imread_is_patched_like_ultralytics_does(monkeypatch, tmp_path):
    """
    Importar o ultralytics (o bot faz isso) troca cv2.imread: com IMREAD_GRAYSCALE ele devolve (h, w, 1) e, para um
    arquivo que não existe, LANÇA FileNotFoundError em vez de devolver None (derrubaria o bot na inicialização).
    """
    import os
    original = cv2.imread

    def patched(path, flags=cv2.IMREAD_COLOR):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        image = original(path, flags)
        return image[..., None] if image is not None and image.ndim == 2 else image

    monkeypatch.setattr(cv2, "imread", patched)
    from engine.logout import LogoutAction
    from vision.xp_tooltip import XpTooltipReader

    watcher = ChatLevelUpWatcher()
    assert watcher.template.ndim == 2
    assert watcher.count(game_frame([700])) == 1
    reader = XpTooltipReader()
    assert reader.ready and all(label.ndim == 2 for label in reader.labels.values())

    # arquivo opcional que ainda não existe (o botão de logout): fica desligado, sem derrubar nada
    assert not LogoutAction(None, None, None, print, is_ingame=lambda f: True,
                            template_path=str(tmp_path / "logout_botao.png")).ready
