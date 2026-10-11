"""Detecta no chat do jogo a mensagem "You've just advanced a <habilidade> level! You have reached level N."."""
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

from vision.imageio import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT, "assets", "chat", "levelup_prefix.png")

MATCH_THRESHOLD = 0.80   # nota mínima do casamento (a frase real dá ~1,0; outras linhas do chat, <= 0,45)
SCAN_INTERVAL = 1.0      # s entre varreduras da tela inteira (~125 ms cada); a mensagem fica vários segundos no chat
SAME_LINE_DX = 40        # picos mais próximos que isso (em x) e SAME_LINE_DY (em y) são a mesma mensagem
SAME_LINE_DY = 8
# O chat pode estar em qualquer lugar da tela (o usuário muda as janelas de posição): examina o frame inteiro


class ChatLevelUpWatcher:
    """
    Conta as mensagens de subida de nível visíveis no chat (pela parte fixa "You've just advanced a", que vale
    para qualquer habilidade). poll() devolve True quando aparece uma mensagem NOVA: a mensagem fica várias
    linhas no chat e sobe conforme chegam outras, então só um aumento na contagem vale como evento.
    O nível em si não é lido do chat: quem chama confirma o nível/XP exato na aba Skills.
    """

    SCAN_INTERVAL = SCAN_INTERVAL   # s entre varreduras (alterável em testes)

    def __init__(self, template_path: str = TEMPLATE_PATH, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._last_scan = float("-inf")
        image = read_image(template_path)   # None se o arquivo não existe
        self.template: Optional[np.ndarray] = None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._previous: Optional[int] = None

    @property
    def ready(self) -> bool:
        return self.template is not None

    def reset(self) -> None:
        """Esquece a contagem: a próxima chamada de poll() só registra o que já está no chat (sem evento)."""
        self._previous = None
        self._last_scan = float("-inf")

    def count(self, frame: np.ndarray) -> int:
        """Quantas mensagens de subida de nível estão visíveis no chat agora."""
        if self.template is None or frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape
        if gray.shape[0] < th or gray.shape[1] < tw:
            return 0
        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= MATCH_THRESHOLD)
        kept = []
        for y, x in sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: -result[p[0], p[1]]):
            if all(abs(x - kx) > SAME_LINE_DX or abs(y - ky) > SAME_LINE_DY for ky, kx in kept):
                kept.append((y, x))
        return len(kept)

    def poll(self, frame: np.ndarray) -> bool:
        """True quando uma mensagem nova apareceu desde a última varredura (no máximo uma por SCAN_INTERVAL)."""
        if frame is None:
            return False
        now = self._clock()
        if now - self._last_scan < self.SCAN_INTERVAL:
            return False
        self._last_scan = now
        count = self.count(frame)
        previous, self._previous = self._previous, count
        return previous is not None and count > previous
