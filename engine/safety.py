"""
Rede de segurança do bot: percebe quando ele está PERDIDO (sem ver rochas há muito tempo: morreu e renasceu longe,
foi teleportado, o alvo sumiu) ou com a TELA CONGELADA (o jogo travou: a imagem não muda mais) e pede para parar e
avisar, em vez de seguir agindo para sempre. Quando está perdido ainda pede para deslogar; com a tela congelada não,
porque um cliente travado não responderia ao logout.
"""
import time
from typing import Callable, Optional

import cv2
import numpy as np

from engine.session import FROZEN, LOST, SessionEvent, format_duration


class SafetyGuard:
    """
    Vigia o FSM e a tela. Dispara UM evento (e só um, até reset()):
      - LOST: o FSM está em PROCURANDO_ALVO e não vê rocha nenhuma há `lost_after_s` segundos;
      - FROZEN: a imagem do jogo ficou praticamente idêntica por `frozen_after_s` segundos. O mundo do jogo está
        sempre animado (relógio do minimapa, água, personagens), então tela parada = jogo travado, janela
        minimizada ou coberta por outra janela. Foi o que aconteceu às 22:13: o jogo parou de desenhar e o bot
        clicou 199 vezes na mesma posição durante 20 minutos.
    """

    FROZEN_AFTER_S = 60.0           # s de tela idêntica até considerar o jogo travado
    SAMPLE_EVERY_S = 1.0            # a tela é comparada uma vez por segundo (barato)
    PIXEL_DIFF = 8                  # diferença de brilho (0-255) a partir da qual um pixel conta como "mudou"
    CHANGED_FRACTION = 0.00002      # fração dos pixels que podem mudar e a tela ainda contar como parada (0,002%)

    def __init__(self, fsm, clock: Callable[[], float] = time.time):
        self.fsm = fsm
        self._clock = clock
        self.enabled = False
        self.lost_after_s = 300.0
        self.logout = False
        self.frozen_after_s = self.FROZEN_AFTER_S
        self._fired = False
        self._previous = None
        self._last_sample = float("-inf")
        self._still_since: Optional[float] = None
        self.last_frozen_frame = None   # a última imagem vista com a tela congelada (para guardar como prova)

    def configure(self, enabled: bool, lost_after_s: float, logout: bool) -> None:
        self.enabled = enabled
        self.lost_after_s = max(float(lost_after_s), 1.0)
        self.logout = logout
        self.reset()

    def reset(self) -> None:
        """Nova sessão: libera o disparo e esquece a tela anterior."""
        self._fired = False
        self._previous = None
        self._last_sample = float("-inf")
        self._still_since = None

    def check(self, frame=None) -> Optional[SessionEvent]:
        """O evento de segurança desta rodada (ou None). Chamar a cada ciclo do bot, com o frame atual."""
        if not self.enabled or self._fired:
            return None
        frozen = self._screen_frozen(frame)
        if frozen:
            self._fired = True
            self.last_frozen_frame = frame
            return SessionEvent(
                FROZEN, "Jogo travado",
                f"A tela do jogo não muda há {format_duration(frozen)}: o jogo travou, foi minimizado ou está coberto "
                "por outra janela. O bot foi parado.", stop=True, logout=False)
        if self.fsm.state == "PROCURANDO_ALVO" and self.fsm.searching_for() >= self.lost_after_s:
            self._fired = True
            return SessionEvent(
                LOST, "Bot perdido",
                f"Nenhuma rocha à vista há {format_duration(self.fsm.searching_for())} "
                "(morreu, foi teleportado ou as rochas sumiram): "
                "o bot foi parado." + (" Deslogando..." if self.logout else ""),
                stop=True, logout=self.logout)
        return None

    def _screen_frozen(self, frame) -> float:
        """Há quantos segundos a tela está idêntica, se já passou de `frozen_after_s`; senão 0."""
        if frame is None:
            return 0.0
        now = self._clock()
        if now - self._last_sample < self.SAMPLE_EVERY_S:
            return 0.0
        self._last_sample = now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)   # resolução cheia: o relógio do minimapa é pequeno
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            self._still_since = None
            return 0.0
        changed = float((np.abs(gray - previous) > self.PIXEL_DIFF).mean())
        if changed > self.CHANGED_FRACTION:
            self._still_since = None          # algo se mexeu: a tela está viva
            return 0.0
        if self._still_since is None:
            self._still_since = now - self.SAMPLE_EVERY_S
        still_for = now - self._still_since
        return still_for if still_for >= self.frozen_after_s else 0.0
