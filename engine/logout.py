import os
import time
from typing import Callable, Optional, Tuple

import cv2

from antiban import humanize
from vision.imageio import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUTTON_PATH = os.path.join(ROOT, "assets", "logout_botao.png")


class LogoutAction:
    """
    Desloga pela interface do jogo: abre o menu (tecla Esc), acha o botão de logout pelo template
    assets/logout_botao.png e clica nele. Confere se deu certo (a interface do jogo some) e tenta de novo se o
    jogo recusar (ex.: "não dá para deslogar logo depois de um combate").

    Só roda em emergência (bot perdido ou personagem morto), com o bot já parado. Sem o template o bot não sabe
    onde clicar: não clica em nada e avisa como criar o arquivo.
    """

    MENU_WAIT = (0.8, 1.2)       # espera o menu abrir depois da tecla
    FIND_TRIES = 5               # procuras do botão (o menu pode demorar a desenhar)
    FIND_WAIT = (0.4, 0.6)
    CLICK_WAIT = (3.5, 4.5)      # espera o jogo processar o logout
    RETRY_WAIT = (8.0, 12.0)     # entre tentativas (o jogo pode recusar logo depois de um combate)
    ATTEMPTS = 3
    BUTTON_MATCH = 0.85

    def __init__(self, capture, mouse, keyboard, logger_func, is_ingame: Callable, focus_callback=None,
                 template_path: str = BUTTON_PATH, menu_key: str = "esc", sleep: Callable[[float], None] = time.sleep):
        self.capture = capture
        self.mouse = mouse
        self.keyboard = keyboard
        self.log = logger_func
        self.is_ingame = is_ingame     # frame -> True se a interface do jogo está na tela (False: tela de login)
        self.focus = focus_callback
        self.menu_key = menu_key
        self._sleep = sleep
        self.template_path = template_path
        self.template = read_image(template_path)   # None se o arquivo ainda não existe

    @property
    def ready(self) -> bool:
        return self.template is not None

    def _wait(self, span: Tuple[float, float]) -> None:
        self._sleep(humanize.uniform(*span) * humanize.pace())

    def _find_button(self, tries: int = FIND_TRIES) -> Optional[Tuple[int, int]]:
        """Centro (x, y) do botão no frame, ou None."""
        for attempt in range(tries):
            frame = self.capture.get_frame()
            if frame is not None:
                th, tw = self.template.shape[:2]
                if frame.shape[0] >= th and frame.shape[1] >= tw:
                    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED))
                    if score >= self.BUTTON_MATCH:
                        return x + tw // 2, y + th // 2
            if attempt + 1 < tries:
                self._wait(self.FIND_WAIT)
        return None

    def run(self) -> bool:
        """True se o logout foi concluído (a interface do jogo sumiu)."""
        if not self.ready:
            self.log("[SEGURANÇA] Não deslogou: falta assets/logout_botao.png (o botão de logout do jogo). Para criar: "
                     "abra o menu de logout no jogo e rode 'python tools/ferramenta_setup.py' para recortar o botão.")
            return False

        for attempt in range(1, self.ATTEMPTS + 1):
            if self.focus and not self.focus():
                self.log("[SEGURANÇA] Não deslogou: a janela do jogo não está acessível.")
                return False
            self.log(f"[SEGURANÇA] Deslogando (tentativa {attempt}/{self.ATTEMPTS})...")
            button = self._find_button(tries=1)          # o menu já está aberto (tentativa anterior)?
            if button is None:
                self.keyboard.press(self.menu_key)
                self._wait(self.MENU_WAIT)
                button = self._find_button()
            region = self.capture.window_region
            if button is None or not region:
                self.log("[SEGURANÇA] Botão de logout não encontrado na tela.")
                self._wait(self.RETRY_WAIT)
                continue

            self.mouse.human_click_at(button[0] + region["left"], button[1] + region["top"])
            self._wait(self.CLICK_WAIT)
            frame = self.capture.get_frame()
            if frame is None or not self.is_ingame(frame):
                self.log("[SEGURANÇA] Deslogado.")
                return True
            self.log("[SEGURANÇA] O jogo ainda está aberto (pode ter recusado o logout).")
            self._wait(self.RETRY_WAIT)

        self.log("[SEGURANÇA] Não consegui deslogar. Faça o logout manualmente.")
        return False
