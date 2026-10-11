import os
import time
from typing import Callable, Optional, Tuple

import cv2

from antiban import humanize
from engine.session import format_number
from vision.skills_panel import ANCHOR_MATCH, SkillsPanel, SkillsWindow, skill_cell
from vision.xp_tooltip import XpReading, XpTooltipReader


class XpChecker:
    """
    Lê o XP real de uma habilidade na aba Skills do jogo: abre a janela (tecla de atalho), passa o mouse no
    ícone da habilidade, lê o tooltip ("Current XP / Next level / Remainder"), fecha a janela e devolve o
    mouse para onde estava. Só abre/fecha a janela se ela não estava aberta antes.

    Bloqueia por alguns segundos: chame entre as ações da FSM (nunca no meio de um drop ou de um clique).
    """

    OPEN_WAIT = (0.6, 0.9)        # espera depois de apertar a tecla da janela
    HOVER_WAIT = (0.7, 1.0)       # o tooltip aparece um instante depois do mouse parar
    TOOLTIP_TRIES = 3             # leituras seguidas quando o tooltip nem aparece
    NUMBER_TRIES = 6              # quando o tooltip aparece mas os números não decifram: o XP muda a cada poucos
                                  # segundos, e cada leitura nova ajuda a decifrar os dígitos ainda não aprendidos
    CLOSE_WAIT = (0.25, 0.4)      # espera entre conferências de que a aba fechou
    CLOSE_POLLS = 6               # conferências (~2 s) antes de desistir de esperar a aba fechar
    HOVER_SIGMA = (7.0, 3.5)      # px (x, y): o mouse cai num ponto sorteado da célula, não no centro exato
    RETURN_SIGMA = 30.0           # px: o mouse volta para perto de onde estava, nunca ao pixel exato
    OPEN_POLLS = 5                # conferências (~2 s) de que a aba abriu depois de apertar a tecla
    DEBUG_DIR = "debug"           # foto da tela quando a aba não é reconhecida, para diagnóstico
    DEBUG_MAX_FILES = 6

    def __init__(self, capture, mouse, keyboard, logger_func, panel: Optional[SkillsPanel] = None,
                 reader: Optional[XpTooltipReader] = None, focus_callback: Optional[Callable[[], bool]] = None,
                 sleep: Callable[[float], None] = time.sleep, debug_dir: Optional[str] = None):
        self.capture = capture
        self.mouse = mouse
        self.keyboard = keyboard
        self.log = logger_func
        self.panel = panel or SkillsPanel()
        self.reader = reader or XpTooltipReader()
        self.focus = focus_callback
        self._sleep = sleep
        self.debug_dir = debug_dir or self.DEBUG_DIR
        self.last_failure = ""
        self.last_failure_kind = ""   # "ui" (tela/teclas) ou "numbers" (tooltip visto, números não decifrados)

    @property
    def ready(self) -> bool:
        return self.panel.ready and self.reader.ready

    def _wait(self, span: Tuple[float, float]) -> None:
        self._sleep(humanize.uniform(*span) * humanize.pace())

    def _fail(self, reason: str, kind: str = "ui") -> None:
        self.last_failure, self.last_failure_kind = reason, kind
        self.log(f"[XP] Leitura falhou: {reason}")

    def _absolute(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        region = self.capture.window_region
        return (point[0] + region["left"], point[1] + region["top"]) if region else None

    def _locate_panel(self, wait: bool) -> Optional[SkillsWindow]:
        for _ in range(self.OPEN_POLLS if wait else 1):
            window = self.panel.locate(self.capture.get_frame())
            if window or not wait:
                return window
            self._wait((0.3, 0.45))
        return None

    def _save_debug(self, label: str) -> Optional[str]:
        """Guarda uma foto da tela em debug/ (no máximo DEBUG_MAX_FILES) para ver o que o bot enxergou."""
        try:
            frame = self.capture.get_frame()
            os.makedirs(self.debug_dir, exist_ok=True)
            if len([f for f in os.listdir(self.debug_dir) if f.startswith("skills_")]) >= self.DEBUG_MAX_FILES:
                return None
            path = os.path.join(self.debug_dir, f"skills_{label}_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return path
        except Exception:
            return None

    def check(self, skill: str, hotkey: str = "h") -> Optional[XpReading]:
        """XpReading da habilidade, ou None (o motivo fica em last_failure e no log)."""
        self.last_failure = self.last_failure_kind = ""
        cell = skill_cell(skill)
        if cell is None:
            self._fail(f"a habilidade '{skill}' não está mapeada na aba Skills")
            return None
        if not self.ready:
            self._fail("faltam os modelos da aba Skills em assets/skills")
            return None
        if self.focus and not self.focus():
            self._fail("a janela do jogo não está em foco")
            return None

        window = self._locate_panel(wait=False)
        opened_by_us = window is None
        mouse_before = self.mouse.position()
        moved = False
        reading = None
        try:
            if opened_by_us:
                self._sleep(humanize.human_delay(0.35, 0.45, 0.4, 3.0))
                self.keyboard.press(hotkey)
                self._wait(self.OPEN_WAIT)
                window = self._locate_panel(wait=True)
                if window is None:
                    saved = self._save_debug("nao_reconhecida")
                    self._fail(f"a aba Skills não foi reconhecida na tela depois da tecla '{hotkey}' "
                               f"(melhor nota {self.panel.last_score:.2f}, mínimo {ANCHOR_MATCH})"
                               + (f"; foto em {saved}" if saved else ""))
                    return None

            target = self._absolute(self.panel.cell_center(window, cell))
            if target is None:
                self._fail("janela do jogo não encontrada")
                return None
            dx, dy = humanize.gauss_offset(*self.HOVER_SIGMA, limit=2.2)
            hover = (round(target[0] + dx), round(target[1] + dy))
            self.log(f"[XP] Mouse no ícone de {skill}: {hover}")
            self.mouse.move_to(*hover)
            moved = True

            for attempt in range(self.NUMBER_TRIES):
                self._wait(self.HOVER_WAIT)
                reading = self.reader.read(self.capture.get_frame())
                if reading or (self.reader.last_failure_kind != "numbers" and attempt + 1 >= self.TOOLTIP_TRIES):
                    break
            if reading is None:
                self._fail(self.reader.last_failure or "tooltip ilegível", self.reader.last_failure_kind or "ui")
        finally:
            if reading:
                self._sleep(humanize.human_delay(0.5, 0.45, 0.4, 3.0))   # lê o número antes de afastar o mouse
            if window is not None and opened_by_us:
                self._close(hotkey)
            if moved:
                region = self.capture.window_region
                bounds = (region["left"] + 40, region["top"] + 80, region["left"] + region["width"] - 40,
                          region["top"] + region["height"] - 40) if region else None
                back = humanize.around(mouse_before, self.RETURN_SIGMA, bounds)
                self.mouse.move_to(*back)
                self.log(f"[XP] Mouse devolvido para perto de {mouse_before}: {back}")

        if reading:
            self.log(f"[XP] {skill}: nível {reading.level}, {format_number(reading.xp)} XP "
                     f"(faltam {format_number(reading.remainder)} para o nível {reading.level + 1}).")
        return reading

    def _close(self, hotkey: str) -> None:
        """
        Fecha a janela com a mesma tecla e espera ela sumir. NUNCA clica no X: o clique sairia do mouse
        parado no ícone e, se a janela já tivesse sumido, cairia no mundo do jogo e o personagem andaria até lá.
        """
        self.keyboard.press(hotkey)
        for _ in range(self.CLOSE_POLLS):
            self._wait(self.CLOSE_WAIT)
            if self._locate_panel(wait=False) is None:
                return
        self.log("[XP] Aviso: a aba Skills não fechou com a tecla; deixei aberta (o bot não clica para fechá-la).")
