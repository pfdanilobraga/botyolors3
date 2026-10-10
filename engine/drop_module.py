import os
import random
import time

import cv2
import numpy as np


class InventoryDropper:
    """
    Esvazia da mochila todos os slots que contêm o item configurado (reconhecido por template).

    Dois métodos: tecla de atalho do slot da barra de ações configurado para dropar
    (cada toque dropa 1 item) e shift-click em cada slot. A cada rodada o método é sorteado
    e mantido até o fim; só troca para o outro se o escolhido não tirar nada da mochila.
    Trabalha em passos de 1 ação por chamada de step(), para o loop do bot continuar
    rodando a sobrevivência entre um drop e outro.
    """

    WORKING = "working"
    DONE = "done"
    FAILED = "failed"

    MATCH_THRESHOLD = 0.85
    MIN_SLOT_DISTANCE = 15      # px entre dois picos para contarem como slots diferentes
    RECHECK_DELAY = 1.2         # o jogo processa o drop no próximo tick (0,6s): espera antes de reclicar
    MAX_ATTEMPTS_PER_SLOT = 3
    MIN_CLICKS_BEFORE_ABORT = 5  # se após N cliques nada sumiu, o shift-click não está dropando
    MIN_PRESSES_BEFORE_SWITCH = 4    # se após N toques (e o tempo abaixo) nada sumiu, a tecla não dropa
    MIN_SECONDS_BEFORE_SWITCH = 2.5
    PRESS_INTERVAL = (0.25, 0.40)    # entre o início de um toque e o próximo (sorteado a cada toque)
    PRESS_HOLD = (0.04, 0.09)        # tempo com a tecla pressionada
    PRESS_SETTLE = (0.02, 0.06)      # pausa após soltar a tecla
    MAX_DURATION = 90.0

    def __init__(self, inventory_checker, mouse, keyboard, screencap, logger_func, focus_callback=None):
        self.inventory = inventory_checker
        self.mouse = mouse
        self.keyboard = keyboard
        self.screencap = screencap
        self.log = logger_func
        self.focus_callback = focus_callback

        self.enabled = False
        self.template = None
        self.hotkey = None
        self._reset()

    def configure(self, enabled: bool, item_image: str = None, hotkey: str = None):
        """
        item_image: nome do arquivo em assets/ (ex.: 'copper_ore.png').
        hotkey: tecla do slot da barra de ações que dropa o item (ex.: '-'). Sem ela, usa shift-click.
        """
        self.enabled = False
        self.template = None
        self.hotkey = hotkey or None
        if not enabled:
            return
        if not item_image:
            self.log("[DROP] Nenhum item de drop definido para este alvo. Auto-drop desativado.")
            return

        path = os.path.join("assets", item_image)
        template = cv2.imread(path)
        if template is None:
            self.log(f"[DROP] Template não encontrado: {path}. Auto-drop desativado.")
            return

        self.template = template
        self.enabled = True
        method = f"tecla '{self.hotkey}' (barra de ações)" if self.hotkey else "shift-click"
        self.log(f"[DROP] Auto-drop ativo para: {item_image} via {method}")

    def start(self):
        """Nova rodada de drop: zera o controle e sorteia o método (tecla ou shift-click)."""
        self._reset()
        self._mode = random.choice(["hotkey", "shift"]) if self.hotkey else "shift"
        self._tried = {self._mode}
        self.log(f"[DROP] Método desta rodada: {self._method_name(self._mode)}")

    def _method_name(self, mode: str) -> str:
        return f"tecla '{self.hotkey}'" if mode == "hotkey" else "shift-click"

    def _reset(self):
        self._mode = "hotkey" if self.hotkey else "shift"
        self._tried = {self._mode}
        self._attempts = {}   # (x, y) do slot -> (tentativas, instante do último clique)
        self._clicks = 0
        self._presses = 0
        self._press_started = None
        self._next_press = 0.0
        self._initial_count = None
        self._started = time.time()

    def _switch_method(self, slots_count: int) -> bool:
        """Troca para o outro método se o atual não tirou nada da mochila. False se já testou os dois."""
        other = "shift" if self._mode == "hotkey" else "hotkey"
        if other == "hotkey" and not self.hotkey:
            return False
        if other in self._tried:
            return False

        self.log(f"[DROP] {self._method_name(self._mode)} não dropou nada. Trocando para {self._method_name(other)}...")
        self._mode = other
        self._tried.add(other)
        self._initial_count = slots_count
        self._attempts = {}
        self._clicks = 0
        self._presses = 0
        self._press_started = None
        return True

    def find_slots(self, frame: np.ndarray):
        """
        Centros (x, y), em coordenadas do frame, dos slots da mochila com o item,
        em ordem de leitura (linha a linha). None se a mochila não estiver visível.
        """
        if self.template is None:
            return None
        return self.inventory.find_item_slots(frame, self.template, self.MATCH_THRESHOLD, self.MIN_SLOT_DISTANCE)

    def step(self, frame: np.ndarray) -> str:
        """Executa no máximo um clique. Retorna WORKING, DONE ou FAILED."""
        now = time.time()
        if now - self._started > self.MAX_DURATION:
            self.log("[DROP] Tempo limite excedido.")
            return self.FAILED

        slots = self.find_slots(frame)
        if slots is None:
            return self.WORKING  # mochila não visível agora; o limite de tempo cobre o caso
        if not slots:
            return self.DONE

        if self._initial_count is None:
            self._initial_count = len(slots)
            self.log(f"[DROP] {len(slots)} item(ns) para dropar.")

        if self._mode == "hotkey":
            return self._step_hotkey(slots, now)
        return self._step_shift(slots, now)

    def _step_hotkey(self, slots, now) -> str:
        """Um toque na tecla do slot de drop da barra de ações."""
        stalled = (
            self._presses >= self.MIN_PRESSES_BEFORE_SWITCH
            and now - self._press_started >= self.MIN_SECONDS_BEFORE_SWITCH
            and len(slots) >= self._initial_count
        )
        if stalled:
            if self._switch_method(len(slots)):
                return self.WORKING
            return self._fail_stalled()

        if now >= self._next_press:
            if self._press_started is None:
                self._press_started = now
            self.keyboard.press(
                self.hotkey,
                duration=random.uniform(*self.PRESS_HOLD),
                settle=self.PRESS_SETTLE,
            )
            self._presses += 1
            self._next_press = now + random.uniform(*self.PRESS_INTERVAL)
        return self.WORKING

    def _fail_stalled(self) -> str:
        self.log("[DROP] Os itens não saíram da mochila. Confira o slot de drop da barra de ações "
                 "e o shift-click para dropar no jogo.")
        return self.FAILED

    def _step_shift(self, slots, now) -> str:
        """Um shift-click em um slot ainda ocupado."""
        # Nada diminuiu mesmo após vários cliques: o shift-click não está funcionando
        if self._clicks >= self.MIN_CLICKS_BEFORE_ABORT and len(slots) >= self._initial_count:
            if self._switch_method(len(slots)):
                return self.WORKING
            return self._fail_stalled()

        pending, waiting = [], False
        for pos in slots:
            key = self._slot_key(pos)
            attempts, last_click = self._attempts.get(key, (0, 0.0))
            if attempts >= self.MAX_ATTEMPTS_PER_SLOT:
                continue
            if now - last_click < self.RECHECK_DELAY:
                waiting = True
                continue
            pending.append((pos, key, attempts))

        if not pending:
            if waiting:
                return self.WORKING  # aguardando o jogo processar o último drop
            self.log("[DROP] Alguns itens não puderam ser dropados.")
            return self.FAILED

        pos, key, attempts = pending[0]
        if self._drop_slot(pos):
            self._attempts[key] = (attempts + 1, time.time())
            self._clicks += 1
        return self.WORKING

    def _slot_key(self, pos):
        """Reaproveita a chave de um slot já clicado (tolerância a 1-2 px de oscilação do match)."""
        for key in self._attempts:
            if abs(key[0] - pos[0]) <= 12 and abs(key[1] - pos[1]) <= 12:
                return key
        return pos

    def _drop_slot(self, pos) -> bool:
        """Shift-click no slot. Garante que o shift seja sempre solto."""
        region = self.screencap.window_region
        if not region:
            return False
        if self.focus_callback and not self.focus_callback():
            return False

        abs_x, abs_y = pos[0] + region["left"], pos[1] + region["top"]
        self.keyboard.key_down("shift")
        try:
            time.sleep(random.uniform(0.04, 0.1))
            self.mouse.human_click_at(abs_x, abs_y)
        finally:
            self.keyboard.key_up("shift")
        return True
