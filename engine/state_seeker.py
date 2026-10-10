import math
import random
import time
from typing import List, Optional

import numpy as np
import os
from transitions import Machine
import vision.uiValidators as uiValidators


class TargetSeekerFSM:
    """
    Motor de Busca definitivo.
    Lê o inventário, procura com YOLO e clica com o Atuador.
    """

    RECLICK_COOLDOWN = 8.0        # s entre recliques
    RECLICK_MAX_DISTANCE = 220    # px: rocha mais próxima do personagem aceita para reclicar
    RECLICK_CHECK_DELAY = 2.5     # s depois do reclique para conferir se a barra caiu
    NO_BAR_RECLICK_AFTER = 25.0   # s sem enxergar a barra (com o XP chegando) até reclicar por segurança
    DEBUG_DIR = "debug"           # fotos da tela em situações estranhas, para diagnóstico
    DEBUG_MAX_FILES = 20
    PRIORITY_CHECK_INTERVAL = 2.5  # s entre buscas por uma classe prioritária enquanto minera
    PRIORITY_MIN_INTERACTION = 2.0  # s minerando antes de aceitar trocar para uma rocha prioritária

    STATES = [
        "IDLE",
        "PROCURANDO_ALVO",
        "MOVENDO_PARA_ALVO",
        "INTERAGINDO",
        "INVENTARIO_CHEIO",
        "DROPANDO",
    ]

    def __init__(
        self,
        inventory_checker,
        detector,
        mouse,
        keyboard,
        screencap,
        target_class=None,  # objeto alvo
        validator_path=None,  # indicador xp
        logger_func=None,
        dropper=None,  # InventoryDropper (opcional): esvazia a mochila quando enche
        stamina_reader=None,  # StaminaBarReader (opcional): lê a barra de stamina da mineração
    ):
        self.machine = Machine(model=self, states=self.STATES, initial="IDLE")

        # 1. Injeção das Dependências (Sensores e Atuadores)
        self.inventory = inventory_checker
        self.detector = detector
        self.mouse = mouse
        self.keyboard = keyboard
        self.screencap = screencap
        self.log = logger_func if logger_func else print
        self.target_class = target_class
        self.validator = None
        self.dropper = dropper
        self.stamina = stamina_reader
        self.reclick_enabled = False       # ligado pelo orquestrador conforme a tela
        self.reclick_threshold = 0.85      # fração da barra a partir da qual reclica na rocha
        self.last_reclick = 0.0
        self._reclick_check = None         # (instante, fração antes) p/ conferir se a barra caiu
        self.park_mouse_after_click = True  # afasta o mouse do personagem depois de clicar
        self._last_bar_seen = 0.0
        self.priority_classes: List[str] = []   # classes clicadas antes do alvo normal (a 1ª vale mais)
        self.current_class: Optional[str] = None  # classe da rocha que está sendo minerada
        self._last_priority_check = 0.0

        if validator_path:
            self.validator = uiValidators.create_validator(validator_path)

        # Variáveis de controle de ação
        self.target_coords_abs = None
        self.interaction_start = 0
        self.last_xp_time = 0
        self.last_inventory_status = None

        # 2. Definição do Fluxo (Transições)
        self.machine.add_transition("iniciar", "IDLE", "PROCURANDO_ALVO")
        self.machine.add_transition("encheu", "*", "INVENTARIO_CHEIO")
        self.machine.add_transition("esvaziou", "INVENTARIO_CHEIO", "PROCURANDO_ALVO")
        self.machine.add_transition("iniciar_drop", "INVENTARIO_CHEIO", "DROPANDO")
        self.machine.add_transition("concluir_drop", "DROPANDO", "PROCURANDO_ALVO")
        self.machine.add_transition("falhar_drop", "DROPANDO", "INVENTARIO_CHEIO")

        self.machine.add_transition(
            "alvo_encontrado", "PROCURANDO_ALVO", "MOVENDO_PARA_ALVO"
        )
        self.machine.add_transition("clicou", "MOVENDO_PARA_ALVO", "INTERAGINDO")
        self.machine.add_transition("voltar_busca", "INTERAGINDO", "PROCURANDO_ALVO")

        # ==========================================
        #    REDEFINIÇÃO DE PERFIL DE AÇÃO
        # ==========================================

    def setup_session(self, target_class: str, validator_name: str = None, prayer_image: str = None,
                      priority_classes: Optional[List[str]] = None):
        """Atualiza o alvo, as classes prioritárias e o validador para a nova sessão."""
        self.target_class = target_class
        self.priority_classes = [c for c in (priority_classes or []) if c]
        self.current_class = None
        if self.priority_classes:
            self.log(f"[PRIORIDADE] Classes prioritárias (nesta ordem): {', '.join(self.priority_classes)}")

        if validator_name:
            # Reconstrói a instância para garantir que o novo template seja carregado
            self.validator = uiValidators.create_validator(validator_name)
            self.log(f"[SISTEMA] Validador configurado: {validator_name}")
        else:
            self.validator = None
            self.log("[SISTEMA] Rodando sem validador de XP.")

    def _pick_target(self, detections: list):
        """
        Escolhe a detecção a clicar: primeiro a classe prioritária (na ordem da lista, a de maior confiança
        entre as dela); se não houver nenhuma, o alvo normal (maior confiança). None se não houver nada.
        """
        for cls in self.priority_classes:
            found = [d for d in detections if d["class"].lower() == cls.lower()]
            if found:
                return max(found, key=lambda d: d["conf"])
        targets = [d for d in detections if self.target_class.lower() in d["class"].lower()]
        return max(targets, key=lambda d: d["conf"]) if targets else None

    def _priority_appeared(self, frame: np.ndarray) -> Optional[str]:
        """
        Enquanto minera uma rocha comum, procura de tempos em tempos uma classe prioritária na tela.
        Retorna o nome da classe encontrada (para trocar de rocha) ou None.
        """
        now = time.time()
        names = {c.lower() for c in self.priority_classes}
        if (self.current_class or "").lower() in names:
            return None  # já está numa rocha prioritária
        if now - self.interaction_start < self.PRIORITY_MIN_INTERACTION:
            return None
        if now - self._last_priority_check < self.PRIORITY_CHECK_INTERVAL:
            return None
        self._last_priority_check = now
        hits = [d for d in self.detector.detect(frame, conf_threshold=0.5) if d["class"].lower() in names]
        return max(hits, key=lambda d: d["conf"])["class"] if hits else None

    def _park_mouse(self, frame: np.ndarray):
        """
        Afasta o mouse do personagem e das rochas logo depois de um clique. Com o mouse parado em cima de
        uma rocha perto do personagem, o destaque/tooltip "Mine ..." fica por cima da área da barra de
        stamina e atrapalha a leitura. O ponto escolhido fica na vista 3D, longe do personagem, fora do painel.
        """
        if not self.park_mouse_after_click:
            return
        win = self.screencap.window_region
        if not win:
            return
        h, w = frame.shape[:2]
        right_limit = int(w * 0.72)
        get_roi = getattr(self.inventory, "get_slots_roi", None)
        if get_roi:
            try:
                roi = get_roi(frame)
                if roi:
                    right_limit = roi[0] - 40  # borda esquerda do painel da mochila
            except Exception:
                pass
        cx, cy = w // 2, h // 2
        x_lo, x_hi = cx + 180, min(cx + 340, right_limit)
        if x_hi <= x_lo:  # sem espaço à direita do personagem: usa a esquerda
            x_lo, x_hi = max(40, cx - 340), max(41, cx - 180)
        y_lo, y_hi = max(70, cy - 300), max(71, cy - 120)
        time.sleep(random.uniform(0.25, 0.5))  # deixa o clique ser registrado antes de mover o mouse
        self.mouse.move_to(random.randint(x_lo, x_hi) + win["left"], random.randint(y_lo, y_hi) + win["top"])

    def _save_debug(self, frame: np.ndarray, label: str) -> Optional[str]:
        """Guarda uma foto da tela em debug/ (no máximo DEBUG_MAX_FILES) para diagnosticar situações estranhas."""
        try:
            import cv2
            os.makedirs(self.DEBUG_DIR, exist_ok=True)
            if len([f for f in os.listdir(self.DEBUG_DIR) if f.endswith(".png")]) >= self.DEBUG_MAX_FILES:
                return None
            path = os.path.join(self.DEBUG_DIR, f"{label}_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(path, frame)
            return path
        except Exception:
            return None

    def _maybe_reclick(self, frame: np.ndarray):
        """
        A barra de stamina da mineração sobe enquanto o personagem minera e cai quando se clica na rocha de
        novo. Ao passar do limite, reclica na rocha mais próxima do personagem. Se a barra ficar muito tempo
        sem aparecer (o XP continua chegando), reclica por segurança, sem depender dela.
        """
        now = time.time()
        reading = self.stamina.read(frame)
        if reading:
            self._last_bar_seen = now

        # Diagnóstico: conferir se o reclique anterior fez a barra cair
        if self._reclick_check and now >= self._reclick_check[0]:
            before = self._reclick_check[1]
            self._reclick_check = None
            after = f"{reading[0]:.0%}" if reading else "sem barra"
            self.log(f"[STAMINA] Após o reclique: {f'{before:.0%}' if before is not None else 'sem leitura'} -> {after}")

        if now - self.last_reclick < self.RECLICK_COOLDOWN:
            return

        blind_for = None
        if reading is None:
            reference = max(self._last_bar_seen, self.interaction_start, self.last_reclick)
            if now - reference < self.NO_BAR_RECLICK_AFTER:
                return
            blind_for = now - reference
        elif reading[0] < self.reclick_threshold:
            return

        self.last_reclick = now  # respeita o cooldown mesmo quando não houver rocha para clicar
        h, w = frame.shape[:2]
        if reading:
            fraction, (bar_x, bar_y) = reading
            player = (bar_x, bar_y + 20)  # o personagem fica logo abaixo da barra
            why = f"Barra em {fraction:.0%} (limite {self.reclick_threshold:.0%})"
        else:
            fraction = None
            player = (w // 2, h // 2 + 20)  # sem barra: o personagem fica no centro da janela
            why = f"Barra não visível há {blind_for:.0f}s"

        detections = self.detector.detect(frame, conf_threshold=0.5)
        wanted = (self.current_class or self.target_class).lower()
        # Prefere a classe exata (ex.: "Iron_rock"); só aceita nomes parecidos se não houver nenhuma
        targets = ([d for d in detections if d["class"].lower() == wanted]
                   or [d for d in detections if wanted in d["class"].lower()])
        if not targets:
            self.log(f"[STAMINA] {why}, mas nenhuma rocha à vista.")
            return

        nearest = min(targets, key=lambda d: math.hypot(d["x"] - player[0], d["y"] - player[1]))
        dist = math.hypot(nearest["x"] - player[0], nearest["y"] - player[1])
        if dist > self.RECLICK_MAX_DISTANCE:
            self.log(f"[STAMINA] {why}, mas a rocha mais próxima está longe ({dist:.0f}px).")
            return

        win = self.screencap.window_region
        if not win:
            return
        self.log(f"[STAMINA] {why}. Reclicando na rocha mais próxima ({dist:.0f}px).")
        if fraction is None:
            saved = self._save_debug(frame, "sem_barra")
            if saved:
                self.log(f"[STAMINA] Foto da tela salva em {saved} (para diagnóstico).")
        self.mouse.human_click_at(nearest["x"] + win["left"], nearest["y"] + win["top"])
        self._park_mouse(frame)
        self.last_reclick = time.time()
        self.last_xp_time = self.last_reclick  # o reclique pode atrasar um pouco o XP
        self._reclick_check = (self.last_reclick + self.RECLICK_CHECK_DELAY, fraction)

    def _drop_step(self, frame: np.ndarray):
        result = self.dropper.step(frame)
        if result == self.dropper.DONE:
            self.log("[DROP] Mochila esvaziada. Retomando...")
            self.concluir_drop()
        elif result == self.dropper.FAILED:
            self.log("[DROP] Falhou. Aguardando esvaziamento manual.")
            self.falhar_drop()

    def update(self, frame: np.ndarray):
        if self.state == "IDLE":
            self.iniciar()
            return
        # ==========================================
        # PASSO 1: VALIDAÇÃO DA MOCHILA (UI)
        # ==========================================
        is_inventory_open = self.inventory.is_open(frame)
        if is_inventory_open != self.last_inventory_status:
            status_msg = (
                "[VISÃO] Mochila Detectada ✅"
                if is_inventory_open
                else "[VISÃO] Mochila NÃO encontrada ❌"
            )
            self.log(status_msg)
            self.last_inventory_status = is_inventory_open

        if not is_inventory_open:
            return

        # Dropando: segue até não restar item, mesmo que já existam slots vazios
        if self.state == "DROPANDO":
            self._drop_step(frame)
            return

        # Verificação de inventário cheio
        if self.inventory.is_full(frame):
            if self.state != "INVENTARIO_CHEIO":
                self.encheu()
                if self.dropper and self.dropper.enabled:
                    self.log("[ESTADO] Inventário Cheio! Iniciando drop...")
                    self.dropper.start()
                    self.iniciar_drop()
                else:
                    self.log("[ESTADO] Inventário Cheio! Aguardando esvaziamento...")
            return
        elif self.state == "INVENTARIO_CHEIO":
            self.log("[ESTADO] Inventário liberado. Retomando...")
            self.esvaziou()

        # ==========================================
        # PRIORIDADE 2: CAÇA E AÇÃO
        # ==========================================
        if self.state == "PROCURANDO_ALVO":
            # Chama o YOLO
            detections = self.detector.detect(frame, conf_threshold=0.5)

            # Classe prioritária (ex.: rocha "buff") tem preferência; senão, a de maior confiança do alvo
            best = self._pick_target(detections)

            if best:
                self.current_class = best["class"]

                # Traduz a coordenada da imagem para a coordenada do seu monitor
                win = self.screencap.window_region
                if win:
                    self.target_coords_abs = (
                        best["x"] + win["left"],
                        best["y"] + win["top"],
                    )
                    self.log(f"[VISÃO] Alvo {best['class']} encontrado em {self.target_coords_abs}")
                    self.alvo_encontrado()

        elif self.state == "MOVENDO_PARA_ALVO":
            if self.target_coords_abs:
                x, y = self.target_coords_abs
                self.log(f"[AÇÃO] Clicando em: {x}, {y}")
                self.mouse.human_click_at(x, y)
                self._park_mouse(frame)
                self.target_coords_abs = None
                self.interaction_start = time.time()
                self.last_xp_time = time.time()
                self.clicou()
            else:
                self.voltar_busca()

        elif self.state == "INTERAGINDO":
            if self.priority_classes:
                appeared = self._priority_appeared(frame)
                if appeared:
                    self.log(f"[PRIORIDADE] {appeared} apareceu na tela. Trocando de rocha.")
                    self.voltar_busca()
                    return

            if self.reclick_enabled and self.stamina:
                self._maybe_reclick(frame)

            if self.validator and self.validator.template is not None:
                if self.validator.is_visible(frame, threshold=0.7):
                    self.last_xp_time = time.time()

            tempo_sem_xp = time.time() - self.last_xp_time
            tempo_total_acao = time.time() - self.interaction_start

            if tempo_sem_xp > 2.2 and tempo_total_acao > 3.5:
                score = f" (indicador: {self.validator.last_score:.2f})" if self.validator else ""
                self.log(f"[ESTADO] XP parou{score}. Buscando nova rocha...")
                self.voltar_busca()
        else:
            if time.time() - self.interaction_start > 5.0:
                self.log("[ESTADO] Timeout (Sem validador). Reiniciando busca.")
                self.voltar_busca()
