import math
import time
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

    def setup_session(self, target_class: str, validator_name: str = None, prayer_image:str=None):
        """Atualiza o alvo e o validador para a nova sessão."""
        self.target_class = target_class

        if validator_name:
            # Reconstrói a instância para garantir que o novo template seja carregado
            self.validator = uiValidators.create_validator(validator_name)
            self.log(f"[SISTEMA] Validador configurado: {validator_name}")
        else:
            self.validator = None
            self.log("[SISTEMA] Rodando sem validador de XP.")

    def _maybe_reclick(self, frame: np.ndarray):
        """
        A barra de stamina da mineração sobe enquanto o personagem minera e cai quando se clica na
        rocha de novo. Ao passar do limite, reclica na rocha mais próxima do personagem.
        """
        now = time.time()

        # Diagnóstico: conferir se o reclique anterior fez a barra cair
        if self._reclick_check and now >= self._reclick_check[0]:
            before = self._reclick_check[1]
            self._reclick_check = None
            reading = self.stamina.read(frame)
            after = f"{reading[0]:.0%}" if reading else "sem barra"
            self.log(f"[STAMINA] Após o reclique: {before:.0%} -> {after}")

        if now - self.last_reclick < self.RECLICK_COOLDOWN:
            return
        reading = self.stamina.read(frame)
        if not reading or reading[0] < self.reclick_threshold:
            return

        fraction, (bar_x, bar_y) = reading
        self.last_reclick = now  # respeita o cooldown mesmo quando não houver rocha para clicar

        detections = self.detector.detect(frame, conf_threshold=0.5)
        wanted = self.target_class.lower()
        # Prefere a classe exata (ex.: "Iron_rock"); só aceita nomes parecidos se não houver nenhuma
        targets = [d for d in detections if d["class"].lower() == wanted] or                   [d for d in detections if wanted in d["class"].lower()]
        if not targets:
            self.log(f"[STAMINA] Barra em {fraction:.0%}, mas nenhuma rocha à vista.")
            return

        player = (bar_x, bar_y + 20)  # o personagem fica logo abaixo da barra
        nearest = min(targets, key=lambda d: math.hypot(d["x"] - player[0], d["y"] - player[1]))
        dist = math.hypot(nearest["x"] - player[0], nearest["y"] - player[1])
        if dist > self.RECLICK_MAX_DISTANCE:
            self.log(f"[STAMINA] Barra em {fraction:.0%}, mas a rocha mais próxima está longe ({dist:.0f}px).")
            return

        win = self.screencap.window_region
        if not win:
            return
        self.log(f"[STAMINA] Barra em {fraction:.0%} (limite {self.reclick_threshold:.0%}). "
                 f"Reclicando na rocha mais próxima ({dist:.0f}px).")
        self.mouse.human_click_at(nearest["x"] + win["left"], nearest["y"] + win["top"])
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

            targets = [
                d for d in detections if self.target_class.lower() in d["class"].lower()
            ]

            if targets:
                # Pega a detecção com maior confiança
                best = sorted(targets, key=lambda x: x["conf"], reverse=True)[0]

                # Traduz a coordenada da imagem para a coordenada do seu monitor
                win = self.screencap.window_region
                if win:
                    self.target_coords_abs = (
                        best["x"] + win["left"],
                        best["y"] + win["top"],
                    )
                    self.log(f"[VISÃO] Alvo encontrado em {self.target_coords_abs}")
                    self.alvo_encontrado()

        elif self.state == "MOVENDO_PARA_ALVO":
            if self.target_coords_abs:
                x, y = self.target_coords_abs
                self.log(f"[AÇÃO] Clicando em: {x}, {y}")
                self.mouse.human_click_at(x, y)
                self.target_coords_abs = None
                self.interaction_start = time.time()
                self.last_xp_time = time.time()
                self.clicou()
            else:
                self.voltar_busca()

        elif self.state == "INTERAGINDO":
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
