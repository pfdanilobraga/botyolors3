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

    STATES = [
        "IDLE",
        "PROCURANDO_ALVO",
        "MOVENDO_PARA_ALVO",
        "INTERAGINDO",
        "INVENTARIO_CHEIO",
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

        if validator_path:
            self.validator = uiValidators.UIValidator(validator_path)

        # Variáveis de controle de ação
        self.target_coords_abs = None
        self.interaction_start = 0
        self.last_xp_time = 0
        self.last_inventory_status = None

        # 2. Definição do Fluxo (Transições)
        self.machine.add_transition("iniciar", "IDLE", "PROCURANDO_ALVO")
        self.machine.add_transition("encheu", "*", "INVENTARIO_CHEIO")
        self.machine.add_transition("esvaziou", "INVENTARIO_CHEIO", "PROCURANDO_ALVO")

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
            self.validator = uiValidators.UIValidator(validator_name)
            self.log(f"[SISTEMA] Validador configurado: {validator_name}")
        else:
            self.validator = None
            self.log("[SISTEMA] Rodando sem validador de XP.")

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

        # Verificação de inventário cheio
        if self.inventory.is_full(frame):
            if self.state != "INVENTARIO_CHEIO":
                self.log("[ESTADO] Inventário Cheio! Aguardando esvaziamento...")
                self.encheu()
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
            if self.validator and self.validator.template is not None:
                if self.validator.is_visible(frame, threshold=0.7):
                    self.last_xp_time = time.time()

            tempo_sem_xp = time.time() - self.last_xp_time
            tempo_total_acao = time.time() - self.interaction_start

            if tempo_sem_xp > 2.2 and tempo_total_acao > 3.5:
                self.log("[ESTADO] XP parou. Buscando nova rocha...")
                self.voltar_busca()
        else:
            if time.time() - self.interaction_start > 5.0:
                self.log("[ESTADO] Timeout (Sem validador). Reiniciando busca.")
                self.voltar_busca()
