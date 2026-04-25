import sys
import os
import time
import numpy as np
from transitions import Machine
from typing import Optional, List

# Add the project root to sys.path to allow absolute imports when running directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from vision.validators import MiningValidator
from vision.inventory import InventoryChecker


class MiningFSM:
    """
    Finite State Machine para o Loop de Mineração.
    Orquestra Visão YOLO, Visão OpenCV e Atuadores.
    """

    STATES = [
        "IDLE",
        "PROCURANDO_ROCHA",
        "MOVENDO_PARA_ROCHA",
        "MINERANDO",
        "INVENTARIO_CHEIO",
    ]

    def __init__(
        self,
        detector: any,
        mouse: any,
        keyboard: any,
        screencap: any,
        target_class: str = "Copper_rock",
    ):
        """
        Inicializa a FSM com as instâncias dos módulos de hardware e visão.
        """
        # Inicializa a máquina de estados
        self.machine = Machine(model=self, states=self.STATES, initial="IDLE")

        # Instâncias dos Módulos
        self.detector = detector
        self.mouse = mouse
        self.keyboard = keyboard
        self.screencap = screencap
        self.target_class = target_class

        # Validadores de Interface
        self.validator = MiningValidator()
        self.inventory = InventoryChecker(
            open_anchor_path="assets/mochila_aberta_ancora.png",
            empty_slot_path="assets/slot_vazio_mochila.png",
        )

        # Definição de Transições
        self.machine.add_transition("iniciar", "IDLE", "PROCURANDO_ROCHA")
        self.machine.add_transition(
            "rocha_encontrada", "PROCURANDO_ROCHA", "MOVENDO_PARA_ROCHA"
        )
        self.machine.add_transition(
            "movimento_concluido", "MOVENDO_PARA_ROCHA", "MINERANDO"
        )
        self.machine.add_transition("mineracao_parada", "MINERANDO", "PROCURANDO_ROCHA")
        self.machine.add_transition("inventario_lotado", "*", "INVENTARIO_CHEIO")
        self.machine.add_transition("reset", "*", "IDLE")

        # Controle de Tempo e Memória
        self.ultimo_check_mochila = 0
        self.target_coords_abs = None  # Coordenadas (x, y) absolutas na tela
        self.roi_mining = (1595, 62, 34, 36)

    # --- Callbacks de Estado (Lógica Ativa) ---

    def on_enter_PROCURANDO_ROCHA(self):
        """
        Executado ao entrar no estado de busca.
        Requer que o frame atual seja processado via YOLO.
        """
        # A lógica de detecção agora ocorre de forma reativa no update()
        # para garantir eficiência de frames.
        pass

    def on_enter_MOVENDO_PARA_ROCHA(self):
        """
        Executado quando um alvo é validado. Move o mouse e clica.
        """
        if self.target_coords_abs:
            x, y = self.target_coords_abs
            print(
                f"[{time.strftime('%X')}] Estado: MOVENDO_PARA_ROCHA - Clicando em ({x}, {y})"
            )

            # Executa movimento Bézier e Clique Humano
            self.mouse.human_click_at(x, y)

            # Limpa alvo e avança
            self.target_coords_abs = None
            self.movimento_concluido()
        else:
            print(
                f"[{time.strftime('%X')}] Erro: Coordenadas do alvo não encontradas. Resetando."
            )
            self.mineracao_parada()

    def on_enter_MINERANDO(self):
        print(
            f"[{time.strftime('%X')}] Estado: MINERANDO - Monitorando Pirâmide de Prioridade."
        )

    def on_enter_INVENTARIO_CHEIO(self):
        print(
            f"[{time.strftime('%X')}] Estado: INVENTARIO_CHEIO - Ciclo interrompido. Mochila cheia."
        )

    # --- Loop Principal (Update) ---

    def update(self, frame: np.ndarray):
        """
        Executa um 'tick' de lógica da FSM.
        Aplica a Pirâmide de Prioridades e Processamento YOLO.
        """
        current_state = self.state
        agora = time.time()

        # 1. Verificação Global: Segurança de Inventário e UI
        if current_state in ["IDLE", "PROCURANDO_ROCHA", "MINERANDO"]:
            # Garante que a mochila está aberta para validação (UI State Guarantee)
            if not self.inventory.is_open(frame):
                if agora - self.ultimo_check_mochila > 5.0:  # Evita spam da tecla 'B'
                    print(f"[{time.strftime('%X')}] [UI] Mochila fechada. Abrindo...")
                    self.keyboard.press("b")
                    self.ultimo_check_mochila = agora
                return  # Espera abrir no próximo frame

            # Se aberta, checa se está cheia
            if self.inventory.is_full(frame):
                if current_state != "INVENTARIO_CHEIO":
                    print(
                        f"[{time.strftime('%X')}] [Segurança] Inventário detectado como CHEIO."
                    )
                    self.inventario_lotado()
                return

        # 2. Lógica Específica por Estado
        if current_state == "MINERANDO":
            # [PRIORIDADE 1] Inventário (Timer de 2 segundos)
            if agora - self.ultimo_check_mochila > 2.0:
                self.ultimo_check_mochila = agora
                if self.inventory.is_full(frame):
                    print(
                        f"[{time.strftime('%X')}] [P1] Inventário encheu! Abortando mineração."
                    )
                    self.inventario_lotado()
                    return

            # [PRIORIDADE 2] Indicador de XP (Visual UI)
            # Buscamos no frame inteiro (que é apenas a janela do jogo) para maior estabilidade
            if not self.validator.esta_minerando(frame):
                print(
                    f"[{time.strftime('%X')}] [P2] Rocha esgotada ou indicador não visível. Buscando novamente..."
                )
                self.mineracao_parada()
                return

        elif current_state == "PROCURANDO_ROCHA":
            # Executa Detecção YOLO Real
            detections = self.detector.detect(frame, conf_threshold=0.5)
            targets = [d for d in detections if d["class"] == self.target_class]

            if targets:
                # Pega o alvo com melhor confiança
                best = sorted(targets, key=lambda x: x["conf"], reverse=True)[0]

                # Tradução de Coordenadas: Frame (Relativo) -> Tela (Absoluto)
                win = self.screencap.window_region
                if win:
                    self.target_coords_abs = (
                        best["x"] + win["left"],
                        best["y"] + win["top"],
                    )
                    print(
                        f"[{time.strftime('%X')}] Alvo encontrado: {self.target_class} ({best['conf']:.2f})"
                    )
                    self.rocha_encontrada()

        elif current_state == "IDLE":
            print(f"[{time.strftime('%X')}] Iniciando Motor de Estados...")
            self.iniciar()


if __name__ == "__main__":
    import cv2

    # Simulador de Teste da Fundação
    print("\n" + "=" * 50)
    print("RS3 VisionBot - SIMULADOR DE FSM (v2.0)")
    print("=" * 50)

    fsm = MiningFSM()
    # Criamos um frame fake (preto) para simular o processamento
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    try:
        while True:
            # 1. Executa o ciclo lógico
            fsm.update(dummy_frame)

            # 2. Simulador de transições para teste manual
            current = fsm.state

            if current == "PROCURANDO_ROCHA":
                print("... (Simulando 2s de busca YOLO)")
                time.sleep(2)
                fsm.rocha_encontrada()

            elif current == "MINERANDO":
                # No simulador, vamos minerar por um tempo ou até o usuário fechar
                # Aqui você verá os logs [P1], [P2] e [P3]
                pass

            elif current == "INVENTARIO_CHEIO":
                print("\n[FIM DE TESTE] O bot parou porque o inventário está cheio.")
                break

            time.sleep(0.5)  # Ritmo do loop do bot

    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo usuário.")
