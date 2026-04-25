import tkinter as tk
import time
import threading  # A biblioteca que resolve o congelamento

from gui.app_window import BotInterface
from vision.capture import ScreenCapture
from vision.inventory import InventoryChecker
from vision.detector import Detector
from actuators.mouse import MouseActuator
from actuators.keyboard import KeyboardActuator
from engine.state_seeker import TargetSeekerFSM


class BotController:
    def __init__(self):
        self.is_running = False
        self.last_state = "IDLE"

        self.root = tk.Tk()
        self.gui = BotInterface(self.root, self.start, self.stop)

        self.gui.log("[SYSTEM] Carregando Sensores, IA e Atuadores...")

        self.capture = ScreenCapture(window_title="RuneScape")
        self.mouse = MouseActuator()
        self.keyboard = KeyboardActuator()

        self.detector = Detector(model_path=None)

        anchor = "assets/mochila_aberta_ancora.png"
        slot = "assets/slot_vazio_mochila.png"
        self.inventory_checker = InventoryChecker(
            open_anchor_path=anchor, empty_slot_path=slot
        )

        self.fsm = TargetSeekerFSM(
            inventory_checker=self.inventory_checker,
            detector=self.detector,
            mouse=self.mouse,
            keyboard=self.keyboard,
            screencap=self.capture,
            target_class=None,
            validator_path=None,
            logger_func=self.gui.log,
        )

        self.gui.log("[SYSTEM] Bot aguardando configuração do usuário.")

    def start(self):
        """Valida as escolhas da GUI e inicializa os recursos sob demanda."""
        if self.is_running:
            return

        # 1. Captura as seleções da GUI
        skill = self.gui.skill_var.get()
        target = self.gui.target_var.get()
        model_path = self.gui.model_path_var.get()
        validator_name = self.gui.xp_anchor_var.get()

        # 2. Lógica de Validação: Não inicia sem o mínimo necessário
        if not model_path or not target or not skill:
            self.gui.log(
                "[ERRO] Selecione uma Skill, um Alvo e um Modelo antes de iniciar!"
            )
            return

        try:
            self.gui.log(f"[SISTEMA] Carregando perfil: {skill} - {target}...")

            # Atualiza o modelo no detector apenas se ele for diferente do atual
            if not self.detector.update_model(model_path):
                raise RuntimeError("Falha ao carregar o arquivo do modelo (.pt)")

            # Passa as configurações para a FSM
            self.fsm.setup_session(
                target_class=target,
                validator_name=validator_name,
            )

            self.is_running = True
            self.bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
            self.bot_thread.start()
            self.gui.log("[ORQUESTRADOR] Motor em execução!")

        except Exception as e:
            self.gui.log(f"[ERRO CRÍTICO] Falha ao iniciar: {e}")

    def stop(self):
        """Apenas muda a flag. A Thread vai ler isso e parar sozinha."""
        self.is_running = False
        self.gui.log("\n[ORQUESTRADOR] Parada solicitada.")

    def _bot_loop(self):
        """Este loop agora roda solto no background, sem atrapalhar a interface."""
        while self.is_running:
            frame = self.capture.get_frame()

            if frame is not None:
                self.fsm.update(frame)

                current_state = self.fsm.state
                if current_state != self.last_state:
                    self.gui.log(f"-> Ação: {current_state}")
                    self.last_state = current_state

            # O bot processa a tela e dorme sem travar a interface
            time.sleep(0.3)

    def run(self):
        self.root.mainloop()


# ==========================================
# INICIALIZAÇÃO COM GRACEFUL SHUTDOWN
# ==========================================
if __name__ == "__main__":
    try:
        controller = BotController()
        controller.run()
    except KeyboardInterrupt:
        # Captura o Ctrl+C e encerra de forma limpa
        print("\n[Sistema] Encerrado pelo usuário.")
    except Exception as e:
        print(f"\n[Erro Fatal] Ocorreu uma exceção inesperada: {e}")
