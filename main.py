import tkinter as tk
import time
import threading

from gui.app_window import BotInterface
from vision.capture import ScreenCapture
from vision.inventory import InventoryChecker
from vision.detector import Detector
from actuators.mouse import MouseActuator
from actuators.keyboard import KeyboardActuator
from engine.state_seeker import TargetSeekerFSM
from engine.survival_manager import SurvivalManager

class BotController:
    def __init__(self):
        # 1. Configurações de Estado
        self.is_running = False
        self.last_state = "IDLE"

        # 2. Inicialização da GUI
        self.root = tk.Tk()
        self.gui = BotInterface(self.root, self.start, self.stop)
        self.gui.log("[SYSTEM] Inicializando Sensores e Atuadores...")

        # 3. Instanciação dos Módulos (Core)
        self._initialize_core_modules()
        
        self.gui.log("[SYSTEM] Bot pronto para configuração.")

    def _initialize_core_modules(self):
        """Instancia os componentes que não mudam durante a execução."""
        self.capture = ScreenCapture(window_title="RuneScape")
        self.mouse = MouseActuator()
        self.keyboard = KeyboardActuator()
        self.detector = Detector(model_path=None)
        
        # Gerenciador de Sobrevivência Modular
        self.survival_manager = SurvivalManager(self.keyboard, self.gui.log)

        # Configuração do Inventário
        self.inventory_checker = InventoryChecker(
            open_anchor_path="assets/mochila_aberta_ancora.png", 
            empty_slot_path="assets/slot_vazio_mochila.png"
        )

        # Máquina de Estados (FSM)
        self.fsm = TargetSeekerFSM(
            inventory_checker=self.inventory_checker,
            detector=self.detector,
            mouse=self.mouse,
            keyboard=self.keyboard,
            screencap=self.capture,
            logger_func=self.gui.log
        )

    def _configure_session(self):
        """Lê os dados da GUI e configura os serviços para o 'Play'."""
        # 1. Captura estados da GUI (Desacoplados)
        use_pot = self.gui.use_potion_var.get()
        use_pray = self.gui.use_prayer_var.get()
        use_mask = self.gui.use_mask_var.get()
        use_ia = self.gui.use_yolo_var.get()
        
        # 2. Captura chaves e caminhos
        pot_key = self.gui.potion_key_var.get().strip()
        pray_key = self.gui.prayer_key_var.get().strip()
        mask_key = self.gui.mask_key_var.get().strip()

        skill = self.gui.skill_var.get()
        target = self.gui.target_var.get()
        model_path = self.gui.model_path_var.get()

        # Passamos cada estado individualmente para o manager
        self.survival_manager.configure_all(
            pot_enabled=use_pot,
            pot_key=pot_key,
            pray_enabled=use_pray,
            pray_key=pray_key,
            mask_enabled=use_mask,
            mask_key=mask_key
        )
        # Configura IA (YOLO)
        has_ia_active = use_ia and all([model_path, target, skill])
        if has_ia_active:
            if not self.detector.update_model(model_path):
                raise RuntimeError("Erro ao carregar modelo .pt")
            self.fsm.setup_session(target_class=target, validator_name=self.gui.xp_anchor_var.get())
        else:
            self.fsm.target_class = None
            if not use_ia:
                self.gui.log("[MODO] IA Desativada pelo usuário.")
            self.gui.log("[MODO] Serviços de suporte ativos conforme seleção.")

        return has_ia_active

    def start(self):
        """Valida e inicia a thread do motor."""
        if self.is_running: return

        try:
            has_ia = self._configure_session()
            
            self.is_running = True
            self.bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
            self.bot_thread.start()
            
            self.gui.log(f"[MOTOR] Executando! (IA: {'Ligada' if has_ia else 'Desligada'})")
        except Exception as e:
            self.gui.log(f"[ERRO] Falha ao iniciar: {e}")

    def stop(self):
        self.is_running = False
        self.gui.log("\n[ORQUESTRADOR] Parada solicitada.")

    def _bot_loop(self):
        """Loop principal de processamento."""
        while self.is_running:
            frame = self.capture.get_frame()
            if frame is None: 
                self.gui.log("[ERRO VISÃO] Janela do jogo não detectada!")
                time.sleep(2)
                continue

            # 1. Prioridade: Sobrevivência (Sempre roda se configurado)
            self.survival_manager.update_all(frame)

            # 2. Lógica de Atividade (FSM) - Só roda se houver alvo
            if self.fsm.target_class:
                self.fsm.update(frame)
                self._update_state_log()

            time.sleep(0.1)

    def _update_state_log(self):
        """Monitora e loga mudanças de estado da FSM."""
        current_state = self.fsm.state
        if current_state != self.last_state:
            self.gui.log(f"-> Estado: {current_state}")
            self.last_state = current_state

   

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        controller = BotController()
        controller.run()
    except Exception as e:
        print(f"\n[Erro Fatal] {e}")