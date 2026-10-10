import os
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
from engine.drop_module import InventoryDropper
from engine.session import ItemGainTracker, SessionMonitor, format_duration, format_number
from engine.skill_goals import ProgressStore, SkillGoal, xp_for_level
from vision.stamina import StaminaBarReader

class BotController:
    def __init__(self):
        # 1. Configurações de Estado
        self.is_running = False
        self.last_state = "IDLE"
        self._goal_skill = ""
        self._goal_target = ""

        # 2. Inicialização da GUI
        self.root = tk.Tk()
        self.gui = BotInterface(self.root, self.start, self.stop)
        self.gui.log("[SYSTEM] Inicializando Sensores e Atuadores...")

        # 3. Instanciação dos Módulos (Core)
        self._initialize_core_modules()

        # 4. Sessão: tempo limite, XP/nível estimados e meta
        self.session = SessionMonitor()
        self.progress = ProgressStore()
        self.gain_tracker = ItemGainTracker(self.inventory_checker)
        self.gui.bind_session(self.session.snapshot, self.sync_progress)
        
        self.gui.log("[SYSTEM] Bot pronto para configuração.")

    def _initialize_core_modules(self):
        """Instancia os componentes que não mudam durante a execução."""
        self.capture = ScreenCapture(window_title="RuneScape")
        # Os atuadores garantem o foco na janela do jogo antes de cada ação
        self.mouse = MouseActuator(focus_callback=self._ensure_game_focus)
        self.keyboard = KeyboardActuator(focus_callback=self._ensure_game_focus)
        self.detector = Detector(model_path=None)
        
        # Gerenciador de Sobrevivência Modular
        self.survival_manager = SurvivalManager(self.keyboard, self.gui.log)

        # Configuração do Inventário
        self.inventory_checker = InventoryChecker(
            open_anchor_path="assets/mochila_aberta_ancora.png", 
            empty_slot_path="assets/slot_vazio_mochila.png"
        )

        # Esvaziamento da mochila (shift-click) quando enche
        self.dropper = InventoryDropper(
            inventory_checker=self.inventory_checker,
            mouse=self.mouse,
            keyboard=self.keyboard,
            screencap=self.capture,
            logger_func=self.gui.log,
            focus_callback=self._ensure_game_focus,
        )

        # Máquina de Estados (FSM)
        self.fsm = TargetSeekerFSM(
            inventory_checker=self.inventory_checker,
            detector=self.detector,
            mouse=self.mouse,
            keyboard=self.keyboard,
            screencap=self.capture,
            logger_func=self.gui.log,
            dropper=self.dropper,
            stamina_reader=StaminaBarReader()
        )

    def _ensure_game_focus(self) -> bool:
        """Traz o RuneScape para o primeiro plano; loga só quando precisou agir."""
        if self.capture.is_focused():
            return True
        if self.capture.focus():
            self.gui.log("[FOCO] Janela do jogo trazida para o primeiro plano.")
            return True
        self.gui.log("[FOCO] Não foi possível focar a janela do jogo!")
        return False

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
        self.dropper.configure(
            enabled=has_ia_active and self.gui.use_drop_var.get(),
            item_image=self.gui.drop_item_var.get().strip(),
            hotkey=self.gui.drop_key_var.get().strip(),
        )
        if has_ia_active:
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Modelo não encontrado: {model_path}. Treine/adicione o .pt para este alvo "
                    "ou desmarque a IA."
                )
            if not self.detector.update_model(model_path):
                raise RuntimeError("Erro ao carregar modelo .pt")
            priority = [c.strip() for c in self.gui.priority_var.get().split(",") if c.strip()]
            self.fsm.setup_session(target_class=target, validator_name=self.gui.xp_anchor_var.get(),
                                   priority_classes=priority)
        else:
            self.fsm.target_class = None
            self.fsm.priority_classes = []
            if not use_ia:
                self.gui.log("[MODO] IA Desativada pelo usuário.")
            self.gui.log("[MODO] Serviços de suporte ativos conforme seleção.")

        # Reclique na rocha conforme a barra de stamina da mineração
        try:
            pct = min(max(int(self.gui.reclick_pct_var.get().strip().rstrip("%")), 30), 100)
        except ValueError:
            pct = 85
        self.fsm.reclick_enabled = bool(has_ia_active and self.gui.use_reclick_var.get())
        self.fsm.reclick_threshold = pct / 100
        if self.fsm.reclick_enabled:
            self.gui.log(f"[STAMINA] Reclique ativo: reclica na rocha quando a barra passar de {pct}%.")

        self._configure_goals(has_ia_active)
        return has_ia_active

    def _configure_goals(self, has_ia_active: bool):
        """Tempo limite + acompanhamento de nível/meta, a partir do painel Sessão & Metas."""
        settings = self.gui.session_panel.read()
        skill, target = self.gui.skill_var.get(), self.gui.target_var.get()
        self._goal_skill, self._goal_target = skill, target

        goal = None
        if settings.level is not None:
            start_xp = settings.xp if settings.xp is not None else xp_for_level(settings.level)
            goal = SkillGoal(skill, start_xp, settings.xp_per_item, settings.target_level)
            if settings.xp_per_item > 0:
                self.progress.save_xp_per_item(skill, target, settings.xp_per_item)

        self.session.configure(settings.duration_s, goal, settings.stop_on_goal)

        # Os itens são contados pelo ícone do minério (o mesmo do Auto-drop), mesmo com o drop desligado
        counting = False
        if goal and has_ia_active:
            counting = self.gain_tracker.configure(self.gui.drop_item_var.get().strip())
        else:
            self.gain_tracker.configure(None)

        if settings.duration_s > 0:
            self.gui.log(f"[SESSÃO] Tempo de bot: {format_duration(settings.duration_s)}.")
        if goal:
            meta = f", meta: nível {goal.target_level}" if goal.target_level else ""
            self.gui.log(f"[SESSÃO] {skill}: nível {goal.level} ({goal.xp:,.0f} XP){meta}.".replace(",", "."))
            if not counting:
                self.gui.log("[SESSÃO] Sem ícone do item (ou IA desligada): o XP não será estimado sozinho. "
                             "Use 'Corrigir nível/XP agora' para atualizar à mão.")
            elif goal.xp_per_item <= 0:
                self.gui.log("[SESSÃO] Informe o 'XP por item' para o bot estimar o XP ganho.")

    def start(self):
        """Valida e inicia a thread do motor."""
        if self.is_running: return

        try:
            has_ia = self._configure_session()

            if not self._ensure_game_focus():
                raise RuntimeError("Janela do RuneScape não encontrada ou não pôde ser focada.")

            self.is_running = True
            self.session.start()
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
        try:
            while self.is_running:
                # Tempo limite e meta valem mesmo se a janela do jogo sumir
                self._handle_session_events(self.session.update())
                if not self.is_running:
                    break

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

                    gained = self.gain_tracker.update(frame, self.fsm.state)
                    if gained:
                        self.session.add_items(gained)

                # Durante o drop o ciclo é mais curto para os toques saírem no ritmo configurado
                time.sleep(0.03 if self.fsm.state == "DROPANDO" else 0.1)
        finally:
            if threading.current_thread() is getattr(self, "bot_thread", None):
                self.is_running = False  # laço terminou (parada, meta, tempo ou erro): libera o Start
            self._finish_session()

    def _handle_session_events(self, events):
        """Registra e avisa (som/popup) subidas de nível, meta atingida e fim do tempo; para o bot se pedido."""
        for event in events:
            self.gui.log(f"[SESSÃO] {event.message}")
            self.gui.post_event(event)
            if event.stop:
                self.is_running = False

    def _finish_session(self):
        """Salva o nível/XP estimados (a próxima sessão continua daí) e registra o resumo."""
        goal = self.session.goal
        if goal:
            try:
                self.progress.save_skill(self._goal_skill, goal.xp, goal.target_level)
            except OSError as e:
                self.gui.log(f"[SESSÃO] Não foi possível salvar o progresso: {e}")
        summary = self.session.finish()
        if summary:
            self.gui.log(f"[SESSÃO] {summary}")

    def sync_progress(self):
        """Botão 'Corrigir nível/XP agora': aplica o nível/XP digitados à sessão em andamento."""
        try:
            settings = self.gui.session_panel.read()
        except ValueError as e:
            self.gui.log(f"[SESSÃO] {e}")
            return
        goal = self.session.goal
        if settings.level is None:
            self.gui.log("[SESSÃO] Informe o nível (ou o XP) atual para corrigir.")
        elif not (self.is_running and goal):
            self.gui.log("[SESSÃO] Sem sessão em andamento: o nível/XP digitados valem a partir do próximo Start.")
        else:
            xp = settings.xp if settings.xp is not None else xp_for_level(settings.level)
            self.session.sync(xp)
            self.gui.log(f"[SESSÃO] XP corrigido para {format_number(xp)} (nível {goal.level}).")

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