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
from antiban import humanize
from engine.logout import LogoutAction
from engine.safety import SafetyGuard
from engine.session import FROZEN, LEVEL_UP, SessionEvent, SessionMonitor, format_duration, format_number
from engine.skill_goals import ProgressStore, SkillGoal, xp_for_level
from engine.xp_checker import XpChecker
from vision.chat_levelup import ChatLevelUpWatcher
from vision.skills_panel import skill_cell
from vision.stamina import StaminaBarReader

class BotController:
    LEVEL_TRIGGER_COOLDOWN = 30.0   # s mínimos entre duas leituras pedidas pela mensagem de nível no chat

    def __init__(self):
        # 1. Configurações de Estado
        self.is_running = False
        self.last_state = "IDLE"
        self._goal_skill = ""
        self._xp_hotkey = "h"
        self._last_level_trigger = 0.0    # instante do último gatilho de leitura por mensagem de nível no chat
        self._xp_lock = threading.Lock()  # uma leitura de XP por vez (laço do bot x botão "Ler XP agora")

        # 2. Inicialização da GUI
        self.root = tk.Tk()
        self.gui = BotInterface(self.root, self.start, self.stop)
        self.gui.log("[SYSTEM] Inicializando Sensores e Atuadores...")

        # 3. Instanciação dos Módulos (Core)
        self._initialize_core_modules()

        # 4. Sessão: tempo limite, XP/nível lidos da aba Skills e meta
        self.session = SessionMonitor()
        self.progress = ProgressStore()
        self.xp_checker = XpChecker(self.capture, self.mouse, self.keyboard, self.gui.log,
                                    focus_callback=self._ensure_game_focus)
        self.chat_watcher = ChatLevelUpWatcher()  # mensagem de subida de nível no chat = hora de ler o XP
        # Segurança: perdido (muito tempo sem rochas) -> parar, avisar e deslogar
        self.safety = SafetyGuard(self.fsm)
        self.logout = LogoutAction(self.capture, self.mouse, self.keyboard, self.gui.log,
                                   is_ingame=self.inventory_checker.is_open, focus_callback=self._ensure_game_focus)
        self._pending_logout = False
        self.gui.bind_session(self.session.snapshot, self.sync_progress, self.read_xp_now)
        
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

        # Câmera: gira com as setas se ficar um tempo sem ver rochas
        try:
            camera_after = max(float(self.gui.camera_after_var.get().strip().replace(",", ".")), 3.0)
        except ValueError:
            camera_after = 10.0
        self.fsm.camera_search_enabled = bool(has_ia_active and self.gui.use_camera_var.get())
        self.fsm.camera_search_after = camera_after
        if self.fsm.camera_search_enabled:
            self.gui.log(f"[CÂMERA] Busca ativa: gira a câmera (setas) se ficar {camera_after:g}s sem ver rochas.")

        self._configure_goals(has_ia_active)
        self._configure_safety(has_ia_active)
        return has_ia_active

    def _configure_safety(self, has_ia_active: bool):
        """Liga a rede de segurança (bot perdido) conforme o painel Sessão & Metas."""
        settings = self.gui.session_panel.read()
        self._pending_logout = False
        enabled = bool(settings.safety)   # "perdido" só dispara com a IA procurando rochas; a tela congelada vale sempre
        self.safety.configure(enabled, settings.lost_after_s, settings.logout)
        if enabled:
            logout = "Deslogará em seguida." if settings.logout and self.logout.ready else (
                "Sem assets/logout_botao.png: só vai parar e avisar, sem deslogar." if settings.logout else "")
            self.gui.log(f"[SEGURANÇA] Ativa: para se ficar {settings.lost_after_s / 60:g} min sem ver rochas ou se a tela do jogo ficar parada por {self.safety.FROZEN_AFTER_S:g}s (jogo travado). {logout}".rstrip())

    def _configure_goals(self, has_ia_active: bool):
        """Tempo limite + leitura do XP/meta de nível, a partir do painel Sessão & Metas."""
        settings = self.gui.session_panel.read()
        skill = self.gui.skill_var.get()
        self._goal_skill, self._xp_hotkey = skill, settings.hotkey

        auto = settings.auto_read
        if auto and not self.xp_checker.ready:
            self.gui.log("[SESSÃO] Faltam os modelos da aba Skills em assets/skills: leitura do XP desligada.")
            auto = False
        elif auto and skill_cell(skill) is None:
            self.gui.log(f"[SESSÃO] '{skill}' ainda não está mapeada na aba Skills: leitura do XP desligada.")
            auto = False

        goal = None
        if auto or settings.level is not None:
            goal = SkillGoal(skill, settings.target_level)
        self.session.configure(settings.duration_s, goal, settings.stop_on_goal, settings.interval_s if auto else 0.0)
        if goal and settings.level is not None:  # ponto de partida digitado; a 1ª leitura do jogo o substitui
            start_xp = settings.xp if settings.xp is not None else xp_for_level(settings.level)
            self.session.observe(start_xp, manual=True)

        if settings.duration_s > 0:
            self.gui.log(f"[SESSÃO] Tempo de bot: {format_duration(settings.duration_s)}.")
        if goal:
            meta = f"meta: nível {goal.target_level}" if goal.target_level else "sem meta"
            how = (f"lendo o XP na aba Skills (tecla '{settings.hotkey}') a cada ~{settings.interval_s / 60:g} min (o intervalo varia)"
                   if auto else "sem leitura automática do XP")
            self.gui.log(f"[SESSÃO] {skill}: {meta}; {how}.")

    def start(self):
        """Valida e inicia a thread do motor."""
        if self.is_running: return

        try:
            has_ia = self._configure_session()

            if not self._ensure_game_focus():
                raise RuntimeError("Janela do RuneScape não encontrada ou não pôde ser focada.")

            self.is_running = True
            self.session.start()
            humanize.new_session()   # o ritmo geral (rápido/lento) muda de uma sessão para outra
            self.chat_watcher.reset()  # o que já está no chat ao iniciar não conta como subida de nível nova
            self.safety.reset()
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

                # Leitura do XP na aba Skills: só entre ações (nunca no meio de um clique ou de um drop)
                if self.fsm.state not in ("DROPANDO", "MOVENDO_PARA_ALVO") and self.session.check_due():
                    self._check_xp()

                frame = self.capture.get_frame()
                if frame is None: 
                    self.gui.log("[ERRO VISÃO] Janela do jogo não detectada!")
                    time.sleep(humanize.uniform(1.4, 3.2))
                    continue

                # 1. Prioridade: Sobrevivência (Sempre roda se configurado)
                self.survival_manager.update_all(frame)

                if self.chat_watcher.poll(frame):
                    self._on_level_up_message()

                danger = self.safety.check(frame)
                if danger:
                    self._pending_logout = danger.logout
                    if danger.kind == FROZEN:
                        self._save_frozen_frame(frame)
                    self._handle_session_events([danger])
                    break

                # 2. Lógica de Atividade (FSM) - Só roda se houver alvo
                if self.fsm.target_class:
                    self.fsm.update(frame)
                    self._update_state_log()

                # Durante o drop o ciclo é mais curto para os toques saírem no ritmo configurado
                time.sleep(humanize.uniform(0.02, 0.05) if self.fsm.state == "DROPANDO" else humanize.uniform(0.07, 0.16))
        finally:
            if threading.current_thread() is getattr(self, "bot_thread", None):
                self.is_running = False  # laço terminou (parada, meta, tempo ou erro): libera o Start
            self._finish_session()
            if self._pending_logout:   # emergência (perdido/morto): com o bot já parado, sai do jogo
                self._pending_logout = False
                self.logout.run()

    def _save_frozen_frame(self, frame):
        """Guarda em debug/ a imagem em que a tela estava congelada (prova do que o bot via)."""
        try:
            import cv2
            os.makedirs("debug", exist_ok=True)
            path = os.path.join("debug", f"jogo_travado_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.gui.log(f"[SEGURANÇA] Imagem congelada salva em {path}")
        except Exception:
            pass

    def _handle_session_events(self, events):
        """Registra e avisa (som/popup) subidas de nível, meta atingida e fim do tempo; para o bot se pedido."""
        for event in events:
            self.gui.log(f"[SESSÃO] {event.message}")
            self.gui.post_event(event)
            if event.stop:
                self.is_running = False

    def _on_level_up_message(self):
        """Apareceu "You've just advanced a ... level!" no chat: confirma o nível/XP exato na aba Skills já."""
        now = time.time()
        if now - self._last_level_trigger < self.LEVEL_TRIGGER_COOLDOWN:
            return
        self._last_level_trigger = now
        self.gui.log("[NÍVEL] Mensagem de subida de nível no chat.")
        if self.session.goal is not None and self.session.auto_check:
            # a leitura confirma o nível e avisa (som/meta) com o valor exato; começa alguns segundos depois da mensagem
            self.session.request_check(delay=humanize.human_delay(3.5, 0.5, 0.4, 3.0))
        else:
            # sem leitura do XP não há o que confirmar: só avisa com o som de subida de nível
            self.gui.post_event(SessionEvent(LEVEL_UP, "Subiu de nível", "Mensagem de subida de nível no chat."))

    def _check_xp(self):
        """Lê o XP real na aba Skills e entrega o resultado à sessão (que avisa nível/meta)."""
        if not self._xp_lock.acquire(blocking=False):
            return
        try:
            reading = self._safe_check(self._goal_skill, self._xp_hotkey)
        finally:
            self._xp_lock.release()
        if reading:
            self.session.observe(reading.xp)
        else:
            self.session.check_failed(self.xp_checker.last_failure or "leitura falhou",
                                      numbers_only=self.xp_checker.last_failure_kind == "numbers")
        self.fsm.last_xp_time = time.time()  # a leitura parou a FSM por alguns segundos: não é "XP parou"
        # A aba Skills/o tooltip podem ter coberto a mensagem do chat durante a leitura: quando ela reaparecer não
        # é uma mensagem nova (senão cada leitura dispararia a próxima)
        self.chat_watcher.reset()

    def _safe_check(self, skill: str, hotkey: str):
        """A leitura é opcional: um erro inesperado nela não pode derrubar o bot (vira uma falha de leitura)."""
        try:
            return self.xp_checker.check(skill, hotkey)
        except Exception as e:
            self.xp_checker.last_failure = f"erro inesperado: {e}"
            self.xp_checker.last_failure_kind = "ui"
            self.gui.log(f"[XP] Leitura falhou: {self.xp_checker.last_failure}")
            return None

    def _finish_session(self):
        """Salva o último nível/XP lido (a tela já abre com ele na próxima sessão) e registra o resumo."""
        goal = self.session.goal
        if goal:
            try:
                self.progress.save_skill(self._goal_skill, goal.xp, goal.target_level)
            except OSError as e:
                self.gui.log(f"[SESSÃO] Não foi possível salvar o progresso: {e}")
        summary = self.session.finish()
        if summary:
            self.gui.log(f"[SESSÃO] {summary}")

    def read_xp_now(self):
        """Botão 'Ler XP agora': com o bot rodando pede a leitura ao laço; parado, lê e mostra o resultado."""
        if self.is_running:
            if self.session.goal is None:
                self.gui.log("[XP] Esta sessão não acompanha o XP: ligue 'Ler o XP real na aba Skills' e reinicie.")
            else:
                self.session.request_check()
                self.gui.log("[XP] Leitura pedida: será feita entre as próximas ações do bot.")
            return
        skill, hotkey = self.gui.skill_var.get(), self.gui.session_panel.hotkey()
        self.gui.log(f"[XP] Lendo o XP de {skill} na aba Skills (tecla '{hotkey}')...")
        threading.Thread(target=self._read_xp_idle, args=(skill, hotkey), daemon=True).start()

    def _read_xp_idle(self, skill: str, hotkey: str):
        if not self._xp_lock.acquire(blocking=False):
            return
        try:
            reading = self._safe_check(skill, hotkey)
        finally:
            self._xp_lock.release()
        if reading:
            self.gui.run_in_gui(lambda: self.gui.session_panel.show_manual_values(reading.level, reading.xp))

    def sync_progress(self):
        """Botão 'Corrigir nível/XP à mão': aplica o nível/XP digitados à sessão em andamento."""
        try:
            settings = self.gui.session_panel.read()
        except ValueError as e:
            self.gui.log(f"[SESSÃO] {e}")
            return
        if settings.level is None:
            self.gui.log("[SESSÃO] Informe o nível (ou o XP) atual para corrigir.")
        elif not (self.is_running and self.session.goal):
            self.gui.log("[SESSÃO] Sem sessão em andamento: o nível/XP digitados valem a partir do próximo Start.")
        else:
            xp = settings.xp if settings.xp is not None else xp_for_level(settings.level)
            self.session.observe(xp, manual=True)
            self.gui.log(f"[SESSÃO] XP corrigido à mão para {format_number(xp)} (nível {settings.level}).")

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