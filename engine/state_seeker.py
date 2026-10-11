import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import os
from antiban import humanize
from transitions import Machine
import vision.uiValidators as uiValidators


class TargetSeekerFSM:
    """
    Motor de Busca definitivo.
    Lê o inventário, procura com YOLO e clica com o Atuador.
    """

    RECLICK_COOLDOWN = 8.0        # s (mediana) entre recliques; cada intervalo é sorteado
    RECLICK_MAX_DISTANCE = 220    # px: rocha mais próxima do personagem aceita para reclicar
    RECLICK_CHECK_DELAY = 2.5     # s (mediana) depois do reclique para conferir se a barra caiu
    NO_BAR_RECLICK_AFTER = 25.0   # s (mediana) sem enxergar a barra (com o XP chegando) até reclicar por segurança
    DEBUG_DIR = "debug"           # fotos da tela em situações estranhas, para diagnóstico
    DEBUG_MAX_FILES = 20
    PICK_CONFIDENCE_MARGIN = 0.12    # rochas com confiança até isso abaixo da melhor também podem ser a escolhida
    RECLICK_THRESHOLD_SPREAD = 0.08  # o limite da barra varia ±8 pontos a cada reclique (não reclica sempre no mesmo %)
    PARK_MIN_DISTANCE = 180         # px do personagem (centro da janela) onde o mouse fica parado depois do clique
    PARK_MAX_DISTANCE = 420
    PARK_MIN_TURN = 50              # graus: a direção de uma parada difere da anterior em pelo menos isso
    PARK_MARGIN = 40                # px de folga até as bordas da janela
    PARK_TOP = 80                   # px do topo (evita a barra de título do Windows)
    LOST_DEBUG_MAX_FILES = 8        # fotos do que o YOLO viu quando não há rocha (debug/sem_rocha_*.jpg)
    CAMERA_STEP_INTERVAL = 1.6      # s (mediana) entre dois giros da câmera (tempo de desenhar a cena e o YOLO olhar)
    CAMERA_HOLD = 0.5               # s (mediana) com a seta pressionada em cada giro
    CAMERA_PITCH_HOLD = 0.4         # idem para inclinar a câmera (seta para cima/baixo)
    CAMERA_TURN_STEPS = (6, 11)     # giros até uma "volta"; depois inclina a câmera e gira de novo (sorteado a cada volta)
    CAMERA_FLIP_CHANCE = 0.12       # chance, a cada giro, de mudar o sentido (uma pessoa muda de ideia)
    CAMERA_REPORT_EVERY = 60.0      # s entre avisos de "ainda sem rochas" no log

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
        self._last_park_angle: Optional[float] = None
        self._last_bar_seen = 0.0
        self.camera_search_enabled = False   # gira a câmera (setas) quando fica um tempo sem ver rochas
        self.camera_search_after = 10.0      # s sem ver o alvo até começar a girar a câmera
        self.clock = time.time               # relógio do FSM (substituível em testes)
        self.sleep = time.sleep              # esperas de reação (substituível em testes)
        self._reset_camera_search()
        self.priority_classes: List[str] = []   # classes clicadas antes do alvo normal (a 1ª vale mais)
        self.current_class: Optional[str] = None  # classe da rocha que está sendo minerada
        self._last_priority_check = 0.0
        self._threshold_base = None
        self._new_interaction()
        self._new_reclick_params()

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
        self._reset_camera_search()
        if self.priority_classes:
            self.log(f"[PRIORIDADE] Classes prioritárias (nesta ordem): {', '.join(self.priority_classes)}")

        if validator_name:
            # Reconstrói a instância para garantir que o novo template seja carregado
            self.validator = uiValidators.create_validator(validator_name)
            self.log(f"[SISTEMA] Validador configurado: {validator_name}")
        else:
            self.validator = None
            self.log("[SISTEMA] Rodando sem validador de XP.")

    def _new_interaction(self) -> None:
        """Sorteia os limiares da interação que vai começar: nenhum tempo do bot é fixo."""
        self._xp_gap_limit = humanize.uniform(2.0, 3.6)       # s sem o indicador de XP até desistir da rocha
        self._min_action = humanize.uniform(3.2, 5.0)         # s mínimos minerando antes de poder desistir
        self._timeout_limit = humanize.uniform(4.5, 7.0)      # idem, sem validador de XP
        self._no_bar_after = humanize.interval(self.NO_BAR_RECLICK_AFTER, 0.3, 0.7, 1.6)
        self._priority_min = humanize.uniform(1.5, 3.5)       # s minerando antes de aceitar trocar para rocha prioritária
        self._priority_gap = humanize.uniform(1.8, 3.8)       # s até a próxima busca por rocha prioritária

    def _new_reclick_params(self) -> None:
        """Sorteia o intervalo mínimo e o limite da barra do PRÓXIMO reclique."""
        self._reclick_cooldown = humanize.interval(self.RECLICK_COOLDOWN, 0.25, 0.7, 1.5)
        self._threshold_base = self.reclick_threshold
        spread = self.RECLICK_THRESHOLD_SPREAD
        self._threshold_now = min(max(self.reclick_threshold + humanize.uniform(-spread, spread), 0.3), 1.0)

    def _effective_threshold(self) -> float:
        """Limite da barra desta vez (o configurado ± sorteio); se o limite configurado mudou, sorteia de novo."""
        if self._threshold_base != self.reclick_threshold:
            self._new_reclick_params()
        return self._threshold_now

    def _click_point(self, detection: dict) -> Tuple[int, int]:
        """Ponto de clique: dentro da caixa da detecção (nunca sempre o centro exato)."""
        box = detection.get("box")
        return humanize.point_in_box(box) if box else (detection["x"], detection["y"])

    def _reset_camera_search(self) -> None:
        self._searching_since: Optional[float] = None   # desde quando procura rocha sem ver nenhuma (p/ a segurança)
        self._lost_since: Optional[float] = None   # desde quando o reclique pede rocha e não há nenhuma à vista
        self._no_target_since: Optional[float] = None
        self._last_camera_step = 0.0
        self._last_camera_report = 0.0
        self._camera_steps = 0
        self._camera_dir = "right"
        self._camera_gap = 0.0           # espera sorteada até o próximo giro
        self._turn_steps = 0             # giros desta volta (sorteado)
        self._steps_in_turn = 0

    def searching_for(self) -> float:
        """Segundos seguidos procurando rocha sem ver nenhuma (0 se achou ou se não está procurando)."""
        return 0.0 if self._searching_since is None else self.clock() - self._searching_since

    def _note_lost(self, frame: np.ndarray, why: str) -> None:
        """Registra o início de um período "barra pedindo reclique, mas nenhuma rocha à vista" (e guarda uma foto)."""
        if self._lost_since is not None:
            return
        self._lost_since = self.clock()
        saved = self._save_detections_debug(frame, "sem_rocha")
        if saved:
            self.log(f"[VISÃO] Foto do que o YOLO viu (caixas laranja = confiança < 0,5): {saved}")

    def _save_detections_debug(self, frame: np.ndarray, label: str) -> Optional[str]:
        """Foto da tela com as detecções do YOLO desenhadas (inclui as de baixa confiança), para diagnóstico."""
        try:
            import cv2
            image = frame.copy()
            for d in self.detector.detect(frame, conf_threshold=0.25):
                x1, y1, x2, y2 = d["box"]
                color = (0, 200, 0) if d["conf"] >= 0.5 else (0, 140, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{d['class']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            os.makedirs(self.DEBUG_DIR, exist_ok=True)
            if len([f for f in os.listdir(self.DEBUG_DIR) if f.startswith(label)]) >= self.LOST_DEBUG_MAX_FILES:
                return None
            path = os.path.join(self.DEBUG_DIR, f"{label}_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return path
        except Exception:
            return None

    def _camera_search(self) -> None:
        """
        Sem nenhuma rocha à vista há `camera_search_after` segundos: gira a câmera com as setas, um passo por vez
        (esperando o YOLO olhar a cena de novo entre um passo e outro). Os giros, as esperas, o sentido e as
        inclinações são sorteados (uma varredura regular seria um padrão fácil de notar). Para assim que uma
        rocha aparece (_reset_camera_search).
        """
        if not self.camera_search_enabled:
            return
        now = self.clock()
        if self._no_target_since is None:
            self._no_target_since = now
            return
        missing = now - self._no_target_since
        if missing < self.camera_search_after or now - self._last_camera_step < self._camera_gap:
            return
        self._last_camera_step = now
        self._camera_gap = humanize.human_delay(self.CAMERA_STEP_INTERVAL, 0.3, 0.6, 1.8)

        if self._camera_steps == 0:
            self._camera_dir = humanize.choice(["left", "right"])
            self._turn_steps = humanize.randint(*self.CAMERA_TURN_STEPS)
            self._last_camera_report = now
            self.log(f"[CÂMERA] Nenhuma rocha à vista há {missing:.0f}s: girando a câmera para procurar.")
        elif now - self._last_camera_report >= self.CAMERA_REPORT_EVERY:
            self._last_camera_report = now
            self.log(f"[CÂMERA] Ainda sem rochas há {missing:.0f}s ({self._camera_steps} giros).")

        if self._steps_in_turn >= self._turn_steps:
            # fim de uma volta: inclina a câmera (para cima ou para baixo, sorteado) e começa outra, às vezes ao contrário
            hold = humanize.human_delay(self.CAMERA_PITCH_HOLD, 0.3, 0.6, 1.8)
            self.keyboard.press(humanize.choice(["up", "down"]), duration=hold, settle=(0.1, 0.2))
            self._steps_in_turn = 0
            self._turn_steps = humanize.randint(*self.CAMERA_TURN_STEPS)
            if humanize.chance(0.35):
                self._camera_dir = "left" if self._camera_dir == "right" else "right"
        else:
            if humanize.chance(self.CAMERA_FLIP_CHANCE):
                self._camera_dir = "left" if self._camera_dir == "right" else "right"
            hold = humanize.human_delay(self.CAMERA_HOLD, 0.3, 0.6, 1.8)
            self.keyboard.press(self._camera_dir, duration=hold, settle=(0.1, 0.25))
            self._steps_in_turn += 1
        self._camera_steps += 1

    def _choose(self, found: list):
        """
        Entre detecções da mesma classe, uma das que têm confiança quase igual à melhor (não SEMPRE a mais
        confiante: quem joga clica numa das rochas boas à vista, e não toda vez na mesma).
        """
        best = max(found, key=lambda d: d["conf"])
        near = [d for d in found if d["conf"] >= best["conf"] - self.PICK_CONFIDENCE_MARGIN]
        return humanize.choice(near) if len(near) > 1 else best

    def _pick_target(self, detections: list):
        """
        Escolhe a detecção a clicar: primeiro a classe prioritária (na ordem da lista); se não houver nenhuma, o
        alvo normal. Dentro da classe escolhida, uma das mais confiantes (veja _choose). None se não houver nada.
        """
        for cls in self.priority_classes:
            found = [d for d in detections if d["class"].lower() == cls.lower()]
            if found:
                return self._choose(found)
        targets = [d for d in detections if self.target_class.lower() in d["class"].lower()]
        return self._choose(targets) if targets else None

    def _priority_appeared(self, frame: np.ndarray) -> Optional[str]:
        """
        Enquanto minera uma rocha comum, procura de tempos em tempos uma classe prioritária na tela.
        Retorna o nome da classe encontrada (para trocar de rocha) ou None.
        """
        now = self.clock()
        names = {c.lower() for c in self.priority_classes}
        if (self.current_class or "").lower() in names:
            return None  # já está numa rocha prioritária
        if now - self.interaction_start < self._priority_min:
            return None
        if now - self._last_priority_check < self._priority_gap:
            return None
        self._last_priority_check = now
        self._priority_gap = humanize.uniform(1.8, 3.8)
        hits = [d for d in self.detector.detect(frame, conf_threshold=0.5) if d["class"].lower() in names]
        return max(hits, key=lambda d: d["conf"])["class"] if hits else None

    def _pick_park_point(self, w: int, h: int) -> Tuple[int, int]:
        """
        Ponto (x, y) do frame onde estacionar o mouse: em volta do personagem (centro da janela), a uma distância
        sorteada, numa DIREÇÃO qualquer (sempre diferente da anterior em pelo menos PARK_MIN_TURN graus). Fica dentro
        da janela, longe de cantos e barra de título.
        """
        cx, cy = w // 2, h // 2
        for _ in range(60):
            angle = humanize.uniform(0.0, 2 * math.pi)
            if self._last_park_angle is not None:
                turn = abs((angle - self._last_park_angle + math.pi) % (2 * math.pi) - math.pi)
                if turn < math.radians(self.PARK_MIN_TURN):
                    continue
            distance = humanize.uniform(self.PARK_MIN_DISTANCE, self.PARK_MAX_DISTANCE)
            x, y = cx + distance * math.cos(angle), cy + distance * math.sin(angle)
            if self.PARK_MARGIN <= x <= w - self.PARK_MARGIN and self.PARK_TOP <= y <= h - self.PARK_MARGIN:
                self._last_park_angle = angle
                return int(x), int(y)
        # janela pequena demais para a distância pedida: qualquer ponto dentro das margens, longe do personagem
        for _ in range(60):
            x = humanize.randint(self.PARK_MARGIN, max(w - self.PARK_MARGIN, self.PARK_MARGIN))
            y = humanize.randint(self.PARK_TOP, max(h - self.PARK_MARGIN, self.PARK_TOP))
            if math.hypot(x - cx, y - cy) >= self.PARK_MIN_DISTANCE * 0.6:
                return x, y
        return min(cx + self.PARK_MIN_DISTANCE, w - 1), cy

    def _park_mouse(self, frame: np.ndarray):
        """
        Afasta o mouse do personagem e das rochas logo depois de um clique. Com o mouse parado em cima de
        uma rocha perto do personagem, o destaque/tooltip "Mine ..." fica por cima da área da barra de
        stamina e atrapalha a leitura. A direção e a distância são sorteadas a cada vez (um mouse que sempre vai
        para o mesmo lado é um padrão fácil de notar).
        """
        if not self.park_mouse_after_click:
            return
        win = self.screencap.window_region
        if not win:
            return
        h, w = frame.shape[:2]
        x, y = self._pick_park_point(w, h)
        self.sleep(humanize.human_delay(0.35, 0.4, 0.5, 2.5))  # deixa o clique ser registrado antes de mover o mouse
        self.mouse.move_to(x + win["left"], y + win["top"])

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
        now = self.clock()
        reading = self.stamina.read(frame)
        if reading:
            self._last_bar_seen = now

        # Diagnóstico: conferir se o reclique anterior fez a barra cair
        if self._reclick_check and now >= self._reclick_check[0]:
            before = self._reclick_check[1]
            self._reclick_check = None
            after = f"{reading[0]:.0%}" if reading else "sem barra"
            self.log(f"[STAMINA] Após o reclique: {f'{before:.0%}' if before is not None else 'sem leitura'} -> {after}")

        if now - self.last_reclick < self._reclick_cooldown:
            return

        blind_for = None
        if reading is None:
            reference = max(self._last_bar_seen, self.interaction_start, self.last_reclick)
            if now - reference < self._no_bar_after:
                return
            blind_for = now - reference
        elif reading[0] < self._effective_threshold():
            return

        self.last_reclick = now  # respeita o cooldown mesmo quando não houver rocha para clicar
        limit = self._effective_threshold()
        self._new_reclick_params()   # o intervalo e o limite do próximo reclique também são sorteados
        h, w = frame.shape[:2]
        if reading:
            fraction, (bar_x, bar_y) = reading
            player = (bar_x, bar_y + 20)  # o personagem fica logo abaixo da barra
            why = f"Barra em {fraction:.0%} (limite {limit:.0%})"
        else:
            fraction = None
            player = (w // 2, h // 2 + 20)  # sem barra: o personagem fica no centro da janela
            why = f"Barra não visível há {blind_for:.0f}s"

        detections = self.detector.detect(frame, conf_threshold=0.5)
        wanted = (self.current_class or self.target_class).lower()
        # Prefere a classe exata (ex.: "Iron_rock"); só aceita nomes parecidos se não houver nenhuma
        family = self.target_class.lower()   # qualquer rocha do alvo (ex.: "adamantite_rock" e "adamantite_rockbuff")
        targets = ([d for d in detections if d["class"].lower() == wanted]
                   or [d for d in detections if wanted in d["class"].lower()]
                   or [d for d in detections if family in d["class"].lower()])
        if not targets:
            self.log(f"[STAMINA] {why}, mas nenhuma rocha à vista.")
            self._note_lost(frame, why)
            return
        self._lost_since = None

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
        self.sleep(humanize.human_delay(0.45, 0.4, 0.4, 3.0))   # tempo de reação: ninguém clica no mesmo instante em que a barra passa do limite
        px, py = self._click_point(nearest)
        self.mouse.human_click_at(px + win["left"], py + win["top"])
        self._park_mouse(frame)
        self.last_reclick = self.clock()
        self.last_xp_time = self.last_reclick  # o reclique pode atrasar um pouco o XP
        self._reclick_check = (self.last_reclick + humanize.human_delay(self.RECLICK_CHECK_DELAY, 0.2, 0.7, 1.5), fraction)

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
                self._reset_camera_search()
                self.current_class = best["class"]

                # Traduz a coordenada da imagem para a coordenada do seu monitor
                win = self.screencap.window_region
                if win:
                    px, py = self._click_point(best)
                    self.target_coords_abs = (px + win["left"], py + win["top"])
                    self.log(f"[VISÃO] Alvo {best['class']} encontrado em {self.target_coords_abs}")
                    self.alvo_encontrado()
            else:
                if self._searching_since is None:
                    self._searching_since = self.clock()
                self._camera_search()

        elif self.state == "MOVENDO_PARA_ALVO":
            if self.target_coords_abs:
                x, y = self.target_coords_abs
                # tempo de reação (e, bem raramente, uma distração): o clique nunca sai logo depois da detecção
                self.sleep(humanize.human_delay(0.3, 0.45, 0.4, 3.0) + humanize.distraction())
                self.log(f"[AÇÃO] Clicando em: {x}, {y}")
                self.mouse.human_click_at(x, y)
                self._park_mouse(frame)
                self.target_coords_abs = None
                self.interaction_start = self.clock()
                self.last_xp_time = self.clock()
                self._lost_since = None
                self._new_interaction()
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
                if (self.camera_search_enabled and self._lost_since is not None
                        and self.clock() - self._lost_since >= self.camera_search_after):
                    lost_for = self.clock() - self._lost_since
                    self.log(f"[CÂMERA] Barra pedindo reclique e nenhuma rocha à vista há {lost_for:.0f}s: "
                             "procurando rochas com a câmera.")
                    self._no_target_since = self._lost_since   # a busca com a câmera já começa a girar
                    self._searching_since = self._lost_since
                    self._lost_since = None
                    self.voltar_busca()
                    return

            if self.validator and self.validator.template is not None:
                if self.validator.is_visible(frame, threshold=0.7):
                    self.last_xp_time = self.clock()

            tempo_sem_xp = self.clock() - self.last_xp_time
            tempo_total_acao = self.clock() - self.interaction_start

            if tempo_sem_xp > self._xp_gap_limit and tempo_total_acao > self._min_action:
                score = f" (indicador: {self.validator.last_score:.2f})" if self.validator else ""
                self.log(f"[ESTADO] XP parou{score}. Buscando nova rocha...")
                self.voltar_busca()
        else:
            if self.clock() - self.interaction_start > self._timeout_limit:
                self.log("[ESTADO] Timeout (Sem validador). Reiniciando busca.")
                self.voltar_busca()
