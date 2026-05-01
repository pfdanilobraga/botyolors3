import time
import cv2 
import numpy as np 
from vision.uiValidators import UIValidator 

class SurvivalModule:
    """Classe base para garantir que todos os módulos tenham a mesma estrutura."""
    def __init__(self, keyboard, logger_func):
        self.keyboard = keyboard
        self.log = logger_func
        self.enabled = False
        self.last_action_time = 0

    def check(self, frame):
        pass

class PotionModule(SurvivalModule):
    def __init__(self, keyboard, logger_func, controller=None):
        super().__init__(keyboard, logger_func)
        self.controller = controller
        self.attempts = 0
        self.max_attempts = 3
        self.is_halted = False

    def configure(self, enabled, pot_key, img_path="prayer_empty.png"):
        self.enabled = enabled
        self.pot_key = pot_key
        self.validator = UIValidator(img_path) if enabled else None
        self.attempts = 0
        self.is_halted = False

    def check(self, frame):
        if not self.enabled or self.validator is None:
            return
            
        is_empty = self.validator.is_visible(frame, threshold=0.9)

        if self.is_halted and not is_empty:
            self.is_halted = False
            self.attempts = 0
            self.log("[POTION] Barra preenchida. Módulo reativado.")
            return

        if self.is_halted:
            return
            
        now = time.time()
        if (now - self.last_action_time < 10.0):
            return
        
        if is_empty:
            if self.attempts < self.max_attempts:
                self.attempts += 1
                self.log(f"[POTION] Prayer 0! Combo de 2 doses (Tentativa {self.attempts}/2)...")
        
                for i in range(2):
                    self.keyboard.press(self.pot_key)
                    time.sleep(1.5) 
                
                self.last_action_time = now
            else:
                self.is_halted = True
                self.log("!!! [ALERTA] Falha crítica: Recurso não restaurado após 2 combos.")
        else:
            if self.attempts > 0:
                self.attempts = 0

class PrayerModule(SurvivalModule):
    def __init__(self, keyboard, logger_func):
        super().__init__(keyboard, logger_func)
        self.last_activation = 0
        self.scales = [1.0, 1.05, 1.1, 1.15] 

    def configure(self, enabled, pray_key, img_path="prayer_activated.png"):
        self.enabled = enabled
        self.pray_key = pray_key
        self.validator = UIValidator(img_path) if enabled else None

    def check(self, frame):
        if not self.enabled or self.validator is None:
            return

        now = time.time()
        # Cooldown de 3 segundos para evitar spam durante a animação de ativação
        if (now - self.last_activation < 3.0):
            return

        confidence, roi = self.validator.get_match_data(frame, threshold=0.8, scales=self.scales)

        if confidence >= 0.85 and roi is not None:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_brightness = np.mean(hsv_roi[:, :, 2]) 

            # Verificação de estado por brilho (Glow da Oração)
            if avg_brightness < 82:
                self.log(f"[PRAYER] Ativando... Brilho detectado: {avg_brightness:.1f}")
                self.keyboard.press(self.pray_key)
                self.last_activation = now

class CrystalMaskModule(SurvivalModule):
    def __init__(self, keyboard, logger_func):
        super().__init__(keyboard, logger_func)
        self.mask_key = None
        self.interval = 290  # 4 minutos e 50 segundos
        self.last_activation = 0

    def configure(self, enabled, mask_key):
        self.enabled = enabled
        self.mask_key = mask_key
        # Inicializa o tempo como 'agora' ao configurar, para não gastar recurso no login
        self.last_activation = time.time()

    def check(self, frame):
        if not self.enabled or not self.mask_key:
            return

        now = time.time()
        elapsed = now - self.last_activation

        if elapsed >= self.interval:
            self.log(f"[CRYSTAL MASK] Renovando máscara ({elapsed:.0f}s passados).")
            self.keyboard.press(self.mask_key)
            self.last_activation = now