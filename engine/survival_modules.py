import time
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
        self.prayer_module = None
        self.controller = controller 
        self.attempts = 0
        self.max_attempts = 3
        self.is_halted = False

    def configure(self, enabled, pot_key, prayer_mod, img_path="prayer_empty.png"):
        self.enabled = enabled
        self.pot_key = pot_key
        self.prayer_module = prayer_mod  
        self.validator = UIValidator(img_path) if enabled else None
        self.attempts = 0
        self.is_halted = False

    def check(self, frame):
        is_empty = self.validator.is_visible(frame, threshold=0.9)

        if self.is_halted and not is_empty:
            self.is_halted = False
            self.attempts = 0
            return

        if not self.enabled or self.is_halted:
            return
            
        if (time.time() - self.last_action_time < 10.0):
            return
        
        if is_empty:
            if self.attempts < self.max_attempts:
                self.attempts += 1
                self.log(f"[SURVIVAL] Prayer 0! Iniciando combo de 2 doses (Tentativa {self.attempts}/3)...")
        
                for i in range(2):
                    self.log(f" > Dose {i+1}/2...")
                    self.keyboard.press(self.pot_key)
                    time.sleep(0.8) 
                
                if self.prayer_module:
                    time.sleep(0.5)
                    self.prayer_module.activate()
                    
                self.last_action_time = time.time()
            else:
                self.is_halted = True
                self.log("!!! [ALERTA] Combo de 3x falhou em restaurar a visão da Prayer.")
        
        else:
            if self.attempts > 0:
                self.log("[SURVIVAL] Prayer restaurada. Resetando contador.")
                self.attempts = 0

class PrayerModule(SurvivalModule):
    def configure(self, enabled, pray_key):
        self.enabled = enabled
        self.pray_key = pray_key

    def activate(self):
        """Método chamado externamente para reativar a prayer."""
        if not self.enabled or not self.pray_key:
            return
            
        time.sleep(1.2)
        self.keyboard.press(self.pray_key)
        self.log(f"[SURVIVAL] Reativando Prayer: {self.pray_key}")

class CrystalMaskModule(SurvivalModule):
    def __init__(self, keyboard, logger_func):
        super().__init__(keyboard, logger_func)
        self.mask_key = None
        self.interval = 290  # 4 minutos e 50 segundos em segundos
        self.last_activation = 0

    def configure(self, enabled, mask_key):
        self.enabled = enabled
        self.mask_key = mask_key
        self.last_activation = time.time()

    def check(self, frame):
        if not self.enabled or not self.mask_key:
            return

        current_time = time.time()
        elapsed = current_time - self.last_activation

        if elapsed >= self.interval:
            self.log(f"[CRYSTAL MASK] Tempo esgotado ({elapsed:.0f}s). Reativando máscara...")
            
            # Executa a ação de apertar a tecla
            self.keyboard.press(self.mask_key)
            
            # Reseta o timer
            self.last_activation = current_time