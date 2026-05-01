from engine.survival_modules import PotionModule, PrayerModule, CrystalMaskModule

class SurvivalManager:
    def __init__(self, keyboard, logger_func):
        self.potion = PotionModule(keyboard, logger_func)
        self.prayer = PrayerModule(keyboard, logger_func)
        self.crystal_mask = CrystalMaskModule(keyboard, logger_func)

    def configure_all(self, pot_enabled, pot_key, pray_enabled, pray_key, mask_enabled, mask_key):
        """Distribui as configurações de forma independente."""
        self.potion.configure(pot_enabled, pot_key) 
        self.prayer.configure(pray_enabled, pray_key)
        self.crystal_mask.configure(mask_enabled, mask_key)

    def update_all(self, frame):
        """Executa a verificação de todos os serviços."""
        self.potion.check(frame)
        self.prayer.check(frame)
        self.crystal_mask.check(frame)