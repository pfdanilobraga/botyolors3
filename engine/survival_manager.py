from engine.survival_modules import PotionModule, PrayerModule, CrystalMaskModule

class SurvivalManager:
    def __init__(self, keyboard, logger_func):
        self.potion = PotionModule(keyboard, logger_func)
        self.prayer = PrayerModule(keyboard, logger_func)
        self.crystal_mask = CrystalMaskModule(keyboard, logger_func)

    def configure_all(self, enabled, pot_key, pray_key, mask_key=None, mask_enabled=False):
        self.prayer.configure(enabled, pray_key)
        self.potion.configure(enabled, pot_key, self.prayer, "prayer_empty.png")
        if hasattr(self, 'crystal_mask'):
            self.crystal_mask.configure(mask_enabled, mask_key)

    def update_all(self, frame):
        self.potion.check(frame)
        self.crystal_mask.check(frame)