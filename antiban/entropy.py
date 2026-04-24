import time
import random
import numpy as np
from typing import Tuple

class AntiBan:
    """
    Generates entropy, human-like pauses, and fatigue simulation.
    """
    
    def __init__(self):
        self.fatigue = 0.0 # 0 to 1
        self.session_start = time.time()

    def sleep_natural(self, base_duration: float, variance: float = 0.2):
        """
        Sleeps with a Gaussian distribution around base_duration.
        """
        actual_duration = random.gauss(base_duration, base_duration * variance)
        actual_duration = max(0.01, actual_duration)
        
        # Add fatigue factor
        actual_duration += self.fatigue * 0.5
        
        time.sleep(actual_duration)

    def micro_pause(self):
        """
        Simulates a small hesitation or micro-pause.
        """
        if random.random() > 0.95:
            # 5% chance of a longer pause
            self.sleep_natural(1.5, 0.5)
        else:
            self.sleep_natural(0.1, 0.05)

    def update_fatigue(self):
        """
        Gradually increases fatigue over time.
        """
        elapsed_hours = (time.time() - self.session_start) / 3600
        self.fatigue = min(1.0, elapsed_hours / 8.0) # Maximum fatigue after 8 hours

    def get_random_offset(self, radius: int = 5) -> Tuple[int, int]:
        """
        Returns a random (x, y) offset for clicks to avoid static spots.
        """
        return (random.randint(-radius, radius), random.randint(-radius, radius))
