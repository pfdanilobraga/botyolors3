import time
import random
import pydirectinput
from typing import Optional

class KeyboardActuator:
    """
    Simulates human-like keyboard interactions using pydirectinput.
    """

    def __init__(self):
        # Set default pause to 0 to manage it manually
        pydirectinput.PAUSE = 0

    def press(self, key: str, duration: Optional[float] = None):
        """
        Presses and releases a key with a small randomized delay.
        
        Args:
            key (str): The key to press (e.g., 'b', 'space').
            duration (float, optional): How long to hold the key. If None, uses a random human-like duration.
        """
        if duration is None:
            # Human-like key press duration (50ms to 150ms)
            duration = random.uniform(0.05, 0.15)
            
        print(f"[{time.strftime('%X')}] Keyboard: Pressionando '{key}' por {duration:.2f}s")
        
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
        
        # Small delay after releasing the key
        time.sleep(random.uniform(0.1, 0.2))

if __name__ == "__main__":
    # Test block
    print("Testando atuador de teclado...")
    kb = KeyboardActuator()
    kb.press('b')
    print("Teste concluído.")
