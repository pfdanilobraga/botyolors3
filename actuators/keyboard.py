import time
import pydirectinput
from antiban import humanize
from typing import Callable, Optional, Tuple

class KeyboardActuator:
    """
    Simulates human-like keyboard interactions using pydirectinput.
    """

    def __init__(self, focus_callback: Optional[Callable[[], bool]] = None):
        # Set default pause to 0 to manage it manually
        pydirectinput.PAUSE = 0
        # Chamado antes de cada tecla: garante o foco na janela do jogo (retorna True/False)
        self.focus_callback = focus_callback

    def press(self, key: str, duration: Optional[float] = None, settle: Tuple[float, float] = (0.1, 0.2)):
        """
        Presses and releases a key with a small randomized delay.
        
        Args:
            key (str): The key to press (e.g., 'b', 'space').
            duration (float, optional): How long to hold the key. If None, uses a random human-like duration.
            settle (tuple): Faixa (min, max) da pausa aleatória após soltar a tecla.
        """
        if self.focus_callback and not self.focus_callback():
            print(f"[{time.strftime('%X')}] Keyboard: janela do jogo sem foco, tecla '{key}' ignorada")
            return

        if duration is None:
            # Human-like key press duration (50ms to 150ms)
            duration = humanize.human_delay(0.09, 0.3, 0.55, 2.0)
            
        print(f"[{time.strftime('%X')}] Keyboard: Pressionando '{key}' por {duration:.2f}s")
        
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
        
        # Small delay after releasing the key
        time.sleep(humanize.uniform(*settle))

    def key_down(self, key: str):
        """Mantém a tecla pressionada (ex.: shift para shift-click). Sempre pareie com key_up."""
        pydirectinput.keyDown(key)

    def key_up(self, key: str):
        pydirectinput.keyUp(key)

if __name__ == "__main__":
    # Test block
    print("Testando atuador de teclado...")
    kb = KeyboardActuator()
    kb.press('b')
    print("Teste concluído.")
