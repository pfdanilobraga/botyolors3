import cv2
import numpy as np
import time
from typing import Tuple, Optional

class InventoryChecker:
    """
    Detects inventory state using OpenCV Template Matching for both 
    backpack visibility and empty slot detection.
    """

    def __init__(self, open_anchor_path: str, empty_slot_path: str):
        """
        Initializes the InventoryChecker.

        Args:
            open_anchor_path (str): Path to anchor image of an open backpack (e.g., "assets/mochila_aberta_ancora.png").
            empty_slot_path (str): Path to template image of a single empty slot (e.g., "assets/slot_vazio_mochila.png").
        """
        self.open_anchor_path = open_anchor_path
        self.empty_slot_path = empty_slot_path
        
        self.anchor_template = cv2.imread(open_anchor_path, cv2.IMREAD_GRAYSCALE)
        self.empty_template = cv2.imread(empty_slot_path, cv2.IMREAD_GRAYSCALE)
        
        if self.anchor_template is None:
            print(f"[Warning] Open backpack anchor not found at {open_anchor_path}")
        if self.empty_template is None:
            print(f"[Warning] Empty slot template not found at {empty_slot_path}")

    def is_open(self, frame: np.ndarray) -> bool:
        """
        Verifies if the backpack UI is currently open and visible.
        """
        if self.anchor_template is None:
            return False
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, self.anchor_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        return max_val > 0.8

    def is_full(self, frame: np.ndarray) -> bool:
        """
        Checks if there are any empty slots visible in the inventory.
        
        Logic: Searches for the 'empty slot' template. 
        If found at least once, the inventory is NOT full.
        """
        if not self.is_open(frame):
            # If not open, we can't say it's full, but we should return True 
            # or handle it in the FSM logic to open it first.
            return False

        if self.empty_template is None:
            return True # Fallback to "full" if we can't check

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We search for the empty slot template
        res = cv2.matchTemplate(gray_frame, self.empty_template, cv2.TM_CCOEFF_NORMED)
        
        # We check if there's any match above a high threshold
        # threshold = 0.8 is usually good for UI elements
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # If the best match is less than 0.8, it means NO empty slots were found
        return max_val < 0.8

if __name__ == "__main__":
    import os
    import sys
    # Add project root to path for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vision.capture import ScreenCapture
    
    print("Iniciando teste de detecção de inventário (Modo Imagem)...")
    
    anchor = "assets/mochila_aberta_ancora.png"
    slot = "assets/slot_vazio_mochila.png"
    
    if not os.path.exists(anchor) or not os.path.exists(slot):
        print("[Erro] Certifique-se de que os arquivos existem em assets/")
        sys.exit(1)
        
    checker = InventoryChecker(anchor, slot)
    cap = ScreenCapture()
    
    try:
        while True:
            frame = cap.get_frame()
            if frame is not None:
                opened = checker.is_open(frame)
                full = checker.is_full(frame) if opened else "N/A"
                print(f"\rMochila Aberta: {opened} | Mochila Cheia: {full}    ", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTeste encerrado.")
