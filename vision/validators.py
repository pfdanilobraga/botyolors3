import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict, Union

class MiningValidator:
    """
    Handles binary mining state validation using Template Matching.
    Adheres strictly to the "Percepção de Interface 2D" rule.
    """

    def __init__(self, anchor_name: str = "indicador_xp.png"):
        """
        Initializes the validator with the UI indicator template.
        """
        self.anchor_path = os.path.join("assets", anchor_name)
        self.template = None
        
        # Load template if it exists
        if os.path.exists(self.anchor_path):
            self.template = cv2.imread(self.anchor_path)
            if self.template is None:
                print(f"[Aviso] Falha ao carregar a imagem em: {self.anchor_path}")
        else:
            print(f"[Aviso] Âncora de mineração não encontrada: {self.anchor_path}")
            print("Dica: Execute 'tools/ferramenta_setup.py' para criar este arquivo.")

    def esta_minerando(self, frame: np.ndarray, roi_coords: Optional[Union[Tuple, Dict]] = None, threshold: float = 0.8) -> bool:
        """
        Checks if the mining indicator is present in the frame.
        
        Args:
            frame (np.ndarray): The current game frame (BGR).
            roi_coords (Optional): Coordinates for Region of Interest (x, y, w, h) or (x1, y1, x2, y2).
            threshold (float): Matching confidence threshold (0 to 1).
            
        Returns:
            bool: True if indicator is visible (mining active), False otherwise.
        """
        # If no template is loaded, we cannot validate
        if self.template is None:
            return False
            
        # 1. Apply ROI optimization if provided
        search_area = frame
        if roi_coords:
            try:
                # Assume (x, y, w, h)
                x, y, w, h = roi_coords
                search_area = frame[y:y+h, x:x+w]
            except Exception as e:
                print(f"[Erro] ROI coords inválidos: {e}. Usando frame inteiro.")
                search_area = frame

        # 2. Perform Template Matching
        # Safety check: Search area must be larger than or equal to template
        if search_area.shape[0] < self.template.shape[0] or search_area.shape[1] < self.template.shape[1]:
            # Area is too small for template matching, preventing OpenCV crash
            return False

        # We use TM_CCOEFF_NORMED as it is robust for UI elements
        res = cv2.matchTemplate(search_area, self.template, cv2.TM_CCOEFF_NORMED)
        
        # 3. Analyze results
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        # Binary result based on threshold
        return max_val >= threshold

if __name__ == "__main__":
    # Internal test block with a dummy frame
    print("Testando MiningValidator...")
    validator = MiningValidator()
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Check return in absence of anchor
    result = validator.esta_minerando(dummy_frame)
    print(f"Resultado do teste (sem âncora): {result}")
