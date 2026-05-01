import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict, Union

class UIValidator:
    """
    Validador genérico de interface 2D. 
    Detecta a presença de elementos visuais (âncoras) na tela.
    """

    def __init__(self, anchor_name: str):
        """
        Args:
            anchor_name (str): Nome do arquivo em /assets/ (ex: "indicador_xp.png")
        """
        self.anchor_path = os.path.join("assets", anchor_name)
        self.template = None
        self._load_template()

    def _load_template(self) -> None:
        if os.path.exists(self.anchor_path):
            self.template = cv2.imread(self.anchor_path)
        else:
            print(f"[AVISO] Asset não encontrado: {self.anchor_path}")

    def is_visible(self, frame: np.ndarray, threshold: float = 0.8, scales: Optional[list] = None) -> bool:
        """
        Verifica se o elemento está visível no frame ou em uma região específica (ROI).
        
        Args:
            frame (np.ndarray): Screenshot do jogo.
            threshold (float): Confiança mínima (0.0 a 1.0).
            scales (Optional[list]): Lista de fatores de escala para buscar o template.
        """
        if self.template is None or frame is None:
            return False

        if scales is None:
            scales = [1.0]

        best_max_val = 0
        for scale in scales:
            # 1. Redimensiona o template baseado na escala
            if scale == 1.0:
                resized_template = self.template
            else:
                width = int(self.template.shape[1] * scale)
                height = int(self.template.shape[0] * scale)
                resized_template = cv2.resize(self.template, (width, height), interpolation=cv2.INTER_AREA)

            # 2. Verifica se o template redimensionado ainda cabe no frame
            if (frame.shape[0] < resized_template.shape[0] or 
                frame.shape[1] < resized_template.shape[1]):
                continue

            # 3. Executa o matching
            res = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_max_val:
                best_max_val = max_val
            
            # Otimização: Se já achamos um valor muito alto, não precisa testar outras escalas
            if best_max_val >= 0.98:
                break

        return best_max_val >= threshold

    def get_match_data(self, frame: np.ndarray, threshold: float = 0.8, scales: Optional[list] = None):
        """
        Versão avançada do is_visible que retorna a confiança e a imagem recortada.
        """
        if self.template is None or frame is None:
            return 0.0, None

        scales = scales or [1.0]
        best_max_val = 0
        best_roi = None

        for scale in scales:
            if scale == 1.0:
                resized_template = self.template
            else:
                width = int(self.template.shape[1] * scale)
                height = int(self.template.shape[0] * scale)
                resized_template = cv2.resize(self.template, (width, height), interpolation=cv2.INTER_AREA)

            if frame.shape[0] < resized_template.shape[0] or frame.shape[1] < resized_template.shape[1]:
                continue

            res = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_max_val:
                best_max_val = max_val
                # Recorta a área encontrada
                x, y = max_loc
                h, w = resized_template.shape[:2]
                best_roi = frame[y:y+h, x:x+w]

            if best_max_val >= 0.98: break

        return best_max_val, best_roi