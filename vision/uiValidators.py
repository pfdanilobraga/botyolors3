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
            if self.template is None:
                print(f"[ERRO] Falha ao decodificar imagem: {self.anchor_path}")
        else:
            print(f"[AVISO] Asset não encontrado: {self.anchor_path}")

    def is_visible(self, frame: np.ndarray, threshold: float = 0.8, roi: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Verifica se o elemento está visível no frame ou em uma região específica (ROI).
        
        Args:
            frame (np.ndarray): Screenshot do jogo.
            threshold (float): Confiança mínima (0.0 a 1.0).
            roi (Tuple): (x, y, largura, altura) para otimizar a busca.
        """
        if self.template is None or frame is None:
            return False

        # 1. Recorte da Região de Interesse (Otimização de Performance)
        search_area = frame
        if roi:
            x, y, w, h = roi
            search_area = frame[y:y+h, x:x+w]

        # 2. Prevenção de erro: Área de busca deve ser maior que o template
        if (search_area.shape[0] < self.template.shape[0] or 
            search_area.shape[1] < self.template.shape[1]):
            return False

        # 3. Template Matching
        res = cv2.matchTemplate(search_area, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold