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
        self.last_score = 0.0  # melhor nota do último is_visible (diagnóstico)
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

        self.last_score = float(best_max_val)
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


class MaskedUIValidator(UIValidator):
    """
    Validador para indicadores cujo desenho muda (ex.: círculo de XP, cujo arco laranja cresce):
    compara só os pixels marcados em assets/<nome>_mask.png (branco = comparar, preto = ignorar).
    Sem arquivo de máscara, se comporta exatamente como o UIValidator.

    Para ser rápido, trabalha em dois estágios: o "núcleo" do ícone localiza candidatos no frame
    inteiro e a máscara confirma cada candidato numa área pequena (descarta ícones parecidos em
    outros lugares, como o da habilidade no painel de Habilidades). A posição achada fica em cache.
    """

    MIN_THRESHOLD = 0.85       # nota mínima do casamento mascarado (verdadeiros >= 0,94; falsos <= 0,76)
    CANDIDATE_THRESHOLD = 0.65
    MAX_CANDIDATES = 4
    NMS_DISTANCE = 15
    SLACK = 6                  # folga (px) ao confirmar um candidato
    CACHE_RADIUS = 40          # área de busca em torno da última posição
    CORE_HALF = 11             # meia-largura do núcleo (quadrado em torno do centro da máscara)

    def __init__(self, anchor_name: str):
        super().__init__(anchor_name)
        self.mask = None
        self._core = None
        self._core_off = (0, 0)
        self._last_loc = None

        mask_path = os.path.splitext(self.anchor_path)[0] + "_mask.png"
        if self.template is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == self.template.shape[:2] and (mask > 127).any():
                self.mask = ((mask > 127) * 255).astype(np.uint8)
                self._prepare_core()

    def _prepare_core(self) -> None:
        ys, xs = np.nonzero(self.mask)
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        h, w = self.template.shape[:2]
        x0, x1 = max(cx - self.CORE_HALF, 0), min(cx + self.CORE_HALF, w)
        y0, y1 = max(cy - self.CORE_HALF, 0), min(cy + self.CORE_HALF, h)
        self._core = self.template[y0:y1, x0:x1].copy()
        self._core_off = (x0, y0)

    def is_visible(self, frame: np.ndarray, threshold: float = 0.8, scales: Optional[list] = None) -> bool:
        if self.mask is None:
            return super().is_visible(frame, threshold, scales)
        if frame is None:
            return False
        self.last_score = self._best_score(frame)
        return self.last_score >= max(threshold, self.MIN_THRESHOLD)

    def _masked_match(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        """Casamento mascarado do template inteiro dentro da área dada. Retorna (nota, (x, y) no frame)."""
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
        th, tw = self.template.shape[:2]
        if x1 - x0 < tw or y1 - y0 < th:
            return 0.0, None
        res = cv2.matchTemplate(frame[y0:y1, x0:x1], self.template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, score, _, loc = cv2.minMaxLoc(res)
        return float(score), (x0 + loc[0], y0 + loc[1])

    def _candidates(self, frame: np.ndarray) -> list:
        """Posições (x, y) do núcleo do ícone com nota razoável, do melhor para o pior."""
        ch, cw = self._core.shape[:2]
        if frame.shape[0] < ch or frame.shape[1] < cw:
            return []
        res = cv2.matchTemplate(frame, self._core, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        found = []
        for _ in range(self.MAX_CANDIDATES):
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < self.CANDIDATE_THRESHOLD:
                break
            found.append((x, y))
            d = self.NMS_DISTANCE
            res[max(y - d, 0):y + d, max(x - d, 0):x + d] = -1
        return found

    def _best_score(self, frame: np.ndarray) -> float:
        th, tw = self.template.shape[:2]

        if self._last_loc is not None:  # caminho rápido: perto de onde o indicador estava
            x, y = self._last_loc
            r = self.CACHE_RADIUS
            score, loc = self._masked_match(frame, x - r, y - r, x + tw + r, y + th + r)
            if score >= self.MIN_THRESHOLD:
                self._last_loc = loc
                return score
            self._last_loc = None

        best, best_loc = 0.0, None
        off_x, off_y = self._core_off
        for cx, cy in self._candidates(frame):
            x0, y0 = cx - off_x, cy - off_y
            score, loc = self._masked_match(frame, x0 - self.SLACK, y0 - self.SLACK,
                                            x0 + tw + self.SLACK, y0 + th + self.SLACK)
            if score > best:
                best, best_loc = score, loc
        if best >= self.MIN_THRESHOLD:
            self._last_loc = best_loc
        return best


def create_validator(anchor_name: str) -> UIValidator:
    """Cria o validador de um indicador; usa a máscara (assets/<nome>_mask.png) se ela existir."""
    return MaskedUIValidator(anchor_name)
