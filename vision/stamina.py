from typing import Optional, Tuple

import cv2
import numpy as np


class StaminaBarReader:
    """
    Lê a barrinha ciano que aparece acima do personagem enquanto ele minera.

    O personagem fica sempre no centro da janela do jogo, então a barra é procurada numa área em volta do
    centro. A barra tem preenchimento ciano (esquerda) sobre uma trilha verde-azulada escura (direita); a fração
    é largura_ciano / (largura_ciano + largura_trilha), o que independe de escala.

    A barra é um FIO fino (3-4 px de altura). Outras coisas ciano que aparecem sobre o personagem, como o banner
    "Critical swing!" / "x4 damage!" (~20 px de altura) ou textos, são descartadas antes da busca (manchas de
    ciano mais altas que MAX_BAR_H), senão a largura do banner seria lida como uma barra quase cheia.
    """

    HALF_W = 100   # px à esquerda/direita do centro
    UP = 60        # px acima do centro
    DOWN = 30      # px abaixo do centro
    MIN_BAR_W = 30  # largura total mínima da barra (ciano + trilha), em px
    MAX_BAR_W = 100
    MAX_BAR_H = 6   # altura máxima de uma mancha ciano para ser a barra (o banner tem ~20)
    MIN_CONTIGUOUS = 0.8    # fração do trecho entre o 1º e o último pixel ciano que tem que ser ciano

    def read(self, frame: np.ndarray) -> Optional[Tuple[float, Tuple[int, int]]]:
        """
        Retorna (fração preenchida 0-1, (x, y) do centro da barra no frame) ou None se a barra
        não está visível.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x0, x1 = max(w // 2 - self.HALF_W, 0), min(w // 2 + self.HALF_W, w)
        y0, y1 = max(h // 2 - self.UP, 0), min(h // 2 + self.DOWN, h)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cyan = self._thin_only((hue >= 86) & (hue <= 95) & (sat >= 70) & (sat <= 175) & (val >= 238))

        b, g, r = (roi[:, :, i].astype(np.int16) for i in (0, 1, 2))
        track = (r <= 0x30) & (g >= 0x18) & (g <= 0x48) & (b >= 0x18) & (b <= 0x48) & (np.abs(g - b) <= 6)

        cyan_rows = cyan.sum(axis=1)
        row = int(cyan_rows.argmax())
        cyan_w = int(cyan_rows[row])
        if cyan_w < 3:
            return None

        xs = np.nonzero(cyan[row])[0]
        left = int(xs.min())
        if cyan_w < self.MIN_CONTIGUOUS * (int(xs.max()) - left + 1):
            return None  # ciano em pedaços soltos (restos de borda de banner, brilhos): não é o preenchimento da barra
        # a trilha escura fica à direita do preenchimento, na mesma linha
        track_w = int(track[row, left:].sum())
        total = cyan_w + track_w
        if total < self.MIN_BAR_W or total > self.MAX_BAR_W:
            return None

        fraction = min(max(cyan_w / total, 0.0), 1.0)
        return fraction, (x0 + left + total // 2, y0 + row)

    def _thin_only(self, cyan: np.ndarray) -> np.ndarray:
        """Mantém só as manchas de ciano com no máximo MAX_BAR_H px de altura (a barra); descarta banners e textos."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cyan.astype(np.uint8), connectivity=8)
        if count <= 1:
            return cyan
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_HEIGHT] <= self.MAX_BAR_H
        return keep[labels]
