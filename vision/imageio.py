import os
from typing import Optional

import cv2
import numpy as np


def read_image(path: str) -> Optional[np.ndarray]:
    """
    Imagem em cores (BGR) ou None se o arquivo não existe/não abre. Existe porque, com o ultralytics importado (o
    bot faz isso), cv2.imread é substituído por uma versão que LANÇA FileNotFoundError em vez de devolver None,
    e também devolve (h, w, 1) para imagens em tons de cinza: ler em cores e converter evita as duas surpresas.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        image = cv2.imread(path)
    except (OSError, cv2.error):
        return None
    return image
