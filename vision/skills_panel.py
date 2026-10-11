"""Janela "Skills" do RuneScape 3 (tecla H): onde ela está na tela e onde fica o ícone de cada habilidade."""
import os
import unicodedata
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

from vision.imageio import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR_PATH = os.path.join(ROOT, "assets", "skills", "janela_ancora.png")
CLOSE_BUTTON_PATH = os.path.join(ROOT, "assets", "skills", "botao_fechar.png")

# Ordem do jogo: 3 colunas, da esquerda para a direita, linha a linha
SKILL_ORDER = [
    "Attack", "Hitpoints", "Mining",
    "Strength", "Agility", "Smithing",
    "Defence", "Herblore", "Fishing",
    "Ranged", "Thieving", "Cooking",
    "Prayer", "Crafting", "Firemaking",
    "Magic", "Fletching", "Woodcutting",
    "Runecrafting", "Slayer", "Farming",
    "Construction", "Hunter", "Summoning",
    "Dungeoneering", "Divination", "Invention",
    "Archaeology", "Necromancy",
]

# Habilidades "elite" têm outra curva de XP (a tabela do bot não vale para elas): não há leitura automática
UNSUPPORTED_SKILLS = {"Invention"}

# Nomes em português (como aparecem na lista de habilidades do bot) -> nome do jogo
SKILL_ALIASES = {
    "mineracao": "Mining", "pesca": "Fishing", "lenhador": "Woodcutting", "corte de lenha": "Woodcutting",
    "culinaria": "Cooking", "ferraria": "Smithing", "artesanato": "Crafting", "agricultura": "Farming",
    "cacador": "Hunter", "caca": "Hunter", "arqueologia": "Archaeology", "invencao": "Invention",
    "adivinhacao": "Divination", "oracao": "Prayer", "magia": "Magic", "ataque": "Attack", "defesa": "Defence",
    "forca": "Strength", "agilidade": "Agility", "furtividade": "Thieving", "fabricacao de runas": "Runecrafting",
    "construcao": "Construction", "necromancia": "Necromancy",
}

# Geometria em pixels, relativa ao canto superior esquerdo da âncora (jogo em escala 100%).
# A janela pode estar em qualquer lugar da tela e com larguras diferentes (o jogo distribui as 3 colunas pela
# largura), então só a altura é fixa; as colunas saem da largura medida a cada leitura.
ROW0_Y, ROW_PITCH = 48, 27        # centro da 1ª linha e distância entre linhas (iguais em qualquer largura)
COL0_X = 38                       # centro da 1ª coluna (margem esquerda fixa)
COL2_FROM_RIGHT = 36.5            # centro da 3ª coluna, contado a partir da borda direita
CLOSE_BUTTON_TO_WIDTH = 25        # largura da janela = x do botão X (recorte) + isto
CLOSE_BUTTON_ROWS = (-6, 36)      # faixa de linhas (em relação à âncora) onde o botão X é procurado
CLOSE_BUTTON_COLS = (100, 420)    # idem para as colunas
DEFAULT_WIDTH = 212               # usada se o botão X não for achado (ex.: coberto por outra janela)
ANCHOR_MATCH = 0.85
CLOSE_BUTTON_MATCH = 0.80


class SkillsWindow(NamedTuple):
    """Onde a janela Skills está no frame: canto superior esquerdo da âncora e largura da janela."""
    x: int
    y: int
    width: int


def _normalize(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return plain.strip().lower()


def skill_cell(skill: str) -> Optional[Tuple[int, int]]:
    """(coluna, linha) da habilidade na janela Skills, pelo nome em inglês ou em português; None se desconhecida."""
    wanted = _normalize(skill)
    english = SKILL_ALIASES.get(wanted)
    for index, name in enumerate(SKILL_ORDER):
        if name.lower() == wanted or name == english:
            return None if name in UNSUPPORTED_SKILLS else (index % 3, index // 3)
    return None


class SkillsPanel:
    """Localiza a janela Skills (em qualquer lugar do frame) e calcula onde fica o ícone de cada habilidade."""

    def __init__(self, anchor_path: str = ANCHOR_PATH, close_button_path: str = CLOSE_BUTTON_PATH):
        self.anchor = read_image(anchor_path)
        self.close_button = read_image(close_button_path)
        self.last_score = 0.0   # melhor nota da âncora na última busca (diagnóstico)

    @property
    def ready(self) -> bool:
        return self.anchor is not None

    def locate(self, frame: np.ndarray) -> Optional[SkillsWindow]:
        """A janela Skills no frame (posição e largura), ou None se ela não está aberta."""
        if self.anchor is None or frame is None:
            return None
        h, w = self.anchor.shape[:2]
        if frame.shape[0] < h or frame.shape[1] < w:
            return None
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(frame, self.anchor, cv2.TM_CCOEFF_NORMED))
        self.last_score = float(score)
        if score < ANCHOR_MATCH:
            return None
        return SkillsWindow(x, y, self._width(frame, x, y))

    def _width(self, frame: np.ndarray, x: int, y: int) -> int:
        """Largura da janela, pela posição do botão X (que fica na borda direita do título)."""
        if self.close_button is None:
            return DEFAULT_WIDTH
        top, bottom = CLOSE_BUTTON_ROWS
        left, right = CLOSE_BUTTON_COLS
        y0 = max(y + top, 0)
        strip = frame[y0:y + bottom, x + left:x + right]
        bh, bw = self.close_button.shape[:2]
        if strip.shape[0] < bh or strip.shape[1] < bw:
            return DEFAULT_WIDTH
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(strip, self.close_button, cv2.TM_CCOEFF_NORMED))
        return left + loc[0] + CLOSE_BUTTON_TO_WIDTH if score >= CLOSE_BUTTON_MATCH else DEFAULT_WIDTH

    @staticmethod
    def cell_center(window: SkillsWindow, cell: Tuple[int, int]) -> Tuple[int, int]:
        """Centro (x, y) no frame da célula (coluna, linha) da janela, já considerando a largura dela."""
        col, row = cell
        centers = (COL0_X, window.width / 2, window.width - COL2_FROM_RIGHT)
        return round(window.x + centers[col]), window.y + ROW0_Y + ROW_PITCH * row
