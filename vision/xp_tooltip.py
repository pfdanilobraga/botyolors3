"""
Leitura do tooltip da aba Skills do RuneScape 3 ("Current XP / Next level / Remainder").

A fonte do jogo é de pixels fixos, então cada dígito é reconhecido por comparação exata do bitmap com uma
biblioteca de glifos (assets/skills/fonte_xp.json + config/fonte_xp_aprendida.json). Os três números se
validam entre si: "Next level" tem que ser um valor da tabela de XP, e Current + Remainder = Next level.
Dígitos ainda desconhecidos são deduzidos por essa mesma conta (e aprendidos para as próximas leituras).
"""
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine.skill_goals import MAX_LEVEL, xp_for_level
from vision.imageio import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(ROOT, "assets", "skills")
FONT_PATH = os.path.join(SKILLS_DIR, "fonte_xp.json")
LEARNED_FONT_PATH = os.path.join(ROOT, "config", "fonte_xp_aprendida.json")

TEXT_THRESHOLD = 120   # tons de cinza acima disso são texto (bege claro sobre fundo quase preto)
LABEL_MATCH = 0.90     # nota mínima do casamento dos rótulos "Current XP:" etc.
LABELS = ("current", "next", "remainder")
LABEL_FILES = {"current": "tooltip_current_xp.png", "next": "tooltip_next_level.png",
               "remainder": "tooltip_remainder.png"}

NUMBER_ROWS = 13        # linhas da faixa do número, a partir do topo do rótulo (dígitos + cauda da vírgula)
NUMBER_GAP = 2          # px entre o fim do rótulo e a área do número
MAX_NUMBER_WIDTH = 150  # largura máxima procurada até a borda do tooltip
BORDER_DIFF = 25        # diferença de cor para a borda do tooltip aparecer sobre o fundo
MAX_UNKNOWN = 5         # glifos desconhecidos distintos que ainda vale tentar deduzir

DIGITS = "0123456789"
KNOWN_CHARS = frozenset(DIGITS + ",")

# Próximo nível -> nível atual (o "Next level" do tooltip é sempre um valor da tabela)
_LEVEL_BY_NEXT = {xp_for_level(level + 1): level for level in range(1, MAX_LEVEL)}


def glyph_key(bitmap: np.ndarray) -> str:
    """Bitmap booleano -> texto ('#' e '.' por linha, linhas separadas por '/')."""
    return "/".join("".join("#" if v else "." for v in row) for row in bitmap)


def key_to_bitmap(key: str) -> np.ndarray:
    return np.array([[c == "#" for c in row] for row in key.split("/")], dtype=bool)


def split_glyphs(gray_roi: np.ndarray) -> List[np.ndarray]:
    """Recorta cada caractere (bitmap justo ao redor dos pixels de texto), da esquerda para a direita."""
    mask = gray_roi > TEXT_THRESHOLD
    columns = mask.any(axis=0)
    glyphs, start = [], None
    for x, has_text in enumerate(list(columns) + [False]):
        if has_text and start is None:
            start = x
        elif not has_text and start is not None:
            part = mask[:, start:x]
            rows = np.nonzero(part.any(axis=1))[0]
            glyphs.append(part[rows.min():rows.max() + 1])
            start = None
    return glyphs


class GlyphFont:
    """Biblioteca de glifos: caractere ('0'-'9' e ',') -> bitmaps conhecidos."""

    def __init__(self, base_path: str = FONT_PATH, learned_path: str = LEARNED_FONT_PATH):
        self.learned_path = learned_path
        self.glyphs: Dict[str, str] = {}      # chave do bitmap -> caractere
        self._learned: Dict[str, List[str]] = {}
        for path, learned in ((base_path, False), (learned_path, True)):
            for char, keys in self._read(path).items():
                for key in keys:
                    self.glyphs[key] = char
                    if learned:
                        self._learned.setdefault(char, []).append(key)

    @staticmethod
    def _read(path: str) -> Dict[str, List[str]]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def classify(self, bitmap: np.ndarray) -> Optional[str]:
        """Caractere do glifo, ou None se for desconhecido. Aceita até 1 pixel diferente (compressão/ruído)."""
        key = glyph_key(bitmap)
        if key in self.glyphs:
            return self.glyphs[key]
        near = [char for known, char in self.glyphs.items()
                if len(known) == len(key) and sum(a != b for a, b in zip(known, key)) <= 1]
        return near[0] if len(set(near)) == 1 else None

    def add(self, char: str, bitmap: np.ndarray) -> None:
        """Aprende um glifo e grava em config/fonte_xp_aprendida.json."""
        key = glyph_key(bitmap)
        if key in self.glyphs:
            return
        self.glyphs[key] = char
        self._learned.setdefault(char, []).append(key)
        try:
            os.makedirs(os.path.dirname(self.learned_path), exist_ok=True)
            tmp = self.learned_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._learned, f, indent=1)
            os.replace(tmp, self.learned_path)
        except OSError:
            pass  # sem gravar, o glifo vale só nesta execução

    @property
    def digits_known(self) -> str:
        return "".join(sorted({c for c in self.glyphs.values() if c.isdigit()}))

    @property
    def digits_missing(self) -> str:
        return "".join(d for d in DIGITS if d not in self.digits_known)


@dataclass(frozen=True)
class XpReading:
    xp: int
    next_xp: int
    remainder: int
    level: int


def _parse(tokens: List[str]) -> Optional[int]:
    digits = "".join(c for c in tokens if c.isdigit())
    return int(digits) if digits else None


def _valid(current: Optional[int], next_xp: Optional[int], remainder: Optional[int]) -> Optional[int]:
    """Nível se os três números são coerentes entre si e com a tabela de XP; senão None."""
    if current is None or next_xp is None or remainder is None:
        return None
    level = _LEVEL_BY_NEXT.get(next_xp)
    if level is None or current + remainder != next_xp or not xp_for_level(level) <= current < next_xp:
        return None
    return level


def solve(lines: List[List[str]], allowed: str = DIGITS) -> Tuple[Optional[XpReading], Dict[str, Set[str]]]:
    """
    lines: [current, next, remainder], cada um uma lista de caracteres ('0'-'9', ',') ou de chaves de glifos
    desconhecidos (qualquer outro texto). Tenta atribuir a cada glifo desconhecido um dígito de `allowed`
    (glifos distintos = dígitos distintos) de modo que os três números fiquem coerentes.
    Retorna (leitura, possibilidades): a leitura só existe se a solução for única; as possibilidades são, por
    glifo, os dígitos que aparecem em alguma solução (vazio se não há solução ou há glifos demais).
    """
    unknown = sorted({t for line in lines for t in line if t not in KNOWN_CHARS})
    if len(unknown) > MAX_UNKNOWN or len(unknown) > len(allowed):
        return None, {}

    solutions = []
    for digits in itertools.permutations(allowed, len(unknown)):
        assignment = dict(zip(unknown, digits))
        resolved = [[assignment.get(t, t) for t in line] for line in lines]
        values = [_parse(line) for line in resolved]
        level = _valid(*values)
        if level is not None:
            solutions.append((assignment, values, level))

    possible = {key: {assignment[key] for assignment, _, _ in solutions} for key in unknown}
    if len(solutions) != 1:
        return None, possible
    _, (current, next_xp, remainder), level = solutions[0]
    return XpReading(xp=current, next_xp=next_xp, remainder=remainder, level=level), possible


class XpTooltipReader:
    """Encontra o tooltip da habilidade em um frame e lê Current XP, Next level e Remainder."""

    def __init__(self, font: Optional[GlyphFont] = None, skills_dir: str = SKILLS_DIR):
        self.font = font or GlyphFont()
        self.labels: Dict[str, np.ndarray] = {}
        for name, filename in LABEL_FILES.items():
            image = read_image(os.path.join(skills_dir, filename))
            if image is not None:
                self.labels[name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.last_failure = ""
        self.last_failure_kind = ""   # "ui": tooltip não está na tela; "numbers": achou, mas não decifrou os números
        self._pending: Dict[str, Set[str]] = {}   # glifo desconhecido -> dígitos ainda possíveis

    @property
    def ready(self) -> bool:
        return len(self.labels) == len(LABELS)

    def _locate(self, gray: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
        found = {}
        for name in LABELS:
            template = self.labels[name]
            if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                return None
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
            if score < LABEL_MATCH:
                return None
            found[name] = loc
        # Os três rótulos formam uma coluna: mesmo x e linhas na ordem certa
        xs = [found[n][0] for n in LABELS]
        ys = [found[n][1] for n in LABELS]
        if max(xs) - min(xs) > 1 or not ys[0] < ys[1] < ys[2]:
            return None
        return found

    def _right_edge(self, frame: np.ndarray, x_start: int, y: int) -> Optional[int]:
        """Coluna da borda direita do tooltip, achada numa linha em branco (entre duas linhas de texto)."""
        row = frame[y, x_start:x_start + MAX_NUMBER_WIDTH].astype(int)
        if len(row) < 10:
            return None
        diff = np.abs(row - row[0]).max(axis=1) > BORDER_DIFF
        hits = np.nonzero(diff)[0]
        return x_start + int(hits[0]) if len(hits) else None

    def read(self, frame: np.ndarray) -> Optional[XpReading]:
        """XpReading do tooltip visível no frame, ou None (o motivo fica em last_failure / last_failure_kind)."""
        self.last_failure = self.last_failure_kind = ""
        if frame is None or not self.ready:
            self.last_failure, self.last_failure_kind = "modelos do tooltip ausentes em assets/skills", "ui"
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self._locate(gray)
        if found is None:
            self.last_failure, self.last_failure_kind = "tooltip da habilidade não encontrado na tela", "ui"
            return None

        cur_x, cur_y = found["current"]
        label_w = self.labels["current"].shape[1]
        edge = self._right_edge(frame, cur_x + label_w + NUMBER_GAP, cur_y + NUMBER_ROWS)
        if edge is None:
            self.last_failure, self.last_failure_kind = "borda do tooltip não encontrada", "ui"
            return None

        for _ in range(3):  # aprender um glifo pode destravar os outros: reavalia até não aprender mais nada
            lines, unknown_bitmaps = self._read_lines(gray, found, edge)
            if lines is None:
                self.last_failure, self.last_failure_kind = "área do número vazia", "ui"
                return None
            reading, possible = solve(lines, self.font.digits_missing or DIGITS)
            if reading is not None:
                self._learn(possible, unknown_bitmaps)
                return reading
            if not self._learn(possible, unknown_bitmaps):
                break
        self.last_failure = "números do tooltip incoerentes ou com dígitos que ainda não foram aprendidos"
        self.last_failure_kind = "numbers"
        return None

    def _read_lines(self, gray, found, edge):
        """Glifos de cada número, já classificados ('0'-'9', ',' ou a chave do bitmap se desconhecido)."""
        lines: List[List[str]] = []
        unknown_bitmaps: Dict[str, np.ndarray] = {}
        for name in LABELS:
            x, y = found[name]
            x0 = x + self.labels[name].shape[1] + NUMBER_GAP
            if edge - x0 < 4:
                return None, {}
            tokens = []
            for bitmap in split_glyphs(gray[y:y + NUMBER_ROWS, x0:edge]):
                char = self.font.classify(bitmap)
                if char is None:
                    char = glyph_key(bitmap)
                    unknown_bitmaps[char] = bitmap
                tokens.append(char)
            lines.append(tokens)
        return lines, unknown_bitmaps

    def _learn(self, possible: Dict[str, Set[str]], unknown_bitmaps: Dict[str, np.ndarray]) -> bool:
        """
        Cruza as possibilidades de cada glifo desconhecido com as de leituras anteriores; quando sobra um
        único dígito, o glifo entra na biblioteca. Retorna True se aprendeu algum.
        """
        learned = False
        for key, digits in possible.items():
            previous = self._pending.get(key)
            merged = digits & previous if previous else set(digits)
            if not merged:  # contradição (ruído): recomeça com o que esta leitura permite
                merged = set(digits)
            if len(merged) == 1:
                self.font.add(next(iter(merged)), unknown_bitmaps[key])
                self._pending.pop(key, None)
                learned = True
            else:
                self._pending[key] = merged
        return learned
