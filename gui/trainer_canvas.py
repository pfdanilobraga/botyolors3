import tkinter as tk
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# (x1, y1, x2, y2, cor, rótulo[, tracejada]) em pixels da IMAGEM (não da tela)
DrawBox = Tuple

CLASS_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#ff6fb5"]


class BoxCanvas(tk.Canvas):
    """
    Mostra uma imagem ajustada ao tamanho do widget e permite desenhar retângulos com o mouse.
    Todas as coordenadas entregues/recebidas pelos callbacks são em pixels da imagem original.
    """

    MIN_DRAG = 6  # menor arrasto (px de tela) que conta como caixa

    def __init__(self, master, on_draw: Optional[Callable[[float, float, float, float], None]] = None,
                 on_right_click: Optional[Callable[[float, float], None]] = None,
                 on_click: Optional[Callable[[float, float], None]] = None, **kwargs):
        kwargs.setdefault("bg", "#1a1a1a")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)
        self.on_draw = on_draw
        self.on_right_click = on_right_click
        self.on_click = on_click    # clique simples (sem arrastar), em coordenadas da imagem
        self.preview_color = "#ffffff"

        self._image: Optional[np.ndarray] = None
        self._photo = None
        self._scale = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._boxes: List[DrawBox] = []
        self._drag_start: Optional[Tuple[int, int]] = None
        self._preview_id = None
        self._redraw_job = None
        self._message = "Sem imagem"

        self.bind("<Configure>", lambda e: self._schedule_redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<ButtonPress-3>", self._right)

    # ------------------------------------------------------------------ API

    @property
    def image_size(self) -> Tuple[int, int]:
        """(largura, altura) da imagem atual, ou (0, 0)."""
        if self._image is None:
            return (0, 0)
        return (self._image.shape[1], self._image.shape[0])

    def set_image(self, image: Optional[np.ndarray], message: str = "Sem imagem") -> None:
        self._image = image
        self._message = message
        self._drag_start = None
        self._schedule_redraw()

    def set_boxes(self, boxes: List[DrawBox]) -> None:
        self._boxes = list(boxes)
        self._schedule_redraw()

    def to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        """Coordenada da tela -> coordenada da imagem (limitada às bordas)."""
        iw, ih = self.image_size
        x = (cx - self._ox) / self._scale
        y = (cy - self._oy) / self._scale
        return (min(max(x, 0.0), iw), min(max(y, 0.0), ih))

    def to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self._scale + self._ox, y * self._scale + self._oy)

    # ------------------------------------------------------------------ desenho

    def _schedule_redraw(self) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_job = None
        self.delete("all")
        cw, ch = max(self.winfo_width(), 2), max(self.winfo_height(), 2)

        if self._image is None:
            self.create_text(cw / 2, ch / 2, text=self._message, fill="#888888", font=("Segoe UI", 13))
            return

        ih, iw = self._image.shape[:2]
        self._scale = min(cw / iw, ch / ih)
        new_w, new_h = max(int(iw * self._scale), 1), max(int(ih * self._scale), 1)
        self._ox, self._oy = (cw - new_w) / 2, (ch - new_h) / 2

        interpolation = cv2.INTER_AREA if self._scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self._image, (new_w, new_h), interpolation=interpolation)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.create_image(self._ox, self._oy, image=self._photo, anchor="nw")

        for box in self._boxes:
            x1, y1, x2, y2, color, label, *extra = box
            dashed = bool(extra and extra[0])
            cx1, cy1 = self.to_canvas(x1, y1)
            cx2, cy2 = self.to_canvas(x2, y2)
            self.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, dash=(6, 3) if dashed else ())
            if label:
                self.create_text(cx1 + 4, cy1 + 2, text=label, fill=color, anchor="nw", font=("Segoe UI", 9, "bold"))

    # ------------------------------------------------------------------ mouse

    def _press(self, event) -> None:
        if self._image is None:
            return
        self._drag_start = (event.x, event.y)
        self._preview_id = self.create_rectangle(event.x, event.y, event.x, event.y,
                                                 outline=self.preview_color, width=2, dash=(4, 2))

    def _motion(self, event) -> None:
        if self._drag_start is not None and self._preview_id is not None:
            self.coords(self._preview_id, self._drag_start[0], self._drag_start[1], event.x, event.y)

    def _release(self, event) -> None:
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        self._drag_start = None
        if self._preview_id is not None:
            self.delete(self._preview_id)
            self._preview_id = None
        if abs(event.x - sx) < self.MIN_DRAG and abs(event.y - sy) < self.MIN_DRAG:
            if self.on_click:  # clique simples, não um arrasto
                self.on_click(*self.to_image(event.x, event.y))
            return
        if abs(event.x - sx) < self.MIN_DRAG or abs(event.y - sy) < self.MIN_DRAG:
            return
        x1, y1 = self.to_image(min(sx, event.x), min(sy, event.y))
        x2, y2 = self.to_image(max(sx, event.x), max(sy, event.y))
        if self.on_draw and x2 > x1 and y2 > y1:
            self.on_draw(x1, y1, x2, y2)

    def _right(self, event) -> None:
        if self._image is not None and self.on_right_click:
            self.on_right_click(*self.to_image(event.x, event.y))
