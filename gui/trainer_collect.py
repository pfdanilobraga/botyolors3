import ctypes
import os
import random
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np

from gui.trainer_canvas import BoxCanvas
from training import dataset as ds
from vision.capture import ScreenCapture

try:
    import keyboard  # atalhos globais (F9/F10) funcionam com o jogo em foco
except Exception:  # pragma: no cover
    keyboard = None


class CropDialog(ctk.CTkToplevel):
    """Mostra um frame do jogo; o usuário arrasta um retângulo e salva o recorte em assets/ (template)."""

    def __init__(self, master, frame: np.ndarray, default_name: str, on_saved=None):
        super().__init__(master)
        self.title("Recortar ícone do item")
        self.geometry("1000x720")
        self.frame = frame
        self.on_saved = on_saved
        self.crop = None

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(top, text="Arraste um retângulo bem justo sobre UM ícone do item na mochila.").pack(side="left")

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=10, pady=8)
        ctk.CTkLabel(bottom, text="Nome do arquivo (assets/):").pack(side="left")
        self.name_entry = ctk.CTkEntry(bottom, width=180)
        self.name_entry.insert(0, default_name)
        self.name_entry.pack(side="left", padx=6)
        self.info = ctk.CTkLabel(bottom, text="", text_color="gray")
        self.info.pack(side="left", padx=8)
        ctk.CTkButton(bottom, text="Salvar recorte", command=self.save).pack(side="right")

        self.canvas = BoxCanvas(self, on_draw=self._on_draw)
        self.canvas.pack(fill="both", expand=True, padx=10)
        self.canvas.preview_color = "#2ecc71"
        self.canvas.set_image(frame)
        self.after(200, self.focus)

    def _on_draw(self, x1, y1, x2, y2):
        self.crop = (int(x1), int(y1), int(x2), int(y2))
        self.canvas.set_boxes([(x1, y1, x2, y2, "#2ecc71", "")])
        self.info.configure(text=f"{int(x2 - x1)}x{int(y2 - y1)} px")

    def save(self):
        if not self.crop:
            return messagebox.showinfo("Recorte", "Desenhe o retângulo sobre o ícone primeiro.", parent=self)
        x1, y1, x2, y2 = self.crop
        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return messagebox.showinfo("Recorte", "Recorte muito pequeno (mínimo 12x12 px).", parent=self)
        name = ds.sanitize_name(os.path.splitext(self.name_entry.get())[0]) + ".png"
        path = os.path.join("assets", name)
        if os.path.exists(path) and not messagebox.askyesno("Recorte", f"{name} já existe. Substituir?", parent=self):
            return
        os.makedirs("assets", exist_ok=True)
        cv2.imwrite(path, self.frame[y1:y2, x1:x2])
        if self.on_saved:
            self.on_saved(name)
        self.destroy()


class CollectTab(ctk.CTkFrame):
    """Aba 1: coletar imagens do jogo (F9 = agora, F10 = liga/desliga a auto-captura)."""

    AUTO_DIFF_THRESHOLD = 3.0   # diferença média mínima (0-255) para a auto-captura salvar outra imagem

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.capture = ScreenCapture()
        self.auto = False
        self.session = 0
        self._last_small = None
        self._hotkeys = False

        box = ctk.CTkFrame(self)
        box.pack(fill="x", padx=10, pady=10)
        self.game_label = ctk.CTkLabel(box, text="", font=ctk.CTkFont(weight="bold"))
        self.game_label.pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(box, text="Posicione o personagem perto do objeto (ex.: rochas de ferro) e use as teclas "
                               "abaixo com o RuneScape na frente. A janela do Treinador não pode cobrir o jogo.",
                     text_color="gray", wraplength=760, justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text="Intervalo da auto-captura (s):").pack(side="left")
        self.interval_entry = ctk.CTkEntry(row, width=70)
        self.interval_entry.insert(0, "2.0")
        self.interval_entry.pack(side="left", padx=8)

        buttons = ctk.CTkFrame(box, fg_color="transparent")
        buttons.pack(fill="x", padx=12, pady=8)
        ctk.CTkButton(buttons, text="Capturar agora  (F9)", width=190, command=lambda: self.capture_frame(False)).pack(side="left", padx=(0, 8))
        self.auto_button = ctk.CTkButton(buttons, text="Auto-captura: DESLIGADA  (F10)", width=250, fg_color="#607D8B",
                                         command=self.toggle_auto)
        self.auto_button.pack(side="left")

        self.status_label = ctk.CTkLabel(box, text="", wraplength=760, justify="left")
        self.status_label.pack(anchor="w", padx=12, pady=(4, 2))
        self.count_label = ctk.CTkLabel(box, text="", text_color="gray")
        self.count_label.pack(anchor="w", padx=12, pady=(0, 10))

        tpl = ctk.CTkFrame(self)
        tpl.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(tpl, text="Ícone do item para o Auto-Drop", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(tpl, text="Com o minério na mochila (aberta e visível), recorte o ícone dele. O bot usa esse recorte "
                               "para achar e dropar o item quando a mochila encher.",
                     text_color="gray", wraplength=760, justify="left").pack(anchor="w", padx=12)
        ctk.CTkButton(tpl, text="Recortar ícone do item…", command=self.open_crop).pack(anchor="w", padx=12, pady=10)

        self._register_hotkeys()
        self._poll_status()
        self.on_item_changed()

    # ------------------------------------------------------------------ estado

    def on_item_changed(self):
        self._last_small = None
        self._update_count()

    def _update_count(self):
        item = self.app.item
        total = len(ds.list_images(item)) if item else 0
        self.count_label.configure(text=f"Imagens no item '{item or '—'}': {total}   |   salvas nesta sessão: {self.session}   "
                                        "(meta: 100 a 300, com ângulos e estados variados)")

    def _set_status(self, text: str, color: str = "gray"):
        self.status_label.configure(text=text, text_color=color)

    def _poll_status(self):
        """Atualiza a linha 'Jogo: ...' a cada segundo."""
        try:
            hwnd_ok = bool(self.capture._hwnd) and ctypes.windll.user32.IsWindow(self.capture._hwnd)
            if not hwnd_ok:
                self.capture.locate()
            if not self.capture._hwnd:
                text, color = "Jogo: RuneScape não encontrado (abra o jogo)", "#e74c3c"
            elif ctypes.windll.user32.IsIconic(self.capture._hwnd):
                text, color = "Jogo: RuneScape minimizado", "#e67e22"
            elif self.capture.is_visible():
                r = self.capture.window_region
                text, color = f"Jogo: RuneScape visível ({r['width']}x{r['height']}) — pronto para capturar", "#2ecc71"
            else:
                text, color = "Jogo: RuneScape coberto por outra janela — traga-o para a frente", "#e67e22"
            self.game_label.configure(text=text, text_color=color)
        except Exception:
            pass
        self.after(1000, self._poll_status)

    # ------------------------------------------------------------------ captura

    def _is_new_scene(self, frame: np.ndarray) -> bool:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)
        if self._last_small is not None and np.abs(small - self._last_small).mean() < self.AUTO_DIFF_THRESHOLD:
            return False
        self._last_small = small
        return True

    def capture_frame(self, auto: bool = False) -> bool:
        item = self.app.item
        if not item:
            self._set_status("Escolha ou crie um item no topo da janela.", "#e67e22")
            return False
        if not self.capture.is_visible():
            if not auto:
                self._set_status("O RuneScape precisa estar visível e na frente (sem esta janela cobrindo).", "#e67e22")
            return False
        frame = self.capture.get_frame()
        if frame is None:
            self._set_status("Falha ao capturar a janela do jogo.", "#e74c3c")
            return False
        if auto:
            if not self._is_new_scene(frame):
                return False   # cena praticamente igual à última salva
        else:
            self._is_new_scene(frame)

        ds.create_dataset(item)
        name = f"{item}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(ds.images_dir(item), name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.session += 1
        self._set_status(f"Salvo: {name}", "#2ecc71")
        self._update_count()
        return True

    def toggle_auto(self):
        self.auto = not self.auto
        if self.auto:
            self.auto_button.configure(text="Auto-captura: LIGADA  (F10)", fg_color="#E53935")
            self._set_status("Auto-captura ligada: 1 imagem a cada ~intervalo, só com o jogo visível e cena diferente.", "#2ecc71")
            self._auto_tick()
        else:
            self.auto_button.configure(text="Auto-captura: DESLIGADA  (F10)", fg_color="#607D8B")
            self._set_status("Auto-captura desligada.")

    def _interval(self) -> float:
        try:
            return min(max(float(self.interval_entry.get().replace(",", ".")), 0.5), 30.0)
        except ValueError:
            return 2.0

    def _auto_tick(self):
        if not self.auto:
            return
        self.capture_frame(auto=True)
        self.after(int(self._interval() * 1000 * random.uniform(0.75, 1.25)), self._auto_tick)

    # ------------------------------------------------------------------ ícone (template)

    def open_crop(self):
        if not self.capture.is_visible():
            return messagebox.showinfo("Recorte", "Deixe o RuneScape visível e na frente, com o item na mochila aberta.")
        frame = self.capture.get_frame()
        if frame is None:
            return messagebox.showerror("Recorte", "Falha ao capturar o jogo.")
        default = (self.app.item or "item") + "_ore"
        CropDialog(self, frame, default, on_saved=self._on_template_saved)

    def _on_template_saved(self, name: str):
        self._set_status(f"Ícone salvo em assets/{name}. Selecione-o na aba 'Treinar' como item de drop.", "#2ecc71")
        self.app.notify_assets_changed()

    # ------------------------------------------------------------------ atalhos globais

    def _register_hotkeys(self):
        if keyboard is None or self._hotkeys:
            return
        try:
            keyboard.add_hotkey("f9", lambda: self.after(0, lambda: self.capture_frame(False)))
            keyboard.add_hotkey("f10", lambda: self.after(0, self.toggle_auto))
            self._hotkeys = True
        except Exception:
            self._set_status("Não foi possível registrar F9/F10 (use os botões).", "#e67e22")

    def shutdown(self):
        self.auto = False
        if keyboard is not None and self._hotkeys:
            try:
                keyboard.unhook_all_hotkeys()
            except Exception:
                pass
