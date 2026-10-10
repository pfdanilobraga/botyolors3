import glob
import os
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional

import cv2
import customtkinter as ctk

from gui.trainer_canvas import BoxCanvas, CLASS_COLORS
from training import dataset as ds


class AnnotateTab(ctk.CTkFrame):
    """Aba 2: desenhar as caixas dos objetos nas imagens coletadas (formato YOLO)."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.images: List[str] = []
        self.idx = 0
        self.boxes: List[list] = []        # [classe, x1, y1, x2, y2] em pixels da imagem
        self.negative = False              # imagem marcada como "sem objetos"
        self.classes: List[str] = []
        self._yolo = None
        self._yolo_path = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = BoxCanvas(self, on_draw=self._on_draw, on_right_click=self._on_right_click)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self._build_side_panel()
        self.on_item_changed()

    # ------------------------------------------------------------------ painel lateral

    def _build_side_panel(self):
        side = ctk.CTkScrollableFrame(self, width=250)
        side.grid(row=0, column=1, sticky="ns")

        self.progress_label = ctk.CTkLabel(side, text="", font=ctk.CTkFont(weight="bold"), wraplength=230, justify="left")
        self.progress_label.pack(anchor="w", pady=(0, 2))
        self.stats_label = ctk.CTkLabel(side, text="", text_color="gray", wraplength=230, justify="left")
        self.stats_label.pack(anchor="w", pady=(0, 8))

        nav = ctk.CTkFrame(side, fg_color="transparent")
        nav.pack(fill="x")
        ctk.CTkButton(nav, text="◀ Anterior (A)", width=110, command=lambda: self.go(-1)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(nav, text="Próxima (D) ▶", width=110, command=lambda: self.go(1)).pack(side="left")
        ctk.CTkButton(side, text="Ir para a próxima sem rótulo", fg_color="#2c6e49",
                      command=self.go_next_unlabeled).pack(fill="x", pady=(6, 10))

        ctk.CTkLabel(side, text="Classe (teclas 1-9):", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.class_var = tk.IntVar(value=0)
        self.class_frame = ctk.CTkFrame(side, fg_color="transparent")
        self.class_frame.pack(fill="x", pady=(2, 4))
        add = ctk.CTkFrame(side, fg_color="transparent")
        add.pack(fill="x")
        self.new_class_entry = ctk.CTkEntry(add, placeholder_text="nova classe (ex.: Iron_rock)", width=170)
        self.new_class_entry.pack(side="left", padx=(0, 4))
        ctk.CTkButton(add, text="+", width=40, command=self.add_class).pack(side="left")

        ctk.CTkLabel(side, text="Ações:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        for text, command in (("Desfazer última caixa (Z)", self.undo),
                              ("Limpar caixas desta imagem", self.clear),
                              ("Marcar SEM objetos (X)", self.toggle_negative),
                              ("Descartar imagem", self.discard)):
            ctk.CTkButton(side, text=text, fg_color="#444444", hover_color="#555555",
                          command=command).pack(fill="x", pady=2)

        ctk.CTkLabel(side, text="Sugerir caixas com um modelo:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        self.model_combo = ctk.CTkComboBox(side, values=self._list_models(), width=230)
        self.model_combo.pack(fill="x")
        ctk.CTkButton(side, text="Sugerir nesta imagem", fg_color="#8e44ad", command=self.suggest).pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(side, text="Dica: treine um modelo com as primeiras ~40 imagens e use-o para pré-anotar o resto; "
                                "depois é só corrigir.", text_color="gray", wraplength=230, justify="left").pack(anchor="w")

        ctk.CTkLabel(side, text="Mouse: arraste para desenhar a caixa; botão direito remove a caixa sob o cursor.",
                     text_color="gray", wraplength=230, justify="left").pack(anchor="w", pady=(12, 0))

    @staticmethod
    def _list_models() -> List[str]:
        found = sorted(glob.glob(os.path.join("models", "**", "*.pt"), recursive=True))
        return [p.replace("\\", "/") for p in found] or ["(nenhum modelo em models/)"]

    # ------------------------------------------------------------------ item / carregamento

    def on_item_changed(self):
        """Chamado quando o item ativo muda (ou a aba precisa recarregar a lista de imagens)."""
        self._save_current()
        item = self.app.item
        self.classes = ds.load_classes(item) if item else []
        self.images = ds.list_images(item) if item else []
        self.idx = 0
        self._rebuild_class_buttons()
        if self.images:
            self.go_next_unlabeled(silent=True)
        else:
            self._clear_view("Nenhuma imagem neste item.\nUse a aba 'Coletar' primeiro.")
        self.model_combo.configure(values=self._list_models())

    def _clear_view(self, message: str):
        self.boxes, self.negative = [], False
        self.canvas.set_image(None, message)
        self.canvas.set_boxes([])
        self.progress_label.configure(text="")
        self._update_stats()

    def _load_image(self):
        if not self.images:
            return self._clear_view("Nenhuma imagem neste item.")
        self.idx = min(max(self.idx, 0), len(self.images) - 1)
        name = self.images[self.idx]
        image = cv2.imread(os.path.join(ds.images_dir(self.app.item), name))
        if image is None:
            self.boxes, self.negative = [], False
            self.canvas.set_image(None, f"Não foi possível abrir {name}")
            self.canvas.set_boxes([])
        else:
            h, w = image.shape[:2]
            stored = ds.read_labels(self.app.item, name)
            self.negative = stored == []
            self.boxes = [[cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                          for cls, cx, cy, bw, bh in (stored or [])]
            self.canvas.set_image(image)
            self._refresh_boxes()
        self.progress_label.configure(text=f"Imagem {self.idx + 1}/{len(self.images)}\n{name}")
        self._update_stats()

    def _update_stats(self):
        if not self.app.item:
            return self.stats_label.configure(text="")
        s = ds.stats(self.app.item)
        per_class = ", ".join(f"{n}: {c}" for n, c in zip(s["classes"], s["per_class"])) or "—"
        self.stats_label.configure(
            text=f"Anotadas: {s['annotated']}/{s['total']} (sem objetos: {s['negatives']})\nCaixas: {s['boxes']}  [{per_class}]")

    # ------------------------------------------------------------------ salvar

    def _save_current(self):
        """Grava os rótulos da imagem atual (se houver imagem carregada)."""
        if not self.images or self.canvas.image_size == (0, 0):
            return
        name = self.images[self.idx]
        w, h = self.canvas.image_size
        if self.boxes:
            ds.write_labels(self.app.item, name, [
                (cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
                for cls, x1, y1, x2, y2 in self.boxes
            ])
        elif self.negative:
            ds.write_labels(self.app.item, name, [])
        else:
            ds.remove_labels(self.app.item, name)

    # ------------------------------------------------------------------ navegação

    def go(self, delta: int):
        if not self.images:
            return
        self._save_current()
        self.idx = (self.idx + delta) % len(self.images)
        self._load_image()

    def go_next_unlabeled(self, silent: bool = False):
        if not self.images:
            return
        self._save_current()
        n = len(self.images)
        for step in range(1, n + 1):
            candidate = (self.idx + step) % n if not silent else (self.idx + step - 1) % n
            if ds.read_labels(self.app.item, self.images[candidate]) is None:
                self.idx = candidate
                self._load_image()
                return
        self._load_image()
        if not silent:
            messagebox.showinfo("Anotação", "Todas as imagens já foram anotadas.")

    # ------------------------------------------------------------------ classes

    def _rebuild_class_buttons(self):
        for child in self.class_frame.winfo_children():
            child.destroy()
        if not self.classes:
            ctk.CTkLabel(self.class_frame, text="Crie a primeira classe abaixo.", text_color="gray").pack(anchor="w")
            return
        self.class_var.set(min(self.class_var.get(), len(self.classes) - 1))
        for i, name in enumerate(self.classes):
            ctk.CTkRadioButton(self.class_frame, text=f"{i + 1}  {name}", variable=self.class_var, value=i,
                               fg_color=CLASS_COLORS[i % len(CLASS_COLORS)],
                               command=self._on_class_selected).pack(anchor="w", pady=2)
        self._on_class_selected()

    def _on_class_selected(self):
        self.canvas.preview_color = CLASS_COLORS[self.class_var.get() % len(CLASS_COLORS)]

    def add_class(self):
        name = ds.sanitize_name(self.new_class_entry.get())
        if not self.app.item:
            return messagebox.showwarning("Classe", "Escolha ou crie um item primeiro.")
        if not self.new_class_entry.get().strip():
            return
        if name.lower() in (c.lower() for c in self.classes):
            return messagebox.showinfo("Classe", f"A classe '{name}' já existe.")
        self.classes.append(name)
        ds.save_classes(self.app.item, self.classes)
        self.new_class_entry.delete(0, "end")
        self.class_var.set(len(self.classes) - 1)
        self._rebuild_class_buttons()
        self.app.notify_classes_changed()

    # ------------------------------------------------------------------ caixas

    def _label_for(self, cls: int) -> str:
        return self.classes[cls] if 0 <= cls < len(self.classes) else f"classe {cls}?"

    def _refresh_boxes(self):
        self.canvas.set_boxes([
            (x1, y1, x2, y2, CLASS_COLORS[cls % len(CLASS_COLORS)], self._label_for(cls))
            for cls, x1, y1, x2, y2 in self.boxes
        ])

    def _on_draw(self, x1: float, y1: float, x2: float, y2: float):
        if not self.classes:
            return messagebox.showinfo("Classe", "Crie ao menos uma classe antes de desenhar caixas.")
        self.boxes.append([self.class_var.get(), x1, y1, x2, y2])
        self.negative = False
        self._refresh_boxes()

    def _on_right_click(self, x: float, y: float):
        inside = [b for b in self.boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        if inside:
            smallest = min(inside, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
            self.boxes.remove(smallest)
            self._refresh_boxes()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self._refresh_boxes()

    def clear(self):
        self.boxes.clear()
        self.negative = False
        self._refresh_boxes()

    def toggle_negative(self):
        if self.boxes:
            return messagebox.showinfo("Sem objetos", "Remova as caixas desta imagem antes de marcá-la como 'sem objetos'.")
        self.negative = not self.negative
        self._save_current()
        self._update_stats()

    def discard(self):
        if not self.images:
            return
        name = self.images[self.idx]
        if not messagebox.askyesno("Descartar", f"Mover {name} para a pasta 'descartadas'?\n(Nada é apagado.)"):
            return
        self.boxes, self.negative = [], False
        ds.discard_image(self.app.item, name)
        self.images = ds.list_images(self.app.item)
        self.idx = min(self.idx, max(len(self.images) - 1, 0))
        self._load_image() if self.images else self._clear_view("Nenhuma imagem neste item.")

    # ------------------------------------------------------------------ sugestão por modelo

    def suggest(self):
        path = self.model_combo.get().strip()
        if not os.path.isfile(path) or not self.images or self.canvas.image_size == (0, 0):
            return messagebox.showinfo("Sugerir", "Escolha um modelo .pt existente e abra uma imagem.")
        if not self.classes:
            return messagebox.showinfo("Sugerir", "Crie as classes do item primeiro.")
        try:
            if self._yolo_path != path:
                from ultralytics import YOLO
                self._yolo, self._yolo_path = YOLO(path), path
            image = cv2.imread(os.path.join(ds.images_dir(self.app.item), self.images[self.idx]))
            result = self._yolo(image, verbose=False, conf=0.4)[0]
        except Exception as e:
            return messagebox.showerror("Sugerir", f"Falha ao rodar o modelo: {e}")

        by_name = {c.lower(): i for i, c in enumerate(self.classes)}
        added = 0
        for box in result.boxes:
            cls_name = str(self._yolo.names[int(box.cls[0])]).lower()
            if cls_name in by_name:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                self.boxes.append([by_name[cls_name], x1, y1, x2, y2])
                added += 1
        self.negative = False
        self._refresh_boxes()
        self.progress_label.configure(text=f"Imagem {self.idx + 1}/{len(self.images)}\n{added} caixa(s) sugerida(s) — confira!")

    # ------------------------------------------------------------------ atalhos

    def handle_key(self, event):
        key = event.keysym
        if key in ("Left", "a", "A"):
            self.go(-1)
        elif key in ("Right", "d", "D"):
            self.go(1)
        elif key in ("z", "Z"):
            self.undo()
        elif key in ("x", "X"):
            self.toggle_negative()
        elif key.isdigit() and 1 <= int(key) <= len(self.classes):
            self.class_var.set(int(key) - 1)
            self._on_class_selected()
