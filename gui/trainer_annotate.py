import glob
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional

import cv2
import customtkinter as ctk

from gui.trainer_canvas import BoxCanvas, CLASS_COLORS
from training import assist, colab
from training import dataset as ds


class AnnotateTab(ctk.CTkFrame):
    """
    Aba 2: desenhar as caixas dos objetos nas imagens coletadas (formato YOLO).

    O assistente aprende com o que já foi anotado (treino rápido) e passa a sugerir as caixas nas
    imagens novas: é só corrigir (botão direito remove, arrastar adiciona) e avançar.
    """

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.images: List[str] = []
        self.idx = 0
        # [classe, x1, y1, x2, y2, sugerida] em pixels da imagem; sugerida=True até o usuário avançar
        self.boxes: List[list] = []
        self.negative = False              # imagem marcada como "sem objetos"
        self._current = None               # (item, nome, largura, altura) a que as caixas em memória pertencem
        self.classes: List[str] = []
        self._yolo = None
        self._yolo_path = None

        self.assistant = assist.Assistant()
        self._learning = False
        self._stop_assist = False
        self._learn_auto = False           # o treino em andamento foi disparado sozinho?
        self._learn_annotated = 0          # nº de anotadas quando o treino começou
        self._auto_block_count = -1        # após parar/falhar, só tenta sozinho de novo com mais imagens anotadas

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = BoxCanvas(self, on_draw=self._on_draw, on_right_click=self._on_right_click,
                                on_click=self._on_click)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self._build_side_panel()
        self.on_item_changed()

    # ------------------------------------------------------------------ painel lateral

    def _build_side_panel(self):
        side = ctk.CTkScrollableFrame(self, width=270)
        side.grid(row=0, column=1, sticky="ns")

        self.progress_label = ctk.CTkLabel(side, text="", font=ctk.CTkFont(weight="bold"), wraplength=250, justify="left")
        self.progress_label.pack(anchor="w", pady=(0, 2))
        self.stats_label = ctk.CTkLabel(side, text="", text_color="gray", wraplength=250, justify="left")
        self.stats_label.pack(anchor="w", pady=(0, 8))

        nav = ctk.CTkFrame(side, fg_color="transparent")
        nav.pack(fill="x")
        ctk.CTkButton(nav, text="◀ Anterior (A)", width=120, command=lambda: self.go(-1)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(nav, text="Próxima (D) ▶", width=120, command=lambda: self.go(1)).pack(side="left")
        ctk.CTkButton(side, text="Ir para a próxima sem rótulo", fg_color="#2c6e49",
                      command=self.go_next_unlabeled).pack(fill="x", pady=(6, 2))
        ctk.CTkButton(side, text="Ir para a mais DIFERENTE das anotadas", fg_color="#1f6f78",
                      command=self.go_most_different).pack(fill="x", pady=(2, 10))

        ctk.CTkLabel(side, text="Classe (teclas 1-9):", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.class_var = tk.IntVar(value=0)
        self.class_frame = ctk.CTkFrame(side, fg_color="transparent")
        self.class_frame.pack(fill="x", pady=(2, 4))
        add = ctk.CTkFrame(side, fg_color="transparent")
        add.pack(fill="x")
        self.new_class_entry = ctk.CTkEntry(add, placeholder_text="nova classe (ex.: Iron_rock)", width=190)
        self.new_class_entry.pack(side="left", padx=(0, 4))
        ctk.CTkButton(add, text="+", width=40, command=self.add_class).pack(side="left")

        ctk.CTkLabel(side, text="Ações:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        for text, command in (("Desfazer última caixa (Z)", self.undo),
                              ("Limpar caixas desta imagem", self.clear),
                              ("Marcar SEM objetos (X)", self.toggle_negative),
                              ("Descartar imagem", self.discard)):
            ctk.CTkButton(side, text=text, fg_color="#444444", hover_color="#555555",
                          command=command).pack(fill="x", pady=2)

        self._build_assistant_panel(side)

        ctk.CTkLabel(side, text="Mouse: arraste para desenhar a caixa; botão direito remove a caixa sob o cursor.",
                     text_color="gray", wraplength=250, justify="left").pack(anchor="w", pady=(12, 0))

    def _build_assistant_panel(self, side):
        box = ctk.CTkFrame(side)
        box.pack(fill="x", pady=(12, 4))
        ctk.CTkLabel(box, text="Assistente (aprende com o que você marca)", font=ctk.CTkFont(weight="bold"),
                     wraplength=240, justify="left").pack(anchor="w", padx=8, pady=(8, 2))
        self.assist_status = ctk.CTkLabel(box, text="", text_color="gray", wraplength=240, justify="left")
        self.assist_status.pack(anchor="w", padx=8)

        params = ctk.CTkFrame(box, fg_color="transparent")
        params.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(params, text="Épocas").pack(side="left")
        self.assist_epochs = ctk.CTkEntry(params, width=56)
        self.assist_epochs.insert(0, "auto")
        self.assist_epochs.pack(side="left", padx=(4, 10))
        ctk.CTkLabel(params, text="imgsz").pack(side="left")
        self.assist_imgsz = ctk.CTkEntry(params, width=56)
        self.assist_imgsz.insert(0, str(assist.DEFAULT_IMGSZ))
        self.assist_imgsz.pack(side="left", padx=4)

        self.learn_button = ctk.CTkButton(box, text="Aprender com o que marquei", fg_color="#8e44ad", command=self.learn)
        self.learn_button.pack(fill="x", padx=8, pady=(8, 2))
        self.stop_learn_button = ctk.CTkButton(box, text="Parar", fg_color="#e74c3c", state="disabled", command=self.stop_learning)
        self.stop_learn_button.pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(box, text="Usar modelo (.pt) do Colab como assistente…", fg_color="#3498db",
                      command=self.import_assistant).pack(fill="x", padx=8, pady=(2, 2))
        auto_row = ctk.CTkFrame(box, fg_color="transparent")
        auto_row.pack(fill="x", padx=8, pady=(6, 0))
        self.auto_learn_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(auto_row, text="Atualizar sozinho a cada", variable=self.auto_learn_var).pack(side="left")
        self.auto_every = ctk.CTkEntry(auto_row, width=44)
        self.auto_every.insert(0, "10")
        self.auto_every.pack(side="left", padx=4)
        ctk.CTkLabel(auto_row, text="novas").pack(side="left")
        self.assist_progress = ctk.CTkProgressBar(box)
        self.assist_progress.set(0)
        self.assist_progress.pack(fill="x", padx=8, pady=(6, 2))
        self.assist_progress_label = ctk.CTkLabel(box, text="", text_color="gray", wraplength=240, justify="left")
        self.assist_progress_label.pack(anchor="w", padx=8)

        self.auto_suggest_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(box, text="Sugerir caixas nas imagens novas", variable=self.auto_suggest_var,
                        command=self._on_auto_suggest_toggle).pack(anchor="w", padx=8, pady=(8, 2))
        conf_row = ctk.CTkFrame(box, fg_color="transparent")
        conf_row.pack(fill="x", padx=8)
        ctk.CTkLabel(conf_row, text="Confiança mínima").pack(side="left")
        self.assist_conf = ctk.CTkEntry(conf_row, width=56)
        self.assist_conf.insert(0, "0.35")
        self.assist_conf.pack(side="left", padx=6)
        self._conf_touched = False   # enquanto o usuário não mexer, o valor segue a maturidade do assistente
        self.assist_conf.bind("<KeyRelease>", lambda e: setattr(self, "_conf_touched", True))
        ctk.CTkLabel(conf_row, text="(auto)", text_color="gray").pack(side="left")
        self.confirm_mode_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(box, text="Só valem as que eu confirmar", variable=self.confirm_mode_var).pack(anchor="w", padx=8, pady=(8, 2))
        ctk.CTkButton(box, text="Confirmar todas desta imagem (C)", fg_color="#2c6e49",
                      command=self.confirm_all).pack(fill="x", padx=8, pady=(4, 2))
        ctk.CTkButton(box, text="Sugerir de novo / Rejeitar (R)", fg_color="#444444", hover_color="#555555",
                      command=self.reject_suggestions).pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkLabel(box, text="Caixas tracejadas com “?” e a confiança são sugestões. CLIQUE na caixa para confirmá-la "
                               "(fica sólida); botão direito remove. Com “Só valem as que eu confirmar” ligado, as não "
                               "confirmadas são descartadas ao avançar e viram “fundo” no treino: ensinam o que NÃO é o alvo. "
                               "Se nenhuma estiver certa, marque SEM objetos (X).",
                     text_color="gray", wraplength=240, justify="left").pack(anchor="w", padx=8, pady=(4, 8))

        ctk.CTkLabel(box, text="Ou sugerir com outro modelo (.pt):", wraplength=240, justify="left").pack(anchor="w", padx=8)
        self.model_combo = ctk.CTkComboBox(box, values=self._list_models(), width=230)
        self.model_combo.pack(fill="x", padx=8, pady=(2, 2))
        ctk.CTkButton(box, text="Sugerir nesta imagem", fg_color="#6c3483", command=self.suggest).pack(fill="x", padx=8, pady=(2, 8))

    @staticmethod
    def _list_models() -> List[str]:
        found = sorted(glob.glob(os.path.join("models", "**", "*.pt"), recursive=True))
        return [p.replace("\\", "/") for p in found] or ["(nenhum modelo em models/)"]

    # ------------------------------------------------------------------ item / carregamento

    def on_item_changed(self):
        """Chamado quando o item ativo muda (ou a aba precisa recarregar a lista de imagens)."""
        self._save_current()
        self._current = None               # nada em memória: os próximos salvamentos viram no-op
        self.boxes, self.negative = [], False
        item = self.app.item
        self.classes = ds.load_classes(item) if item else []
        self.images = ds.list_images(item) if item else []
        self.idx = 0
        self._rebuild_class_buttons()
        self.model_combo.configure(values=self._list_models())
        if item and self.assistant.exists(item) and not self.assistant.is_ready(item):
            self.assistant.load_async(item, lambda ok: self._ui(self._after_assistant_loaded))
        if self.images:
            self.go_next_unlabeled(silent=True)
        else:
            self._clear_view("Nenhuma imagem neste item.\nUse a aba 'Coletar' primeiro.")

    def _ui(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            pass

    def _clear_view(self, message: str):
        self._current = None
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
        note = ""
        self._current = None
        if image is None:
            self.boxes, self.negative = [], False
            self.canvas.set_image(None, f"Não foi possível abrir {name}")
            self.canvas.set_boxes([])
        else:
            h, w = image.shape[:2]
            self._current = (self.app.item, name, w, h)
            stored = ds.read_labels(self.app.item, name)
            self.negative = stored == []
            self.boxes = [[cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, False]
                          for cls, cx, cy, bw, bh in (stored or [])]
            self.canvas.set_image(image)
            if stored is None:
                note = self._apply_suggestions()
            self._refresh_boxes()
        self.progress_label.configure(text=f"Imagem {self.idx + 1}/{len(self.images)}\n{name}{note}")
        self._update_stats()

    def _update_stats(self):
        if not self.app.item:
            self.stats_label.configure(text="")
            return
        s = ds.stats(self.app.item)
        per_class = ", ".join(f"{n}: {c}" for n, c in zip(s["classes"], s["per_class"])) or "—"
        self.stats_label.configure(
            text=f"Anotadas: {s['annotated']}/{s['total']} (sem objetos: {s['negatives']})\nCaixas: {s['boxes']}  [{per_class}]")
        self._update_assist_status(s["annotated"])
        self._maybe_auto_learn(s["annotated"])

    def _maybe_auto_learn(self, annotated: int):
        """Treina o assistente sozinho quando houver imagens anotadas novas suficientes."""
        item = self.app.item
        if not item or self._learning or not self.auto_learn_var.get() or annotated < assist.MIN_ANNOTATED:
            return
        if annotated <= self._auto_block_count:
            return  # o treino automático foi parado/falhou: só tenta de novo com mais imagens anotadas
        try:
            every = max(int(self.auto_every.get()), 1)
        except ValueError:
            every = 10
        if assist.new_since_training(item) >= every:
            self.learn(auto=True)

    # ------------------------------------------------------------------ salvar

    def _save_current(self):
        """
        Grava os rótulos da imagem a que as caixas em memória pertencem. Sugestões ainda não confirmadas são
        descartadas no modo "só valem as que eu confirmar" (padrão) e aceitas no modo contrário.
        """
        if not self._current:
            return
        item, name, w, h = self._current
        if not os.path.isfile(os.path.join(ds.images_dir(item), name)):
            return  # a imagem foi descartada/movida
        keep = [b for b in self.boxes if not (b[5] and self.confirm_mode_var.get())]
        if keep:
            ds.write_labels(item, name, [
                (b[0], (b[1] + b[3]) / 2 / w, (b[2] + b[4]) / 2 / h, (b[3] - b[1]) / w, (b[4] - b[2]) / h)
                for b in keep
            ])
        elif self.negative:
            ds.write_labels(item, name, [])
        else:
            ds.remove_labels(item, name)

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

    def go_most_different(self):
        """Abre a imagem sem rótulo mais diferente das já anotadas: ensina mais ao assistente."""
        if not self.images:
            return
        self._save_current()
        name = ds.most_different_unlabeled(self.app.item)
        if name is None:
            return messagebox.showinfo("Anotação", "Todas as imagens já foram anotadas.")
        self.idx = self.images.index(name)
        self._load_image()

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
            (b[1], b[2], b[3], b[4], CLASS_COLORS[b[0] % len(CLASS_COLORS)],
             self._label_for(b[0]) + (f" ? {b[5]:.2f}" if b[5] else ""), bool(b[5]))
            for b in self.boxes
        ])

    def _smallest_box_at(self, x: float, y: float):
        inside = [b for b in self.boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        return min(inside, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])) if inside else None

    def _on_click(self, x: float, y: float):
        """Clique simples numa caixa sugerida: confirma (vira sólida e passa a valer)."""
        box = self._smallest_box_at(x, y)
        if box is not None and box[5]:
            box[5] = False
            self.negative = False
            self._refresh_boxes()

    def confirm_all(self):
        """Confirma todas as sugestões da imagem atual (útil quando o assistente já está confiável)."""
        if any(b[5] for b in self.boxes):
            for b in self.boxes:
                b[5] = False
            self._refresh_boxes()

    def _on_draw(self, x1: float, y1: float, x2: float, y2: float):
        if not self.classes:
            return messagebox.showinfo("Classe", "Crie ao menos uma classe antes de desenhar caixas.")
        self.boxes.append([self.class_var.get(), x1, y1, x2, y2, False])
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
        self.boxes, self.negative, self._current = [], False, None
        ds.discard_image(self.app.item, name)
        self.images = ds.list_images(self.app.item)
        self.idx = min(self.idx, max(len(self.images) - 1, 0))
        self._load_image() if self.images else self._clear_view("Nenhuma imagem neste item.")

    # ------------------------------------------------------------------ assistente

    def _conf(self) -> float:
        try:
            return min(max(float(self.assist_conf.get().replace(",", ".")), 0.05), 0.95)
        except ValueError:
            return 0.35

    def _apply_suggestions(self) -> str:
        """Acrescenta as sugestões do assistente à imagem atual (sem rótulo). Retorna uma nota para o painel."""
        item = self.app.item
        if not self.auto_suggest_var.get() or not self.assistant.exists(item):
            return ""
        if not self.assistant.is_ready(item):
            if not self.assistant.loading:
                self.assistant.load_async(item, lambda ok: self._ui(self._after_assistant_loaded))
            return "\nAssistente carregando…"
        conf = self._conf()
        everything = self.assistant.suggest(item, self.canvas._image, self.classes, 0.03)
        shown = sorted((s for s in everything if s[5] >= conf), key=lambda s: -s[5])[:15]
        for cls, x1, y1, x2, y2, c in shown:
            self.boxes.append([cls, x1, y1, x2, y2, c])   # c = confiança (> 0): marca a caixa como sugestão
        if shown:
            if self.confirm_mode_var.get():
                return f"\n{len(shown)} sugestão(ões): CLIQUE nas corretas; as demais são descartadas ao avançar"
            return f"\n{len(shown)} sugestão(ões): corrija e avance (avançar aceita)"
        trained = assist.trained_count(item)
        if everything:
            top = max(s[5] for s in everything)
            return (f"\nSem sugestões acima de {conf:.2f}: o assistente só vê candidatas fracas (máx. {top:.2f}). "
                    f"Ele aprendeu com {trained} imagens — anote mais imagens VARIADAS e clique em “Aprender” "
                    f"de novo, ou reduza a confiança mínima.")
        return "\nO assistente não viu nada parecido aqui (ainda pouco treinado)."

    def _after_assistant_loaded(self):
        """O assistente terminou de carregar: sugere na imagem atual se ela ainda estiver sem rótulo."""
        if (not self.images or self.boxes or self.negative or self.canvas.image_size == (0, 0)
                or ds.read_labels(self.app.item, self.images[self.idx]) is not None):
            return
        note = self._apply_suggestions()
        self._refresh_boxes()
        self.progress_label.configure(text=f"Imagem {self.idx + 1}/{len(self.images)}\n{self.images[self.idx]}{note}")

    def reject_suggestions(self):
        """
        Com sugestões na imagem: remove-as. Sem nenhuma e sem caixas próprias: pede sugestões de novo
        (com a confiança atual). Se a imagem já tem caixas confirmadas, não acrescenta nada.
        """
        if not self.images:
            return
        head = f"Imagem {self.idx + 1}/{len(self.images)}\n{self.images[self.idx]}"
        if any(b[5] for b in self.boxes):
            self.boxes = [b for b in self.boxes if not b[5]]
            self._refresh_boxes()
            self.progress_label.configure(text=head + "\nSugestões removidas")
        elif not self.boxes:
            note = self._apply_suggestions()
            self._refresh_boxes()
            if note:
                self.progress_label.configure(text=head + note)
        else:
            self.progress_label.configure(text=head + "\nA imagem já tem caixas; limpe-as antes de sugerir de novo.")

    def _on_auto_suggest_toggle(self):
        if self.auto_suggest_var.get() and self.images and not self.boxes and not self.negative \
                and ds.read_labels(self.app.item, self.images[self.idx]) is None:
            self.reject_suggestions()

    def _update_assist_status(self, annotated: int):
        item = self.app.item
        if self._learning:
            return
        if not item or not self.assistant.exists(item):
            if annotated < assist.MIN_ANNOTATED:
                text = f"Anote ~{assist.MIN_ANNOTATED}+ imagens à mão; depois clique em “Aprender” para o assistente começar a sugerir."
            else:
                text = f"💡 Você já anotou {annotated} imagens: clique em “Aprender com o que marquei”."
            self.assist_status.configure(text=text, text_color="#f1c40f" if annotated >= assist.MIN_ANNOTATED else "gray")
            return
        trained = assist.trained_count(item)
        if not self._conf_touched:  # assistente jovem tem confiança baixa mesmo acertando: limite menor
            recommended = "0.25" if trained >= 30 else "0.05"
            if self.assist_conf.get() != recommended:
                self.assist_conf.delete(0, "end")
                self.assist_conf.insert(0, recommended)
        if assist.needs_update(item):
            self.assist_status.configure(
                text=f"💡 +{annotated - trained} imagens anotadas desde o último treino ({trained}). Vale atualizar.",
                text_color="#f1c40f")
        else:
            self.assist_status.configure(text=f"Assistente treinado com {trained} imagens.", text_color="#2ecc71")

    def learn(self, auto: bool = False):
        """Treino rápido do assistente com as imagens anotadas (em segundo plano). auto=True: disparado sozinho."""
        if self._learning or not self.app.item:
            return
        try:
            raw = self.assist_epochs.get().strip().lower()
            epochs = None if raw in ("", "auto") else int(raw)
            imgsz = int(self.assist_imgsz.get())
        except ValueError:
            if auto:
                return
            return messagebox.showwarning("Assistente", "Épocas ('auto' ou um número) e imgsz devem ser válidos.")
        stats = ds.stats(self.app.item)
        if stats["annotated"] < assist.MIN_ANNOTATED:
            if auto:
                return
            return messagebox.showinfo("Assistente", f"Anote ao menos {assist.MIN_ANNOTATED} imagens primeiro "
                                                     f"(hoje: {stats['annotated']}).")
        self._save_current()  # grava a imagem atual antes de treinar
        self._learn_auto, self._learn_annotated = auto, stats["annotated"]
        self._learning, self._stop_assist = True, False
        self.learn_button.configure(state="disabled")
        self.stop_learn_button.configure(state="normal")
        self.assist_progress.set(0)
        self.assist_status.configure(
            text=("Atualizando o assistente sozinho… " if auto else "Treinando… ")
                 + "pode continuar anotando; as sugestões usam o modelo anterior.", text_color="white")
        self.assist_progress_label.configure(text="Carregando bibliotecas (pode levar ~30 s)…")
        item = self.app.item
        threading.Thread(target=self._learn_worker, args=(item, epochs, imgsz), daemon=True).start()

    def import_assistant(self, path: Optional[str] = None):
        """Usa um .pt treinado no Colab (GPU) como assistente: mais rápido e mais forte que o treino local."""
        if not self.app.item:
            return messagebox.showinfo("Assistente", "Escolha ou crie um item primeiro.")
        if self._learning:
            return messagebox.showinfo("Assistente", "Espere o treino em andamento terminar (ou clique em Parar).")
        if not self.classes:
            return messagebox.showinfo("Assistente", "Crie as classes do item antes de importar um modelo.")
        source = path or filedialog.askopenfilename(title="Escolha o best.pt treinado no Colab", initialdir=colab.downloads_dir(),
                                                    filetypes=[("Modelo YOLO", "*.pt")])
        if not source:
            return
        item = self.app.item
        self.assist_progress_label.configure(text="Importando o modelo… (a 1ª vez carrega bibliotecas, ~30 s)")

        def work():
            error, info = None, None
            try:
                info = assist.install_external(item, source)
            except Exception as e:  # noqa: BLE001 - mostrado ao usuário
                error = str(e)
            self._ui(lambda: self._import_finished(item, info, error))
        threading.Thread(target=work, daemon=True).start()

    def _import_finished(self, item: str, info, error):
        if error:
            self.assist_progress_label.configure(text="Importação falhou.")
            return messagebox.showerror("Assistente", error)
        text = "Modelo importado: agora ele é o assistente deste item."
        if info["missing"]:
            text += f" Atenção: ele não conhece {', '.join(info['missing'])}."
        self.assist_progress_label.configure(text=text)
        self.assistant.load_async(item, lambda ok: self._ui(self._after_assistant_loaded))
        self._update_stats()

    def stop_learning(self):
        self._stop_assist = True
        self._auto_block_count = max(self._auto_block_count, self._learn_annotated)  # não recomeça sozinho já
        self.stop_learn_button.configure(state="disabled")

    def _learn_worker(self, item: str, epochs: int, imgsz: int):
        error = None
        result = None
        try:
            result = assist.train_assistant(
                item, epochs=epochs, imgsz=imgsz,
                on_progress=lambda f, t: self._ui(lambda: self._set_assist_progress(f, t)),
                should_stop=lambda: self._stop_assist,
            )
        except Exception as e:  # noqa: BLE001 - mostrado ao usuário
            error = str(e)
        self._ui(lambda: self._learn_finished(item, result, error))

    def _set_assist_progress(self, fraction: float, text: str):
        self.assist_progress.set(fraction)
        self.assist_progress_label.configure(text=text)

    def _learn_finished(self, item: str, result, error):
        self._learning = False
        self.learn_button.configure(state="normal")
        self.stop_learn_button.configure(state="disabled")
        if error:
            self._auto_block_count = max(self._auto_block_count, self._learn_annotated)
            self.assist_progress_label.configure(text=f"Falhou: {error}" if self._learn_auto else "Falhou.")
            self._update_stats()
            return None if self._learn_auto else messagebox.showerror("Assistente", error)
        if result and result.get("path"):
            self.assist_progress.set(1)
            self.assist_progress_label.configure(text="Pronto. As próximas imagens virão com sugestões.")
            self.assistant.load_async(item, lambda ok: self._ui(self._after_assistant_loaded))
        else:
            self._auto_block_count = max(self._auto_block_count, self._learn_annotated)
            self.assist_progress_label.configure(text="Interrompido (o assistente anterior foi mantido).")
        self._update_stats()

    # ------------------------------------------------------------------ sugestão por outro modelo

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
            result = self._yolo(image, verbose=False, conf=self._conf())[0]
        except Exception as e:
            return messagebox.showerror("Sugerir", f"Falha ao rodar o modelo: {e}")

        by_name = {c.lower(): i for i, c in enumerate(self.classes)}
        added = 0
        for box in result.boxes:
            cls_name = str(self._yolo.names[int(box.cls[0])]).lower()
            if cls_name in by_name:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                self.boxes.append([by_name[cls_name], x1, y1, x2, y2, float(box.conf[0])])
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
        elif key in ("r", "R"):
            self.reject_suggestions()
        elif key in ("c", "C"):
            self.confirm_all()
        elif key.isdigit() and 1 <= int(key) <= len(self.classes):
            self.class_var.set(int(key) - 1)
            self._on_class_selected()
