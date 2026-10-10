import tkinter as tk
from typing import Optional

import customtkinter as ctk

from gui.trainer_annotate import AnnotateTab
from gui.trainer_collect import CollectTab
from gui.trainer_train import TrainTab
from training import dataset as ds

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

TAB_COLLECT, TAB_ANNOTATE, TAB_TRAIN = "1. Coletar", "2. Anotar", "3. Treinar"


class TrainerApp(ctk.CTk):
    """Treinador: coletar imagens, anotar, treinar e registrar um novo alvo no bot."""

    def __init__(self, start_tab: Optional[str] = None):
        super().__init__()
        self.title("RS3 VisionBot — Treinador")
        self.geometry("1240x820")
        self.minsize(1000, 680)
        self.item = ""

        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(header, text="Item (alvo):", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(10, 6), pady=8)
        items = ds.list_items()
        self.item_combo = ctk.CTkComboBox(header, values=items or ["iron"], width=220)
        self.item_combo.set(items[0] if items else "iron")
        self.item_combo.pack(side="left")
        ctk.CTkButton(header, text="Usar / Criar", width=110, command=self.use_item).pack(side="left", padx=8)
        self.header_hint = ctk.CTkLabel(header, text="Cada item (ex.: iron, copper) tem seu próprio dataset em dataset/<item>/",
                                        text_color="gray")
        self.header_hint.pack(side="left", padx=10)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        for name in (TAB_COLLECT, TAB_ANNOTATE, TAB_TRAIN):
            self.tabview.add(name)

        self.collect_tab = CollectTab(self.tabview.tab(TAB_COLLECT), self)
        self.annotate_tab = AnnotateTab(self.tabview.tab(TAB_ANNOTATE), self)
        self.train_tab = TrainTab(self.tabview.tab(TAB_TRAIN), self)
        for tab in (self.collect_tab, self.annotate_tab, self.train_tab):
            tab.pack(fill="both", expand=True)
        self.tabview.configure(command=self._on_tab_change)

        self.bind("<Key>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if items:
            self.use_item()
        if start_tab:
            self.tabview.set(start_tab)

    # ------------------------------------------------------------------ item ativo

    def use_item(self):
        name = ds.create_dataset(self.item_combo.get())
        self.item = name
        self.item_combo.configure(values=ds.list_items())
        self.item_combo.set(name)
        self.header_hint.configure(text=f"Item ativo: {name}  (dataset/{name}/)")
        self.notify_item_changed()

    def notify_item_changed(self):
        self.collect_tab.on_item_changed()
        self.annotate_tab.on_item_changed()
        self.train_tab.on_item_changed()

    def notify_classes_changed(self):
        self.train_tab.refresh_classes()

    def notify_assets_changed(self):
        self.train_tab.refresh_assets()

    # ------------------------------------------------------------------ abas / atalhos

    def _on_tab_change(self):
        current = self.tabview.get()
        if current == TAB_ANNOTATE:
            self.annotate_tab.on_item_changed()
        elif current == TAB_TRAIN:
            self.train_tab.on_item_changed()
        elif current == TAB_COLLECT:
            self.collect_tab.on_item_changed()

    def _on_key(self, event):
        if self.tabview.get() != TAB_ANNOTATE:
            return
        if isinstance(self.focus_get(), (tk.Entry, tk.Text)):
            return
        self.annotate_tab.handle_key(event)

    def on_close(self):
        self.annotate_tab._save_current()
        self.train_tab._stop = True
        self.collect_tab.shutdown()
        self.destroy()


def main(start_tab: Optional[str] = None):
    app = TrainerApp(start_tab=start_tab)
    app.mainloop()
