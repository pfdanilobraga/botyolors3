import glob
import os
import shutil
import threading
import tkinter as tk
import unicodedata
from tkinter import filedialog, messagebox
from typing import Dict, List

import customtkinter as ctk

from config.activities import ACTIVITIES
from training import dataset as ds
from training import registry
from training.trainer import run_training

NO_DROP = "(nenhum)"
NO_XP = "(nenhum)"

COLAB_TEXT = """# 1) No Colab: Ambiente de execução > Alterar tipo de ambiente > GPU (T4)
# 2) Envie o arquivo {zip_name} (ícone de pasta, à esquerda) e rode esta célula:
!pip -q install ultralytics
!unzip -qo /content/{zip_name} -d /content/ds

# 3) Em outra célula (termina imprimindo um bloco enorme de números: é normal, é o resultado do treino):
from ultralytics import YOLO
YOLO("yolov8n.pt").train(data="/content/ds/data.yaml", imgsz=512, epochs=100, batch=16)

# 4) Em outra célula: baixa o melhor modelo do treino mais recente
import glob, os
from google.colab import files
best = max(glob.glob("/content/runs/detect/*/weights/best.pt"), key=os.path.getmtime)
print(best)
files.download(best)

# 5) Volte ao Treinador e use "Instalar .pt treinado no Colab".
"""


def slug(text: str) -> str:
    """'Mineração' -> 'mineracao' (pasta em models/)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return ds.sanitize_name(ascii_text).lower()


class TrainTab(ctk.CTkFrame):
    """Aba 3: preparar o dataset, treinar (local ou Colab), instalar o modelo e registrá-lo no bot."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._stop = False
        self._training = False

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()
        self.on_item_changed()

    # ------------------------------------------------------------------ construção

    def _field(self, parent, label: str, row: int, widget):
        ctk.CTkLabel(parent, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=(10, 6), pady=3)
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=3)
        return widget

    def _build_left(self):
        left = ctk.CTkScrollableFrame(self, width=430)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self.stats_label = ctk.CTkLabel(left, text="", justify="left", wraplength=400, anchor="w")
        self.stats_label.pack(fill="x", padx=6, pady=(4, 8))

        params = ctk.CTkFrame(left)
        params.pack(fill="x", pady=4)
        params.columnconfigure(1, weight=1)
        ctk.CTkLabel(params, text="Treino", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 2))
        self.epochs = self._field(params, "Épocas:", 1, ctk.CTkEntry(params))
        self.epochs.insert(0, "100")
        self.imgsz = self._field(params, "Tamanho da imagem (imgsz):", 2, ctk.CTkEntry(params))
        self.imgsz.insert(0, "512")
        self.batch = self._field(params, "Batch:", 3, ctk.CTkEntry(params))
        self.batch.insert(0, "16")
        self.base_model = self._field(params, "Modelo base:", 4, ctk.CTkEntry(params))
        self.base_model.insert(0, "yolov8n.pt")
        ctk.CTkLabel(params, text="Sem GPU neste PC, o treino local leva horas — prefira o Colab (botão abaixo) "
                                  "ou teste com 5 épocas e imgsz 320.", text_color="gray", wraplength=390,
                     justify="left").grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(2, 8))

        reg = ctk.CTkFrame(left)
        reg.pack(fill="x", pady=4)
        reg.columnconfigure(1, weight=1)
        ctk.CTkLabel(reg, text="Registro no bot", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 2))
        self.skill = self._field(reg, "Habilidade:", 1, ctk.CTkComboBox(reg, values=list(ACTIVITIES.keys())))
        self.skill.set("Mineração")
        self.name = self._field(reg, "Nome na lista do bot:", 2, ctk.CTkEntry(reg))
        self.target = self._field(reg, "Classe-alvo (o que clicar):", 3, ctk.CTkComboBox(reg, values=[""]))
        self.model_path = self._field(reg, "Arquivo do modelo:", 4, ctk.CTkEntry(reg))
        self.xp = self._field(reg, "Indicador de XP (assets/):", 5, ctk.CTkComboBox(reg, values=[NO_XP]))
        self.drop = self._field(reg, "Item de drop (assets/):", 6, ctk.CTkComboBox(reg, values=[NO_DROP]))
        self.skill.configure(command=lambda _=None: self._suggest_model_path())
        self.name.bind("<KeyRelease>", lambda e: self._suggest_model_path())

        actions = ctk.CTkFrame(left, fg_color="transparent")
        actions.pack(fill="x", pady=8)
        self.train_button = ctk.CTkButton(actions, text="Treinar agora (este PC)", fg_color="#2ecc71", command=self.start_training)
        self.train_button.pack(fill="x", pady=2)
        self.stop_button = ctk.CTkButton(actions, text="Parar treino", fg_color="#e74c3c", state="disabled", command=self.request_stop)
        self.stop_button.pack(fill="x", pady=2)
        ctk.CTkButton(actions, text="Gerar pacote para o Colab (.zip)", fg_color="#3498db", command=self.make_colab).pack(fill="x", pady=(10, 2))
        ctk.CTkButton(actions, text="Instalar .pt treinado no Colab…", fg_color="#8e44ad", command=self.install_external).pack(fill="x", pady=2)

    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Progresso", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))
        self.progress = ctk.CTkProgressBar(right)
        self.progress.set(0)
        self.progress.grid(row=1, column=0, sticky="ew", padx=12)
        self.progress_label = ctk.CTkLabel(right, text="Aguardando", text_color="gray", anchor="w")
        self.progress_label.grid(row=1, column=0, sticky="w", padx=12, pady=(24, 0))
        self.log_box = ctk.CTkTextbox(right, state="disabled", wrap="word")
        self.log_box.grid(row=2, column=0, sticky="nsew", padx=12, pady=(32, 12))

    # ------------------------------------------------------------------ atualizações vindas do app

    def on_item_changed(self):
        item = self.app.item
        if item and not self.name.get():
            self.name.insert(0, item.replace("_", " ").title().replace(" ", "_"))
        self.refresh_classes()
        self.refresh_assets()
        self._suggest_model_path()
        self._refresh_stats()

    def refresh_classes(self):
        classes = ds.load_classes(self.app.item) if self.app.item else []
        self.target.configure(values=classes or [""])
        if classes and self.target.get() not in classes:
            self.target.set(classes[0])
        self._refresh_stats()

    def refresh_assets(self):
        pngs = sorted(os.path.basename(p) for p in glob.glob(os.path.join("assets", "*.png")))
        xps = sorted({v.get("xp_validator", "") for items in ACTIVITIES.values() for v in items.values() if v.get("xp_validator")})
        for sub in glob.glob(os.path.join("assets", "indicadores", "*.png")):
            rel = "indicadores/" + os.path.basename(sub)
            if rel not in xps:
                xps.append(rel)
        self.xp.configure(values=[NO_XP] + xps)
        if self.xp.get() not in [NO_XP] + xps:
            self.xp.set(xps[0] if xps else NO_XP)
        self.drop.configure(values=[NO_DROP] + pngs)
        if self.drop.get() not in [NO_DROP] + pngs:
            self.drop.set(NO_DROP)

    def _suggest_model_path(self):
        name = ds.sanitize_name(self.name.get() or self.app.item or "item").lower()
        path = f"models/{slug(self.skill.get() or 'geral')}/mini_{name}.pt"
        self.model_path.delete(0, "end")
        self.model_path.insert(0, path)

    def _refresh_stats(self):
        item = self.app.item
        if not item:
            return self.stats_label.configure(text="Escolha ou crie um item no topo da janela.")
        s = ds.stats(item)
        per_class = "\n".join(f"   • {n}: {c} caixa(s)" for n, c in zip(s["classes"], s["per_class"])) or "   (nenhuma classe criada)"
        warn = ""
        if s["annotated"] < 50:
            warn = "\n⚠ Menos de 50 imagens anotadas — o modelo tende a ficar fraco (ideal: 100 a 300)."
        self.stats_label.configure(text=f"Item: {item}\nImagens anotadas: {s['annotated']} de {s['total']} "
                                        f"(sem objetos: {s['negatives']})\n{per_class}{warn}")

    # ------------------------------------------------------------------ log / progresso (thread-safe)

    def _ui(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            pass

    def log(self, message: str):
        def write():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self._ui(write)

    def _on_progress(self, fraction: float, text: str):
        def update():
            self.progress.set(fraction)
            self.progress_label.configure(text=text or f"{fraction * 100:.0f}%", text_color="white")
        self._ui(update)

    # ------------------------------------------------------------------ validação

    def _collect(self) -> Dict:
        item = self.app.item
        if not item:
            raise ValueError("Escolha ou crie um item no topo da janela.")
        s = ds.stats(item)
        if not s["classes"]:
            raise ValueError("Crie ao menos uma classe na aba 'Anotar'.")
        if s["annotated"] < 2:
            raise ValueError("Anote pelo menos 2 imagens na aba 'Anotar'.")
        target = self.target.get().strip()
        if target not in s["classes"]:
            raise ValueError("Escolha a classe-alvo (o objeto que o bot deve clicar).")
        model_path = self.model_path.get().strip().replace("\\", "/")
        if not model_path.endswith(".pt") or model_path.startswith("/") or ":" in model_path:
            raise ValueError("O arquivo do modelo deve ser um caminho relativo terminado em .pt (ex.: models/mineracao/mini_iron.pt).")
        name = ds.sanitize_name(self.name.get())
        try:
            params = {"epochs": int(self.epochs.get()), "imgsz": int(self.imgsz.get()), "batch": int(self.batch.get())}
        except ValueError:
            raise ValueError("Épocas, imgsz e batch devem ser números inteiros.")
        xp = self.xp.get().strip()
        drop = self.drop.get().strip()
        return {
            "item": item, "target": target, "model_path": model_path, "name": name, "skill": self.skill.get().strip() or "Geral",
            "xp": "" if xp == NO_XP else xp, "drop": "" if drop == NO_DROP else drop,
            "base_model": self.base_model.get().strip() or "yolov8n.pt", **params,
        }

    def _confirm_overwrite(self, model_path: str) -> bool:
        return not os.path.exists(model_path) or messagebox.askyesno(
            "Modelo existente", f"{model_path} já existe e será substituído ao fim do treino.\nContinuar?")

    # ------------------------------------------------------------------ treino local

    def start_training(self):
        if self._training:
            return
        try:
            p = self._collect()
        except ValueError as e:
            return messagebox.showwarning("Treinar", str(e))
        if not self._confirm_overwrite(p["model_path"]):
            return
        self._stop = False
        self._training = True
        self.train_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.set(0)
        self.progress_label.configure(text="Carregando bibliotecas (pode levar ~30 s na primeira vez)…", text_color="white")
        threading.Thread(target=self._worker, args=(p,), daemon=True).start()

    def request_stop(self):
        self._stop = True
        self.stop_button.configure(state="disabled")
        self.log("[TREINO] Parada solicitada — termina no próximo lote.")

    def _worker(self, p: Dict):
        try:
            split = ds.prepare_split(p["item"])
            self.log(f"[DATASET] preparado: {split['n_train']} treino / {split['n_val']} validação")
            result = run_training(
                split["yaml"], p["model_path"], model=p["base_model"], imgsz=p["imgsz"], epochs=p["epochs"],
                batch=p["batch"], on_progress=self._on_progress, on_log=self.log, should_stop=lambda: self._stop,
            )
            if result["path"]:
                self._register_activity(p)
                self._on_progress(1.0, "Concluído e registrado no bot.")
            elif result["status"] == "stopped":
                self._on_progress(self.progress.get(), "Interrompido (nada foi instalado).")
        except Exception as e:  # noqa: BLE001 - qualquer falha vai para o log da tela
            self.log(f"[ERRO] {e}")
            self._on_progress(0, "Falhou — veja o log.")
        finally:
            self._ui(self._training_finished)

    def _training_finished(self):
        self._training = False
        self.train_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _register_activity(self, p: Dict):
        registry.register_activity(p["skill"], p["name"], p["model_path"], p["target"], p["xp"], p["drop"])
        self.log(f"[REGISTRO] '{p['skill']} > {p['name']}' registrado (alvo: {p['target']}, modelo: {p['model_path']}"
                 f"{', drop: ' + p['drop'] if p['drop'] else ''}). Reinicie o bot para ver na lista.")

    # ------------------------------------------------------------------ Colab

    def make_colab(self):
        try:
            p = self._collect()
            split = ds.prepare_split(p["item"])
            zip_path = ds.make_colab_zip(p["item"])
        except ValueError as e:
            return messagebox.showwarning("Colab", str(e))
        self.log(f"[COLAB] Pacote gerado: {zip_path} ({split['n_train']} treino / {split['n_val']} validação)")

        win = ctk.CTkToplevel(self)
        win.title("Treinar no Google Colab")
        win.geometry("760x420")
        box = ctk.CTkTextbox(win, wrap="none")
        box.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        box.insert("1.0", COLAB_TEXT.format(zip_name=os.path.basename(zip_path)))
        ctk.CTkLabel(win, text=f"Arquivo para enviar: {zip_path}", text_color="gray", wraplength=720).pack(padx=10)

        def copy():
            self.clipboard_clear()
            self.clipboard_append(box.get("1.0", "end"))
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(pady=8)
        ctk.CTkButton(row, text="Copiar roteiro", command=copy).pack(side="left", padx=4)
        ctk.CTkButton(row, text="Abrir a pasta do zip", command=lambda: os.startfile(os.path.dirname(zip_path))).pack(side="left", padx=4)
        win.after(200, win.focus)

    def install_external(self):
        try:
            p = self._collect()
        except ValueError as e:
            return messagebox.showwarning("Instalar", str(e))
        source = filedialog.askopenfilename(title="Escolha o best.pt baixado do Colab", filetypes=[("Modelo YOLO", "*.pt")])
        if not source or not self._confirm_overwrite(p["model_path"]):
            return
        self.log(f"[INSTALAR] Verificando {source}…")

        def work():
            try:
                from ultralytics import YOLO
                names = YOLO(source).names
                if p["target"] not in names.values():
                    raise ValueError(f"O modelo não tem a classe '{p['target']}'. Classes do modelo: {list(names.values())}")
                os.makedirs(os.path.dirname(p["model_path"]) or ".", exist_ok=True)
                shutil.copy2(source, p["model_path"])
                self.log(f"[INSTALAR] Modelo copiado para {p['model_path']} | classes: {names}")
                self._register_activity(p)
            except Exception as e:  # noqa: BLE001
                self.log(f"[ERRO] {e}")
        threading.Thread(target=work, daemon=True).start()
