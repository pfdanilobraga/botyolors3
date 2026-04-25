import tkinter as tk
from tkinter import ttk, scrolledtext
import datetime
from config.activities import ACTIVITIES  # Importa o seu dicionário de atividades


class BotInterface:
    def __init__(self, root, start_callback, stop_callback):
        self.root = root
        self.root.title("RS3 VisionBot Engine v2.0")
        self.root.geometry("500x600")

        # Variáveis de Controle
        self.skill_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.xp_anchor_var = tk.StringVar()

        self._setup_widgets(start_callback, stop_callback)
        self._set_default_values()

    def _setup_widgets(self, start_cmd, stop_cmd):
        config_frame = ttk.LabelFrame(
            self.root, text=" Configurações de Atividade ", padding=10
        )
        config_frame.pack(fill="x", padx=10, pady=5)
        config_frame.columnconfigure(1, weight=1)

        # Habilidade (Pega as chaves do dicionário ACTIVITIES automaticamente)
        ttk.Label(config_frame, text="Habilidade:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.skill_combo = ttk.Combobox(
            config_frame, textvariable=self.skill_var, state="readonly"
        )

        # DINÂMICO: Transforma as chaves do dicionário em opções (Mineração, Pesca...)
        self.skill_combo["values"] = list(ACTIVITIES.keys())

        self.skill_combo.grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.skill_combo.bind("<<ComboboxSelected>>", self._on_skill_change)

        # Alvo (Será preenchido pelo _on_skill_change)
        ttk.Label(config_frame, text="Alvo:").grid(row=1, column=0, sticky="w", pady=2)
        self.target_combo = ttk.Combobox(
            config_frame, textvariable=self.target_var, state="readonly"
        )
        self.target_combo.grid(row=1, column=1, sticky="ew", pady=2, padx=5)
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_change)

        # Caminho do Modelo
        ttk.Label(config_frame, text="Modelo (.pt):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.model_entry = ttk.Entry(config_frame, textvariable=self.model_path_var)
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=2, padx=5)

        # Âncora de XP
        ttk.Label(config_frame, text="Ícone XP:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.xp_entry = ttk.Entry(config_frame, textvariable=self.xp_anchor_var)
        self.xp_entry.grid(row=3, column=1, sticky="ew", pady=2, padx=5)

        # ... (Restante dos botões e log_area permanecem iguais)
        # --- Frame de Controles ---
        button_frame = tk.Frame(self.root, pady=10)
        button_frame.pack()

        self.btn_start = tk.Button(
            button_frame,
            text="START BOT",
            bg="#2ecc71",
            fg="white",
            width=15,
            font=("Arial", 10, "bold"),
            command=start_cmd,
        )
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = tk.Button(
            button_frame,
            text="STOP BOT",
            bg="#e74c3c",
            fg="white",
            width=15,
            font=("Arial", 10, "bold"),
            command=stop_cmd,
        )
        self.btn_stop.pack(side="left", padx=5)

        log_frame = ttk.LabelFrame(self.root, text=" Log do Sistema ", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_area = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            height=15,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#dcdcdc",
        )
        self.log_area.pack(fill="both", expand=True)

    def _on_skill_change(self, event=None):
        """Atualiza a lista de alvos baseado na skill selecionada."""
        skill = self.skill_var.get()
        if skill in ACTIVITIES:
            # Pega as sub-chaves (ex: Copper, Iron, Lobster)
            targets = list(ACTIVITIES[skill].keys())
            self.target_combo["values"] = targets
            self.target_var.set(targets[0])  # Seleciona o primeiro alvo por padrão
            self._on_target_change()  # Atualiza os campos de modelo e XP

    def _on_target_change(self, event=None):
        """Preenche o modelo e o validador baseado no alvo específico."""
        skill = self.skill_var.get()
        target = self.target_var.get()

        if skill in ACTIVITIES and target in ACTIVITIES[skill]:
            config = ACTIVITIES[skill][target]
            self.model_path_var.set(config["model"])
            self.xp_anchor_var.set(config["xp_validator"])

    def _set_default_values(self):
        """Inicializa a GUI com a primeira skill disponível no dicionário."""
        if ACTIVITIES:
            first_skill = list(ACTIVITIES.keys())[0]
            self.skill_var.set(first_skill)
            self._on_skill_change()

    def log(self, message):
        self.log_area.configure(state="normal")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")
