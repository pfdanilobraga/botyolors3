import customtkinter as ctk
import datetime
from config.activities import ACTIVITIES

# Configuração global do tema
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BotInterface:
    def __init__(self, root, start_callback, stop_callback):
        self.root = root
        self.root.title("RS3 VisionBot Engine v2.0")
        self.root.geometry("550x700")
        self.root.resizable(False, False)
        
        # --- Variáveis de Controle ---
        self.use_yolo_var = ctk.BooleanVar(value=True) # Controla se a IA está ativa
        self.skill_var = ctk.StringVar()
        self.target_var = ctk.StringVar()
        self.model_path_var = ctk.StringVar()
        self.xp_anchor_var = ctk.StringVar()
        
        # Variáveis de Sobrevivência
        self.use_prayer_var = ctk.BooleanVar(value=False)
        self.potion_key_var = ctk.StringVar(value="2")
        self.prayer_key_var = ctk.StringVar(value="3")
         # Variáveis de Máscara
        self.use_mask_var = ctk.BooleanVar(value=False)
        self.mask_key_var = ctk.StringVar(value="1")

        self._setup_widgets(start_callback, stop_callback)
        self._set_default_values()

    def _setup_widgets(self, start_cmd, stop_cmd):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ctk.CTkLabel(main_frame, text="Configurações do Bot", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(10, 10))

        # ==========================================
        # SEÇÃO: CONTROLE DE IA (YOLO)
        # ==========================================
        yolo_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        yolo_frame.pack(fill="x", padx=10, pady=5)

        self.yolo_checkbox = ctk.CTkCheckBox(
            yolo_frame, 
            text="Ativar Inteligência Artificial (Modelos YOLO)", 
            variable=self.use_yolo_var,
            command=self._toggle_yolo_fields,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#3498db"
        )
        self.yolo_checkbox.pack(side="left", pady=5)

        # --- Container dos Campos de IA ---
        self.ia_config_frame = ctk.CTkFrame(main_frame)
        self.ia_config_frame.pack(fill="x", padx=10, pady=5)
        self.ia_config_frame.columnconfigure(1, weight=1)

        # Habilidade
        ctk.CTkLabel(self.ia_config_frame, text="Habilidade:").grid(row=0, column=0, sticky="w", pady=5, padx=10)
        self.skill_combo = ctk.CTkOptionMenu(self.ia_config_frame, variable=self.skill_var, values=list(ACTIVITIES.keys()), command=self._on_skill_change)
        self.skill_combo.grid(row=0, column=1, sticky="ew", pady=5, padx=10)

        # Alvo
        ctk.CTkLabel(self.ia_config_frame, text="Alvo:").grid(row=1, column=0, sticky="w", pady=5, padx=10)
        self.target_combo = ctk.CTkOptionMenu(self.ia_config_frame, variable=self.target_var, command=self._on_target_change)
        self.target_combo.grid(row=1, column=1, sticky="ew", pady=5, padx=10)

        # Modelo
        ctk.CTkLabel(self.ia_config_frame, text="Modelo (.pt):", text_color="gray").grid(row=2, column=0, sticky="w", pady=5, padx=10)
        self.model_entry = ctk.CTkEntry(self.ia_config_frame, textvariable=self.model_path_var, state="disabled")
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=10)

        # ==========================================
        # SEÇÃO: SOBREVIVÊNCIA (Layout Ajustado)
        # ==========================================
        prayer_group = ctk.CTkFrame(main_frame)
        prayer_group.pack(fill="x", padx=10, pady=10)
        
        # Configuramos as colunas para os inputs ficarem alinhados
        prayer_group.columnconfigure(1, weight=1)

        # 1. Checkbox Principal (Ocupa as duas colunas)
        self.prayer_checkbox = ctk.CTkCheckBox(
            prayer_group, text="Auto-Restore (Potion/Prayer)", 
            variable=self.use_prayer_var, command=self._toggle_prayer_entry,
            font=ctk.CTkFont(weight="bold")
        )
        self.prayer_checkbox.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        # 2. Linha da Hotkey Potion
        ctk.CTkLabel(prayer_group, text="Hotkey Potion:", text_color="gray").grid(row=1, column=0, padx=(20, 5), pady=5, sticky="w")
        self.potion_entry = ctk.CTkEntry(prayer_group, textvariable=self.potion_key_var, width=60, state="disabled")
        self.potion_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # 3. Linha da Hotkey Auto-Restore (Prayer)
        ctk.CTkLabel(prayer_group, text="Hotkey Prayer:", text_color="gray").grid(row=2, column=0, padx=(20, 5), pady=5, sticky="w")
        self.prayer_entry = ctk.CTkEntry(prayer_group, textvariable=self.prayer_key_var, width=60, state="disabled")
        self.prayer_entry.grid(row=2, column=1, padx=10, pady=(5, 10), sticky="w")
        # ==========================================
        # SEÇÃO: CRYSTAL MASK (Timer Service)
        # ==========================================
        mask_group = ctk.CTkFrame(main_frame)
        mask_group.pack(fill="x", padx=10, pady=10)
        
        mask_group.columnconfigure(1, weight=1)

        # 1. Checkbox para Ativar/Desativar
        self.mask_checkbox = ctk.CTkCheckBox(
            mask_group, text="Auto Crystal Mask (4:50 min)", 
            variable=self.use_mask_var,
            font=ctk.CTkFont(weight="bold")
        )
        self.mask_checkbox.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        # 2. Linha da Hotkey Mask
        ctk.CTkLabel(mask_group, text="Hotkey Mask:", text_color="gray").grid(row=1, column=0, padx=(20, 5), pady=5, sticky="w")
        self.mask_entry = ctk.CTkEntry(mask_group, textvariable=self.mask_key_var, width=60, state="disabled")
        self.mask_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # --- Botões ---
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=10)

        self.btn_start = ctk.CTkButton(button_frame, text="START BOT", fg_color="#2ecc71", command=start_cmd, font=ctk.CTkFont(weight="bold"))
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(button_frame, text="STOP BOT", fg_color="#e74c3c", command=stop_cmd, font=ctk.CTkFont(weight="bold"))
        self.btn_stop.pack(side="left", padx=10)

        # --- Logs ---
        self.log_area = ctk.CTkTextbox(main_frame, state='disabled', height=150)
        self.log_area.pack(fill="both", expand=True, padx=10, pady=10)

    # --- Lógica de Interface ---

    def _toggle_yolo_fields(self):
        """Habilita ou desabilita os seletores de IA."""
        state = "normal" if self.use_yolo_var.get() else "disabled"
        self.skill_combo.configure(state=state)
        self.target_combo.configure(state=state)
        self.ia_config_frame.configure(fg_color="transparent" if state == "normal" else "#2b2b2b")

    def _toggle_prayer_entry(self):
        state = "normal" if self.use_prayer_var.get() else "disabled"
        self.potion_entry.configure(state=state)
        self.prayer_entry.configure(state=state)

    def _on_skill_change(self, value=None):
        skill = self.skill_var.get()
        if skill in ACTIVITIES:
            targets = list(ACTIVITIES[skill].keys())
            self.target_combo.configure(values=targets)
            self.target_var.set(targets[0])
            self._on_target_change()

    def _on_target_change(self, value=None):
        skill = self.skill_var.get()
        target = self.target_var.get()
        if skill in ACTIVITIES and target in ACTIVITIES[skill]:
            config = ACTIVITIES[skill][target]
            self.model_path_var.set(config["model"])
            self.xp_anchor_var.set(config["xp_validator"])

    def _set_default_values(self):
        if ACTIVITIES:
            first_skill = list(ACTIVITIES.keys())[0]
            self.skill_var.set(first_skill)
            self._on_skill_change()

    def log(self, message):
        self.log_area.configure(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_area.insert("end", f"[{timestamp}] {message}\n")
        self.log_area.see("end")
        self.log_area.configure(state='disabled')

  