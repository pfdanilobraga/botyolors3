import customtkinter as ctk
import datetime
from config.activities import ACTIVITIES

# Configuração global do tema
ctk.set_appearance_mode("Dark")  # Modos: "System" (padrão), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Temas: "blue" (padrão), "green", "dark-blue"

class BotInterface:
    def __init__(self, root, start_callback, stop_callback):
        self.root = root
        self.root.title("RS3 VisionBot Engine v2.0")
        self.root.geometry("550x650")
        self.root.resizable(False, False)
        
        # Variáveis de Controle
        self.skill_var = ctk.StringVar()
        self.target_var = ctk.StringVar()
        self.model_path_var = ctk.StringVar()
        self.xp_anchor_var = ctk.StringVar()

        self._setup_widgets(start_callback, stop_callback)
        self._set_default_values()

    def _setup_widgets(self, start_cmd, stop_cmd):
        # --- Frame Principal ---
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título
        title_label = ctk.CTkLabel(main_frame, text="Configurações do Bot", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(10, 20))

        # --- Frame de Configuração ---
        config_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        config_frame.pack(fill="x", padx=10, pady=5)
        config_frame.columnconfigure(1, weight=1)

        # Habilidade
        ctk.CTkLabel(config_frame, text="Habilidade:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", pady=10, padx=5)
        self.skill_combo = ctk.CTkOptionMenu(config_frame, variable=self.skill_var, values=list(ACTIVITIES.keys()), command=self._on_skill_change)
        self.skill_combo.grid(row=0, column=1, sticky="ew", pady=10, padx=5)

        # Alvo
        ctk.CTkLabel(config_frame, text="Alvo:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w", pady=10, padx=5)
        self.target_combo = ctk.CTkOptionMenu(config_frame, variable=self.target_var, command=self._on_target_change)
        self.target_combo.grid(row=1, column=1, sticky="ew", pady=10, padx=5)

        # Caminho do Modelo
        ctk.CTkLabel(config_frame, text="Modelo (.pt):", text_color="gray").grid(row=2, column=0, sticky="w", pady=10, padx=5)
        self.model_entry = ctk.CTkEntry(config_frame, textvariable=self.model_path_var, state="disabled")
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=10, padx=5)

        # Âncora de XP
        ctk.CTkLabel(config_frame, text="Ícone XP:", text_color="gray").grid(row=3, column=0, sticky="w", pady=10, padx=5)
        self.xp_entry = ctk.CTkEntry(config_frame, textvariable=self.xp_anchor_var, state="disabled")
        self.xp_entry.grid(row=3, column=1, sticky="ew", pady=10, padx=5)

        # --- Frame de Controles ---
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20)

        self.btn_start = ctk.CTkButton(button_frame, text="START BOT", fg_color="#2ecc71", hover_color="#27ae60", 
                                       text_color="white", font=ctk.CTkFont(weight="bold"), command=start_cmd)
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(button_frame, text="STOP BOT", fg_color="#e74c3c", hover_color="#c0392b", 
                                      text_color="white", font=ctk.CTkFont(weight="bold"), command=stop_cmd)
        self.btn_stop.pack(side="left", padx=10)

        # --- Console de Logs ---
        log_label = ctk.CTkLabel(main_frame, text="Console do Sistema", font=ctk.CTkFont(size=14, weight="bold"))
        log_label.pack(anchor="w", padx=10)

        self.log_area = ctk.CTkTextbox(main_frame, state='disabled', font=ctk.CTkFont(family="Consolas", size=12))
        self.log_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))

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