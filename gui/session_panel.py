"""Painel "Sessão & Metas": tempo de bot, meta de nível, leitura do XP na aba Skills e acompanhamento ao vivo."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import customtkinter as ctk

from engine.session import format_duration, format_eta, format_number
from engine.skill_goals import MAX_LEVEL, ProgressStore, level_for_xp, parse_decimal, parse_int

_DASH = "—"
MIN_INTERVAL_MIN = 1.0   # menos que isso seria apertar a tecla da aba o tempo todo


@dataclass
class SessionSettings:
    """O que o usuário configurou no painel, já validado."""
    duration_s: float            # 0 = sem limite de tempo
    target_level: Optional[int]  # meta (None: sem meta)
    stop_on_goal: bool
    auto_read: bool              # lê o XP real na aba Skills
    interval_s: float            # tempo entre leituras (só vale com auto_read)
    hotkey: str                  # tecla que abre/fecha a aba Skills
    level: Optional[int]         # nível digitado (ponto de partida manual; None: não informado)
    xp: Optional[int]            # XP digitado (None: usa o XP base do nível)
    safety: bool = True          # para o bot se ficar perdido (muito tempo sem ver rochas)
    lost_after_s: float = 300.0  # tempo sem ver rochas até considerar o bot perdido
    logout: bool = True          # e desloga do jogo (precisa de assets/logout_botao.png)


def _ago(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"há {seconds}s"
    return f"há {seconds // 60}min {seconds % 60:02d}s"


class SessionPanel(ctk.CTkFrame):
    def __init__(self, master, store: ProgressStore, on_sync: Callable[[], None], on_read_now: Callable[[], None]):
        super().__init__(master)
        self.store = store
        self._skill = ""
        self._target = ""

        self.hours_var = ctk.StringVar(value="0")
        self.minutes_var = ctk.StringVar(value="0")
        self.goal_var = ctk.StringVar()
        self.auto_read_var = ctk.BooleanVar(value=True)
        self.interval_var = ctk.StringVar(value="3")
        self.hotkey_var = ctk.StringVar(value="h")
        self.level_var = ctk.StringVar()
        self.xp_var = ctk.StringVar()
        self.stop_on_goal_var = ctk.BooleanVar(value=True)
        self.safety_var = ctk.BooleanVar(value=True)
        self.lost_minutes_var = ctk.StringVar(value="5")
        self.logout_var = ctk.BooleanVar(value=True)

        self._build_inputs(on_sync, on_read_now)
        self._build_status()
        self.render(None)

    # --- construção ---

    def _entry(self, parent, var, width=70):
        return ctk.CTkEntry(parent, textvariable=var, width=width)

    def _build_inputs(self, on_sync, on_read_now):
        ctk.CTkLabel(self, text="Sessão & Metas", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 4))
        self.context_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.context_label.pack()

        # Tempo de bot
        time_row = ctk.CTkFrame(self, fg_color="transparent")
        time_row.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(time_row, text="Tempo de bot:", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self._entry(time_row, self.hours_var, 50).pack(side="left", padx=(8, 3))
        ctk.CTkLabel(time_row, text="h").pack(side="left")
        self._entry(time_row, self.minutes_var, 50).pack(side="left", padx=(8, 3))
        ctk.CTkLabel(time_row, text="min").pack(side="left")
        ctk.CTkLabel(self, text="0 h 0 min = sem limite de tempo", text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)

        # Meta
        goal_row = ctk.CTkFrame(self, fg_color="transparent")
        goal_row.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(goal_row, text="Meta (nível):", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self._entry(goal_row, self.goal_var, 60).pack(side="left", padx=(8, 0))
        ctk.CTkCheckBox(self, text="Parar o bot ao atingir a meta", variable=self.stop_on_goal_var,
                        fg_color="#2ecc71").pack(anchor="w", padx=16, pady=(6, 2))

        # Segurança: perdido / morto
        safety_box = ctk.CTkFrame(self)
        safety_box.pack(fill="x", padx=14, pady=(8, 2))
        ctk.CTkCheckBox(safety_box, text="Parar se ficar perdido ou o jogo travar", variable=self.safety_var,
                        font=ctk.CTkFont(weight="bold"), fg_color="#e74c3c").pack(anchor="w", padx=10, pady=(8, 2))
        lost_row = ctk.CTkFrame(safety_box, fg_color="transparent")
        lost_row.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(lost_row, text="sem ver rochas por").pack(side="left")
        self._entry(lost_row, self.lost_minutes_var, 45).pack(side="left", padx=4)
        ctk.CTkLabel(lost_row, text="min").pack(side="left")
        ctk.CTkCheckBox(safety_box, text="e deslogar do jogo", variable=self.logout_var,
                        fg_color="#e74c3c").pack(anchor="w", padx=10, pady=(2, 8))

        # Leitura do XP real (aba Skills)
        read_box = ctk.CTkFrame(self)
        read_box.pack(fill="x", padx=14, pady=(8, 2))
        ctk.CTkCheckBox(read_box, text="Ler o XP real na aba Skills", variable=self.auto_read_var,
                        font=ctk.CTkFont(weight="bold"), fg_color="#3498db").pack(anchor="w", padx=10, pady=(8, 2))
        interval_row = ctk.CTkFrame(read_box, fg_color="transparent")
        interval_row.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(interval_row, text="a cada ~").pack(side="left")
        self._entry(interval_row, self.interval_var, 50).pack(side="left", padx=4)
        ctk.CTkLabel(interval_row, text="min   tecla da aba:").pack(side="left")
        self._entry(interval_row, self.hotkey_var, 45).pack(side="left", padx=4)
        ctk.CTkButton(read_box, text="Ler XP agora", fg_color="#16a085", hover_color="#1abc9c",
                      command=on_read_now).pack(fill="x", padx=10, pady=(4, 10))

        # Ponto de partida / correção manual
        manual = ctk.CTkFrame(self, fg_color="transparent")
        manual.pack(fill="x", padx=14, pady=(8, 0))
        manual.columnconfigure(1, weight=1)
        for i, (label, var) in enumerate([("Nível atual:", self.level_var), ("XP atual:", self.xp_var)]):
            ctk.CTkLabel(manual, text=label).grid(row=i, column=0, sticky="w", pady=2)
            self._entry(manual, var, 110).grid(row=i, column=1, sticky="e", pady=2)
        ctk.CTkButton(self, text="Corrigir nível/XP à mão", fg_color="gray35", hover_color="gray45",
                      command=on_sync).pack(fill="x", padx=14, pady=(4, 2))
        ctk.CTkLabel(self, text="Nível/XP à mão são opcionais: servem de ponto de partida se a leitura da "
                                "aba Skills estiver desligada ou falhar.", text_color="gray", wraplength=360,
                     justify="left", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16, pady=(2, 6))

    def _build_status(self):
        ctk.CTkFrame(self, height=2, fg_color="gray30").pack(fill="x", padx=14, pady=4)
        self.state_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(weight="bold"))
        self.state_label.pack(pady=(4, 2))

        box = ctk.CTkFrame(self, fg_color="transparent")
        box.pack(fill="x", padx=14)
        box.columnconfigure(1, weight=1)
        self._values: Dict[str, ctk.CTkLabel] = {}
        for i, (key, caption) in enumerate([("time", "Tempo"), ("level", "Nível"), ("xp", "XP atual"),
                                            ("gained", "XP ganho"), ("to_go", "Faltam"), ("rate", "XP/hora"),
                                            ("eta", "Previsão da meta"), ("read", "Última leitura")]):
            ctk.CTkLabel(box, text=caption, text_color="gray").grid(row=i, column=0, sticky="w", pady=2)
            self._values[key] = ctk.CTkLabel(box, text=_DASH, anchor="e")
            self._values[key].grid(row=i, column=1, sticky="e", pady=2)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(fill="x", padx=14, pady=(10, 2))
        self.progress_label = ctk.CTkLabel(self, text="", text_color="gray", wraplength=360)
        self.progress_label.pack(pady=(0, 10))

    # --- dados do painel ---

    def set_context(self, skill: str, target: str) -> None:
        """Chamado quando a habilidade/alvo muda: se a habilidade mudou, traz o progresso salvo dela."""
        changed = skill != self._skill
        self._skill, self._target = skill, target
        self.context_label.configure(text=f"{skill} · {target}" if skill else "")
        if changed:
            self.load_progress()

    def load_progress(self) -> None:
        """Preenche nível/XP/meta com o que ficou salvo da última sessão desta habilidade."""
        saved = self.store.skill(self._skill)
        xp = saved.get("xp")
        self.level_var.set(str(level_for_xp(xp)) if xp else "")
        self.xp_var.set(str(int(xp)) if xp else "")
        goal = saved.get("target_level")
        self.goal_var.set(str(goal) if goal else "")

    def hotkey(self) -> str:
        """Tecla da aba Skills digitada (h se estiver vazia)."""
        return self.hotkey_var.get().strip().lower() or "h"

    def show_manual_values(self, level: int, xp: float) -> None:
        """Mostra nos campos 'Nível atual / XP atual' um valor lido do jogo."""
        self.level_var.set(str(level))
        self.xp_var.set(str(int(xp)))

    def read(self) -> SessionSettings:
        """Lê e valida os campos. Levanta ValueError com a mensagem do problema."""
        hours = parse_int(self.hours_var.get(), "Horas") or 0
        minutes = parse_int(self.minutes_var.get(), "Minutos") or 0
        goal = parse_int(self.goal_var.get(), "Meta")
        level = parse_int(self.level_var.get(), "Nível atual")
        xp = parse_int(self.xp_var.get(), "XP atual")
        auto_read = self.auto_read_var.get()
        interval = parse_decimal(self.interval_var.get(), "Intervalo entre leituras") or 0.0
        hotkey = self.hotkey_var.get().strip().lower()

        if xp is not None:  # o XP manda: o nível é o dele
            level = level_for_xp(xp)
            self.level_var.set(str(level))
        if level is not None and not 1 <= level <= MAX_LEVEL:
            raise ValueError(f"Nível atual deve ficar entre 1 e {MAX_LEVEL}.")
        if auto_read:
            if interval < MIN_INTERVAL_MIN:
                raise ValueError(f"O intervalo entre leituras deve ser de pelo menos {MIN_INTERVAL_MIN:g} min.")
            if not hotkey:
                raise ValueError("Informe a tecla que abre a aba Skills (ex.: h).")
        safety = self.safety_var.get()
        lost_minutes = parse_decimal(self.lost_minutes_var.get(), "Tempo sem rochas") or 0.0
        if safety and lost_minutes < 1:
            raise ValueError("O tempo sem ver rochas para considerar o bot perdido deve ser de pelo menos 1 min.")
        if goal is not None:
            if not 2 <= goal <= MAX_LEVEL:
                raise ValueError(f"A meta deve ficar entre 2 e {MAX_LEVEL}.")
            if level is None and not auto_read:
                raise ValueError("Informe o nível (ou o XP) atual, ou ligue a leitura da aba Skills, para acompanhar a meta.")
            if level is not None and goal <= level:
                raise ValueError(f"A meta (nível {goal}) precisa ser maior que o nível atual ({level}). "
                                 "Defina uma nova meta.")
        return SessionSettings(duration_s=hours * 3600 + minutes * 60, target_level=goal,
                               stop_on_goal=self.stop_on_goal_var.get(), auto_read=auto_read,
                               interval_s=interval * 60, hotkey=hotkey or "h", level=level, xp=xp,
                               safety=safety, lost_after_s=lost_minutes * 60, logout=self.logout_var.get())

    # --- status ao vivo ---

    def _clear(self, message: str = "") -> None:
        for label in self._values.values():
            label.configure(text=_DASH)
        self.progress_bar.set(0)
        self.progress_label.configure(text=message)

    def render(self, snap: Optional[dict]) -> None:
        """Mostra o estado da sessão (o dicionário de SessionMonitor.snapshot) ou o painel vazio."""
        if not snap or not snap.get("started"):
            self.state_label.configure(text="● Parado", text_color="gray")
            self._clear()
            return

        goal = snap.get("goal")
        if goal and goal["reached"]:
            self.state_label.configure(text="✔ Meta atingida", text_color="#2ecc71")
        elif snap["running"]:
            self.state_label.configure(text="● Rodando", text_color="#2ecc71")
        else:
            self.state_label.configure(text="● Sessão encerrada", text_color="#e67e22")

        elapsed = format_duration(snap["elapsed"])
        if snap["duration"] > 0:
            time_text = f"{elapsed} / {format_duration(snap['duration'])}  (restam {format_duration(snap['remaining'])})"
        else:
            time_text = f"{elapsed}  (sem limite)"

        if not goal:
            self._clear("Sem acompanhamento de nível: ligue a leitura da aba Skills ou informe o nível atual.")
            self._values["time"].configure(text=time_text)
            return
        if not goal["has_reading"]:
            self._clear(f"Leitura falhou: {snap['last_failure']}" if snap["last_failure"]
                        else "Aguardando a 1ª leitura da aba Skills...")
            self._values["time"].configure(text=time_text)
            return

        v = self._values
        v["time"].configure(text=time_text)
        v["level"].configure(text=f"{goal['level']}" + (f" → {goal['target_level']}" if goal["target_level"] else ""))
        v["xp"].configure(text=format_number(goal["xp"]))
        v["gained"].configure(text=f"+{format_number(goal['gained'])}" if goal["readings"] > 1 else _DASH)
        v["to_go"].configure(text=f"{format_number(goal['xp_to_goal'])} XP" if goal["xp_to_goal"] is not None else _DASH)
        rate = goal["xp_per_hour"]
        v["rate"].configure(text=format_number(rate) if rate else _DASH)
        v["eta"].configure(text=format_eta(goal["eta"]) if rate else _DASH)

        read_text = f"{_ago(goal['read_age'])} ({goal['readings']}x)"
        if snap["check_gave_up"]:
            read_text = "desligada (falhou)"
        elif snap["check_failures"]:
            read_text += f" · falhas: {snap['check_failures']}"
        v["read"].configure(text=read_text)

        progress = goal["progress"]
        self.progress_bar.set(progress or 0)
        if snap["check_gave_up"]:
            self.progress_label.configure(
                text=f"Leitura automática desligada: {snap['last_failure']}. 'Ler XP agora' tenta de novo.")
        elif progress is None:
            self.progress_label.configure(text="Sem meta definida")
        else:
            self.progress_label.configure(text=f"{progress:.0%} do caminho até o nível {goal['target_level']}")
