"""Painel "Sessão & Metas": tempo de bot, nível atual/meta, XP por item e acompanhamento ao vivo."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import customtkinter as ctk

from engine.session import format_duration, format_eta, format_number
from engine.skill_goals import MAX_LEVEL, ProgressStore, level_for_xp, parse_decimal, parse_int

_DASH = "—"


@dataclass
class SessionSettings:
    """O que o usuário configurou no painel, já validado."""
    duration_s: float            # 0 = sem limite de tempo
    level: Optional[int]         # nível atual (None: sem acompanhamento de nível)
    xp: Optional[int]            # XP atual informado (None: usa o XP base do nível)
    target_level: Optional[int]  # meta (None: sem meta)
    xp_per_item: float
    stop_on_goal: bool


class SessionPanel(ctk.CTkFrame):
    def __init__(self, master, store: ProgressStore, on_sync: Callable[[], None]):
        super().__init__(master)
        self.store = store
        self._skill = ""
        self._target = ""

        self.hours_var = ctk.StringVar(value="0")
        self.minutes_var = ctk.StringVar(value="0")
        self.level_var = ctk.StringVar()
        self.xp_var = ctk.StringVar()
        self.goal_var = ctk.StringVar()
        self.xp_item_var = ctk.StringVar()
        self.stop_on_goal_var = ctk.BooleanVar(value=True)

        self._build_inputs(on_sync)
        self._build_status()
        self.render(None)

    # --- construção ---

    def _entry(self, parent, var, width=70):
        return ctk.CTkEntry(parent, textvariable=var, width=width)

    def _build_inputs(self, on_sync):
        ctk.CTkLabel(self, text="Sessão & Metas", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 4))
        self.context_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.context_label.pack()

        # Tempo de bot
        time_row = ctk.CTkFrame(self, fg_color="transparent")
        time_row.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(time_row, text="Tempo de bot:", font=ctk.CTkFont(weight="bold")).pack(side="left")
        self._entry(time_row, self.hours_var, 50).pack(side="left", padx=(8, 3))
        ctk.CTkLabel(time_row, text="h").pack(side="left")
        self._entry(time_row, self.minutes_var, 50).pack(side="left", padx=(8, 3))
        ctk.CTkLabel(time_row, text="min").pack(side="left")
        ctk.CTkLabel(self, text="0 h 0 min = sem limite de tempo", text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)

        # Nível e meta
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(10, 2))
        grid.columnconfigure(1, weight=1)
        rows = [("Nível atual:", self.level_var), ("XP atual (opcional):", self.xp_var),
                ("Meta (nível):", self.goal_var), ("XP por item:", self.xp_item_var)]
        for i, (label, var) in enumerate(rows):
            ctk.CTkLabel(grid, text=label).grid(row=i, column=0, sticky="w", pady=3)
            self._entry(grid, var, 110).grid(row=i, column=1, sticky="e", pady=3)

        ctk.CTkCheckBox(self, text="Parar o bot ao atingir a meta", variable=self.stop_on_goal_var,
                        fg_color="#2ecc71").pack(anchor="w", padx=16, pady=(6, 4))
        ctk.CTkButton(self, text="Corrigir nível/XP agora", fg_color="#16a085", hover_color="#1abc9c",
                      command=on_sync).pack(fill="x", padx=14, pady=(4, 2))
        ctk.CTkLabel(self, text="O XP é estimado (itens obtidos × XP por item). Informe o nível/XP reais "
                                "para corrigir quando quiser.", text_color="gray", wraplength=360,
                     justify="left", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16, pady=(2, 8))

    def _build_status(self):
        ctk.CTkFrame(self, height=2, fg_color="gray30").pack(fill="x", padx=14, pady=4)
        self.state_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(weight="bold"))
        self.state_label.pack(pady=(4, 2))

        box = ctk.CTkFrame(self, fg_color="transparent")
        box.pack(fill="x", padx=14)
        box.columnconfigure(1, weight=1)
        self._values: Dict[str, ctk.CTkLabel] = {}
        for i, (key, caption) in enumerate([("time", "Tempo"), ("level", "Nível"), ("gained", "XP ganho"),
                                            ("to_go", "Faltam"), ("rate", "XP/hora"), ("eta", "Previsão da meta"),
                                            ("items", "Itens obtidos")]):
            ctk.CTkLabel(box, text=caption, text_color="gray").grid(row=i, column=0, sticky="w", pady=2)
            self._values[key] = ctk.CTkLabel(box, text=_DASH, anchor="e")
            self._values[key].grid(row=i, column=1, sticky="e", pady=2)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(fill="x", padx=14, pady=(10, 2))
        self.progress_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.progress_label.pack(pady=(0, 10))

    # --- dados do painel ---

    def set_context(self, skill: str, target: str) -> None:
        """Chamado quando a habilidade/alvo muda: troca o XP por item e, se a habilidade mudou, o progresso salvo."""
        previous = (self._skill, self._target)
        self._skill, self._target = skill, target
        self.context_label.configure(text=f"{skill} · {target}" if skill else "")
        if skill != previous[0]:
            self.load_progress()
        if (skill, target) != previous:  # não sobrescreve o que o usuário digitou ao só atualizar a lista de alvos
            self.xp_item_var.set(f"{self.store.xp_per_item(skill, target):g}")

    def load_progress(self) -> None:
        """Preenche nível/XP/meta com o que ficou salvo da última sessão desta habilidade."""
        saved = self.store.skill(self._skill)
        xp = saved.get("xp")
        self.level_var.set(str(level_for_xp(xp)) if xp else "")
        self.xp_var.set(str(int(xp)) if xp else "")
        goal = saved.get("target_level")
        self.goal_var.set(str(goal) if goal else "")

    def show_level_xp(self, level: int, xp: float) -> None:
        self.level_var.set(str(level))
        self.xp_var.set(str(int(xp)))

    def read(self) -> SessionSettings:
        """Lê e valida os campos. Levanta ValueError com a mensagem do problema."""
        hours = parse_int(self.hours_var.get(), "Horas") or 0
        minutes = parse_int(self.minutes_var.get(), "Minutos") or 0
        level = parse_int(self.level_var.get(), "Nível atual")
        xp = parse_int(self.xp_var.get(), "XP atual")
        goal = parse_int(self.goal_var.get(), "Meta")
        xp_per_item = parse_decimal(self.xp_item_var.get(), "XP por item") or 0.0

        if xp is not None:  # o XP manda: o nível é o dele
            level = level_for_xp(xp)
            self.level_var.set(str(level))
        if level is not None and not 1 <= level <= MAX_LEVEL:
            raise ValueError(f"Nível atual deve ficar entre 1 e {MAX_LEVEL}.")
        if goal is not None:
            if level is None:
                raise ValueError("Informe o nível (ou o XP) atual para acompanhar a meta.")
            if not 2 <= goal <= MAX_LEVEL:
                raise ValueError(f"A meta deve ficar entre 2 e {MAX_LEVEL}.")
            if goal <= level:
                raise ValueError(f"A meta (nível {goal}) precisa ser maior que o nível atual ({level}). "
                                 "Defina uma nova meta.")
        return SessionSettings(duration_s=hours * 3600 + minutes * 60, level=level, xp=xp, target_level=goal,
                               xp_per_item=xp_per_item, stop_on_goal=self.stop_on_goal_var.get())

    # --- status ao vivo ---

    def render(self, snap: Optional[dict]) -> None:
        """Mostra o estado da sessão (o dicionário de SessionMonitor.snapshot) ou o painel vazio."""
        if not snap or not snap.get("started"):
            self.state_label.configure(text="● Parado", text_color="gray")
            for label in self._values.values():
                label.configure(text=_DASH)
            self.progress_bar.set(0)
            self.progress_label.configure(text="")
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
        self._values["time"].configure(text=time_text)

        if not goal:
            for key in ("level", "gained", "to_go", "rate", "eta", "items"):
                self._values[key].configure(text=_DASH)
            self.progress_bar.set(0)
            self.progress_label.configure(text="Sem acompanhamento de nível (informe o nível atual).")
            return

        level_text = f"{goal['level']}" + (f" → {goal['target_level']}" if goal["target_level"] else "") + "  (estimado)"
        self._values["level"].configure(text=level_text)
        self._values["gained"].configure(text=f"+{format_number(goal['gained'])}" if goal["xp_per_item"] > 0 else _DASH)
        self._values["to_go"].configure(
            text=f"{format_number(goal['xp_to_goal'])} XP" if goal["xp_to_goal"] is not None else _DASH)
        has_rate = goal["xp_per_hour"] > 0 and snap["elapsed"] >= 60
        self._values["rate"].configure(text=format_number(goal["xp_per_hour"]) if has_rate else _DASH)
        self._values["eta"].configure(text=format_eta(goal["eta"]) if has_rate else _DASH)
        self._values["items"].configure(text=str(goal["items"]))

        progress = goal["progress"]
        self.progress_bar.set(progress or 0)
        if progress is None:
            self.progress_label.configure(text="Sem meta definida")
        else:
            self.progress_label.configure(text=f"{progress:.0%} do caminho até o nível {goal['target_level']}")
        if goal["xp_per_item"] <= 0:
            self.progress_label.configure(text="Informe o XP por item para estimar o XP ganho.")
