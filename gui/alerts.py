"""Avisos ao usuário quando a sessão pede atenção: sons, piscar na barra de tarefas e janela de aviso."""
import ctypes
import threading
import time
from ctypes import wintypes

import customtkinter as ctk

from engine.session import FROZEN, GOAL_REACHED, LEVEL_UP, LOST, TIME_UP

try:
    import winsound
except ImportError:  # fora do Windows: sem som
    winsound = None

# (frequência Hz, duração ms) de cada nota
_SOUNDS = {
    LEVEL_UP: [(880, 120), (1175, 180)],
    GOAL_REACHED: [(784, 160), (988, 160), (1175, 160), (1568, 400)] * 2,
    TIME_UP: [(988, 250), (784, 250), (659, 250), (523, 500)],
    LOST: [(1000, 300), (700, 300)] * 4,                    # alarme: o bot está perdido
    FROZEN: [(1400, 160), (700, 160)] * 6,                  # alarme rápido: a tela do jogo congelou
}

_FLASHW_ALL = 0x03
_FLASHW_TIMERNOFG = 0x0C  # pisca até a janela voltar ao primeiro plano


class _FLASHWINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND), ("dwFlags", wintypes.DWORD),
                ("uCount", wintypes.UINT), ("dwTimeout", wintypes.DWORD)]


def play_sound(kind: str) -> None:
    """Toca o som do evento sem travar a tela."""
    notes = _SOUNDS.get(kind)
    if not notes or winsound is None:
        return

    def _play():
        try:
            for freq, ms in notes:
                winsound.Beep(freq, ms)
                time.sleep(0.03)
        except RuntimeError:
            pass  # sem dispositivo de som

    threading.Thread(target=_play, daemon=True).start()


def flash_taskbar(root) -> None:
    """Faz o botão do bot piscar na barra de tarefas (útil quando o jogo está na frente)."""
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        info = _FLASHWINFO(ctypes.sizeof(_FLASHWINFO), hwnd, _FLASHW_ALL | _FLASHW_TIMERNOFG, 0, 0)
        ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))
    except (AttributeError, OSError):
        pass


def show_popup(root, title: str, message: str) -> None:
    """Janela de aviso por cima de tudo, com botão OK."""
    popup = ctk.CTkToplevel(root)
    popup.title(title)
    popup.geometry("420x170")
    popup.resizable(False, False)
    popup.attributes("-topmost", True)
    ctk.CTkLabel(popup, text=title, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 6))
    ctk.CTkLabel(popup, text=message, wraplength=380, justify="center").pack(padx=20, pady=4)
    ctk.CTkButton(popup, text="OK", width=100, command=popup.destroy).pack(pady=14)
    popup.after(150, popup.lift)


def notify(root, kind: str, title: str, message: str) -> None:
    """Aviso completo: som + (nas metas e no tempo esgotado) piscar na barra e janela. Chamar na thread da tela."""
    play_sound(kind)
    if kind != LEVEL_UP:
        flash_taskbar(root)
        show_popup(root, title, message)
