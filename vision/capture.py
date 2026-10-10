import ctypes
import time
from ctypes import wintypes
import mss
import numpy as np
import pygetwindow as gw
import cv2
import threading
from typing import Optional, Tuple, Dict

# --- Win32 (via ctypes, sem dependência extra) ---
_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
_kernel32 = ctypes.windll.kernel32 if hasattr(ctypes, "windll") else None

_SW_RESTORE = 9
_VK_MENU = 0x12  # Alt
_KEYEVENTF_KEYUP = 0x0002
_GA_ROOT = 2

if _user32 is not None:
    _user32.WindowFromPoint.argtypes = [wintypes.POINT]
    _user32.WindowFromPoint.restype = wintypes.HWND
    _user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    _user32.GetAncestor.restype = wintypes.HWND


class ScreenCapture:
    """
    Handles high-speed screen capture and RuneScape 3 window detection.
    """

    def __init__(self, window_title: str = "RuneScape"):
        self.window_title = window_title
        self._thread_local = threading.local()
        self.window_region = None
        self._hwnd = None
        self._find_window()

    @property
    def sct(self):
        """
        Thread-safe access to mss instance.
        """
        if not hasattr(self._thread_local, "sct"):
            self._thread_local.sct = mss.mss()
        return self._thread_local.sct

    def _pick_window(self):
        """
        Escolhe a janela do jogo pelo título EXATO (case-insensitive). Títulos que apenas contêm
        "RuneScape" (abas de navegador, pastas do projeto) são ignorados de propósito: focar ou
        clicar numa dessas janelas seria perigoso. Sem janela exata, retorna None.
        """
        wanted = self.window_title.strip().lower()
        for win in gw.getWindowsWithTitle(self.window_title):
            if win.title.strip().lower() == wanted:
                return win
        return None

    def locate(self) -> bool:
        """Procura a janela do jogo sem restaurá-la se estiver minimizada."""
        return self._find_window(restore=False)

    def _find_window(self, restore: bool = True) -> bool:
        """
        Locates the game window and updates the region.
        """
        try:
            win = self._pick_window()
            if win:
                # Ensure the window is visible and active
                if restore and win.isMinimized:
                    win.restore()

                self._hwnd = win._hWnd
                self.window_region = {
                    "top": win.top,
                    "left": win.left,
                    "width": win.width,
                    "height": win.height
                }
                return True
        except Exception as e:
            print(f"Error finding window: {e}")

        self._hwnd = None
        self.window_region = None
        return False

    def is_focused(self) -> bool:
        """True se a janela do jogo é a janela em primeiro plano."""
        return bool(self._hwnd) and _user32.GetForegroundWindow() == self._hwnd

    def is_visible(self) -> bool:
        """
        True se a janela do jogo está à vista, sem nada por cima (inclusive outras janelas do
        projeto): confere 9 pontos internos da janela. Não depende de ela estar em foco.
        """
        if _user32 is None or not self._hwnd or not self.window_region:
            return False
        if not _user32.IsWindow(self._hwnd) or _user32.IsIconic(self._hwnd):
            return False

        region = self.window_region
        for fx in (0.2, 0.5, 0.8):
            for fy in (0.2, 0.5, 0.8):
                point = wintypes.POINT(int(region["left"] + region["width"] * fx),
                                       int(region["top"] + region["height"] * fy))
                top = _user32.WindowFromPoint(point)
                if not top or _user32.GetAncestor(top, _GA_ROOT) != self._hwnd:
                    return False
        return True

    def focus(self, timeout: float = 1.0) -> bool:
        """
        Traz a janela do jogo para o primeiro plano (restaurando se minimizada).
        Retorna True se, ao final, o jogo estiver em foco. Barato quando já está em foco.
        """
        if _user32 is None:
            return False

        try:
            if not self._hwnd or not _user32.IsWindow(self._hwnd):
                if not self._find_window():
                    return False

            if self.is_focused():
                return True

            if _user32.IsIconic(self._hwnd):
                _user32.ShowWindow(self._hwnd, _SW_RESTORE)
                time.sleep(0.3)
                # A janela pode reaparecer em outra posição: atualiza a região de captura
                self._find_window()

            deadline = time.time() + timeout
            # 1ª tentativa: anexar a fila de entrada à da janela em primeiro plano
            self._force_foreground(use_alt_key=False)
            while time.time() < deadline and not self.is_focused():
                time.sleep(0.05)

            # 2ª tentativa: o truque da tecla Alt libera o bloqueio de foreground do Windows
            if not self.is_focused():
                self._force_foreground(use_alt_key=True)
                while time.time() < deadline + 0.5 and not self.is_focused():
                    time.sleep(0.05)

            return self.is_focused()
        except Exception as e:
            print(f"Error focusing window: {e}")
            return False

    def _force_foreground(self, use_alt_key: bool):
        fg_hwnd = _user32.GetForegroundWindow()
        fg_thread = _user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
        cur_thread = _kernel32.GetCurrentThreadId()
        attached = bool(fg_thread) and fg_thread != cur_thread

        if use_alt_key:
            _user32.keybd_event(_VK_MENU, 0, 0, 0)
            _user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
        if attached:
            _user32.AttachThreadInput(cur_thread, fg_thread, True)
        try:
            _user32.BringWindowToTop(self._hwnd)
            _user32.SetForegroundWindow(self._hwnd)
        finally:
            if attached:
                _user32.AttachThreadInput(cur_thread, fg_thread, False)

    def _refresh_region(self):
        """Acompanha a janela caso ela tenha sido movida ou redimensionada."""
        if _user32 is None or not self._hwnd:
            return
        if not _user32.IsWindow(self._hwnd) or _user32.IsIconic(self._hwnd):
            return
        rect = wintypes.RECT()
        if _user32.GetWindowRect(self._hwnd, ctypes.byref(rect)):
            width, height = rect.right - rect.left, rect.bottom - rect.top
            if width > 0 and height > 0:
                self.window_region = {
                    "top": rect.top,
                    "left": rect.left,
                    "width": width,
                    "height": height
                }

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Captures a frame from the defined window region.
        """
        if not self.window_region and not self._find_window():
            return None
        self._refresh_region()

        try:
            screenshot = self.sct.grab(self.window_region)
            # Convert mss screenshot to numpy array (BGRA)
            frame = np.array(screenshot)
            # Convert BGRA to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            self.window_region = None # Force re-search on next call
            return None

if __name__ == "__main__":
    # Test block
    print("Testing screen capture...")
    cap = ScreenCapture()
    print(f"Focus -> {cap.focus()}")
    frame = cap.get_frame()
    if frame is not None:
        print(f"Captured frame shape: {frame.shape}")
        cv2.imshow("Test Capture", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print("Could not find window or capture frame.")
