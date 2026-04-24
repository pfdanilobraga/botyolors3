import mss
import numpy as np
import pygetwindow as gw
import cv2
import threading
from typing import Optional, Tuple, Dict

class ScreenCapture:
    """
    Handles high-speed screen capture and RuneScape 3 window detection.
    """
    
    def __init__(self, window_title: str = "RuneScape"):
        self.window_title = window_title
        self._thread_local = threading.local()
        self.window_region = None
        self._find_window()

    @property
    def sct(self):
        """
        Thread-safe access to mss instance.
        """
        if not hasattr(self._thread_local, "sct"):
            self._thread_local.sct = mss.mss()
        return self._thread_local.sct

    def _find_window(self) -> bool:
        """
        Locates the game window and updates the region.
        """
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                win = windows[0]
                # Ensure the window is visible and active
                if win.isMinimized:
                    win.restore()
                
                self.window_region = {
                    "top": win.top,
                    "left": win.left,
                    "width": win.width,
                    "height": win.height
                }
                return True
        except Exception as e:
            print(f"Error finding window: {e}")
        
        self.window_region = None
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Captures a frame from the defined window region.
        """
        if not self.window_region and not self._find_window():
            return None
            
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
    frame = cap.get_frame()
    if frame is not None:
        print(f"Captured frame shape: {frame.shape}")
        cv2.imshow("Test Capture", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print("Could not find window or capture frame.")
