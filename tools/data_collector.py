import sys
import os

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tkinter as tk
from tkinter import messagebox
import threading
import keyboard
import cv2
from datetime import datetime
from vision.capture import ScreenCapture

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RS3 Data Collector")
        self.root.geometry("300x200")
        self.root.attributes("-topmost", True) # Keep on top
        
        self.capture_engine = ScreenCapture()
        self.dataset_path = os.path.join("dataset", "raw_images")
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        self.label = tk.Label(root, text="RS3 Data Collector", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_capture = tk.Button(root, text="Capturar Frame (F9)", command=self.capture_frame, 
                                    height=2, width=20, bg="#4CAF50", fg="white")
        self.btn_capture.pack(pady=10)
        
        self.status_label = tk.Label(root, text="Pronto para coletar", fg="blue")
        self.status_label.pack(pady=5)
        
        # Start hotkey listener thread
        self.running = True
        self.hotkey_thread = threading.Thread(target=self.start_hotkey_listener, daemon=True)
        self.hotkey_thread.start()

    def start_hotkey_listener(self):
        """
        Listens for global hotkeys in the background.
        """
        keyboard.add_hotkey('f9', self.capture_frame)
        while self.running:
            time.sleep(0.1)

    def update_status(self, text, color="blue"):
        """
        Thread-safe method to update the status label.
        """
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def capture_frame(self):
        """
        Uses ScreenCapture to grab a frame and save it to disk.
        """
        try:
            frame = self.capture_engine.get_frame()
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"rs3_{timestamp}.jpg"
                filepath = os.path.join(self.dataset_path, filename)
                
                # Save as JPG with high quality
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                self.update_status(f"Salvo: {filename}", "green")
                print(f"Frame saved: {filepath}")
            else:
                self.update_status("Erro: Janela não encontrada", "red")
        except Exception as e:
            print(f"Capture failed: {e}")
            self.update_status(f"Erro na captura: {str(e)[:20]}", "red")

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
