import time
import cv2
import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.capture import ScreenCapture
from vision.detector import Detector
from actuators.mouse import MouseActuator
from actuators.keyboard import KeyboardActuator
from engine.state_mining import MiningFSM

def run_bot():
    """
    Main orchestrator for the RS3 VisionBot.
    Captures real frames and feeds them into the State Machine.
    """
    print("\n" + "="*50)
    print("RS3 VisionBot Engine v2.0 - LIVE MODE")
    print("="*50)
    
    # 1. Initialize Components
    # Game window must be named "RuneScape"
    cap = ScreenCapture(window_title="RuneScape")
    
    # Initialize Vision and Actuators
    print("[Info] Carregando modelos e periféricos...")
    detector = Detector(model_path="models/best_rs3.pt")
    mouse = MouseActuator()
    keyboard = KeyboardActuator()
    
    # Initialize FSM with all modules
    fsm = MiningFSM(
        detector=detector,
        mouse=mouse,
        keyboard=keyboard,
        screencap=cap,
        target_class="Copper_rock" # Adjust if your YOLO class is different
    )
    
    print("\n[Status] Bot pronto e integrado!")
    print("[Instrução] Garanta que o RuneScape está visível e não minimizado.")
    print("[Controle] Pressione 'Ctrl + C' no terminal para parar o bot.\n")

    try:
        while True:
            # 2. Capture Real Frame
            frame = cap.get_frame()
            
            if frame is not None:
                # 3. Update Finite State Machine with real world data
                fsm.update(frame)
            else:
                # Window not found, wait and retry
                print(f"[{time.strftime('%X')}] [Erro] Não foi possível capturar a janela 'RuneScape'.")
                time.sleep(2)
            
            # 4. Tick Rate control
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%X')}] [Encerrado] Bot interrompido pelo usuário.")
    except Exception as e:
        print(f"\n[{time.strftime('%X')}] [Erro Fatal] Ocorreu uma exceção: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bot()
