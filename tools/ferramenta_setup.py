import cv2
import mss
import numpy as np
import os
import time
from typing import Tuple

def run_setup_tool():
    """
    Captures the screen, allows the user to select a Region of Interest (ROI),
    saves the selection as a template, and prints the coordinates.
    """
    print("-" * 50)
    print("RS3 VisionBot - Ferramenta de Setup de UI")
    print("-" * 50)
    
    # 1. Start preparation time
    print("\n[Instrução] Prepare a janela do jogo. O script capturará em 3 segundos...")
    time.sleep(3)
    
    # 2. Capture screen using mss
    with mss.mss() as sct:
        # Use the primary monitor
        monitor = sct.monitors[1]
        print(f"[Info] Capturando monitor principal: {monitor['width']}x{monitor['height']}")
        
        screenshot = sct.grab(monitor)
        # Convert to numpy array (BGRA)
        img = np.array(screenshot)
        # Convert BGRA to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    print("\n[Instrução] Uma janela de captura foi aberta.")
    print("1. Clique e arraste para desenhar o retângulo sobre o indicador de mineração.")
    print("2. Aperte 'ENTER' ou 'ESPAÇO' para confirmar a seleção.")
    print("3. Aperte 'c' ou 'ESC' para cancelar.")
    
    # 3. Select ROI
    # showCrosshair=True helps with precision
    # fromCenter=False means we drag from a corner
    roi = cv2.selectROI("Selecione o Indicador de Mineracao", img, showCrosshair=True, fromCenter=False)
    
    # roi format: (x, y, w, h)
    x, y, w, h = roi
    
    if w > 0 and h > 0:
        # 4. Crop and Save
        crop_img = img[y:y+h, x:x+w]
        
        assets_dir = "assets"
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
            print(f"[Info] Pasta '{assets_dir}' criada.")
            
        output_path = os.path.join(assets_dir, "indicador_xp.png")
        cv2.imwrite(output_path, crop_img)
        
        print("\n" + "=" * 50)
        print("CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 50)
        print(f"Arquivo salvo em: {output_path}")
        print(f"Coordenadas ROI (x, y, w, h): ({x}, {y}, {w}, {h})")
        print(f"Coordenadas Absolutas (x1, y1, x2, y2): ({x}, {y}, {x+w}, {y+h})")
        print("=" * 50)
        print("\nUse esses valores no seu script de visão/validação para otimizar o processamento.")
    else:
        print("\n[Aviso] Seleção cancelada pelo usuário.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_setup_tool()
    except Exception as e:
        print(f"\n[Erro] Ocorreu um problema inesperado: {e}")
