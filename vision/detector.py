from ultralytics import YOLO
import numpy as np
import torch
import ultralytics
import os
from typing import List, Dict, Any

# Fix for PyTorch 2.6+ unpickling security restrictions when loading weights
try:
    torch.serialization.add_safe_globals(
        [
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.nn.tasks.ClassificationModel,
            ultralytics.nn.tasks.PoseModel,
            ultralytics.nn.tasks.SegmentationModel,
            ultralytics.nn.tasks.OBBModel,
        ]
    )
except Exception:
    pass


class Detector:
    """
    Handles YOLOv8 object detection.
    """

    def __init__(self, model_path: str = None):
        """
        Inicializa o Detector. Se model_path for None, o modelo não é carregado.
        """
        self.model = None
        self.current_path = None

        if model_path:
            self.update_model(model_path)

    def update_model(self, new_model_path: str) -> bool:
        """
        Carrega ou troca o modelo YOLO sem destruir a instância do Detector.
        Retorna True se o carregamento foi bem sucedido.
        """
        if not new_model_path:
            return False

        if os.path.exists(new_model_path):
            try:
                # Evita recarregar o mesmo modelo se já estiver na memória
                if self.current_path == new_model_path and self.model is not None:
                    return True

                self.model = YOLO(new_model_path)
                self.current_path = new_model_path
                print(f"[IA] Modelo carregado com sucesso: {new_model_path}")
                return True
            except Exception as e:
                print(f"[ERRO IA] Falha ao carregar pesos: {e}")
                return False
        else:
            print(f"[ERRO IA] Arquivo do modelo não encontrado: {new_model_path}")
            return False

    def detect(
        self, frame: np.ndarray, conf_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO inference on a frame.
        Returns a list of detections: [{"class": "nome_da_classe", "x": 100, "y": 200, "conf": 0.9}]
        Where x and y are the center coordinates of the bounding box.
        """
        if self.model is None:
            print("[ERRO] Modelo não carregado")
            return []

        if frame is None:
            return []

        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Calculate center coordinates
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    detections.append(
                        {
                            "class": class_name,
                            "x": center_x,
                            "y": center_y,
                            "conf": conf,
                            "box": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                            ],  # Keep raw box just in case
                        }
                    )

        return detections
