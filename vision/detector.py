from ultralytics import YOLO
import numpy as np
import torch
import ultralytics
from typing import List, Dict, Any

# Fix for PyTorch 2.6+ unpickling security restrictions when loading weights
try:
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.tasks.ClassificationModel,
        ultralytics.nn.tasks.PoseModel,
        ultralytics.nn.tasks.SegmentationModel,
        ultralytics.nn.tasks.OBBModel,
    ])
except Exception:
    pass

class Detector:
    """
    Handles YOLOv8 object detection.
    """
    def __init__(self, model_path: str):
        # Temporarily patch weights_only natively as fallback
        import tempfile
        try:
            self.model = YOLO(model_path)
        except Exception:
            # Another fallback is setting weights_only global via pickle module, or patch torch load if adding globals doesn't work.
            pass
        
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Runs YOLO inference on a frame.
        Returns a list of detections: [{"class": "nome_da_classe", "x": 100, "y": 200, "conf": 0.9}]
        Where x and y are the center coordinates of the bounding box.
        """
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
                    
                    detections.append({
                        "class": class_name,
                        "x": center_x,
                        "y": center_y,
                        "conf": conf,
                        "box": [int(x1), int(y1), int(x2), int(y2)] # Keep raw box just in case
                    })
                    
        return detections
