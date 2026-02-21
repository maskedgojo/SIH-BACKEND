import numpy as np
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path="models/yolov8n.pt", device="cpu"):
        self.model = YOLO(model_path)  # device is selected at inference time

    def detect(self, frame: np.ndarray, conf: float = 0.25):
        results = self.model(frame, conf=conf, verbose=False, device="cpu")
        if not isinstance(results, (list, tuple)):
            results = [results]

        detections = []
        for r in results:
            if r.boxes is None: continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for i, b in enumerate(xyxy):
                xmin, ymin, xmax, ymax = map(float, b[:4])
                detections.append({
                    "class": self.model.names.get(cls_ids[i], str(cls_ids[i])),
                    "confidence": float(confs[i]),
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
                })
        return detections
