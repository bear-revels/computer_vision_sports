import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = detection[:6]
                width = x2 - x1
                height = y2 - y1
                detections.append((int(x1), int(y1), int(width), int(height), float(score), int(class_id)))
        return detections