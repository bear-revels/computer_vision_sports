from ultralytics import YOLO
import os

# Debugging: Print the current working directory
print("Current working directory:", os.getcwd())

# Load the YOLOv8 model (you can use 'yolov8s.pt' or another model version)
model = YOLO('yolov8x.pt')

# Train the model with the new dataset
model.train(data='datasets/people_&_ball/data.yaml', epochs=50, imgsz=640, batch=16)