from ultralytics import YOLO

model = YOLO('models/yolov8x') # n= nano, s= small, m= medium, l= large, x= extra large

results = model.predict('input/short_input.mp4', save=True)
print(results[0])
print('==============================================')
for box in results[0].boxes:
    print(box)