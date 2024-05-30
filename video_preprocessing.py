import cv2
import numpy as np
import pytesseract
from sklearn.cluster import KMeans
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import multiprocessing as mp
from functools import partial

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # 'yolov8s.pt' is a small version of YOLOv8, you can choose other versions as well

# Initialize Deep SORT
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Parameters
n_trace_frames = 10  # Number of frames for the pink tracing line

def classify_team_or_referee(crop):
    # Resize the crop to a smaller size for faster processing
    crop_resized = cv2.resize(crop, (50, 50), interpolation=cv2.INTER_AREA)
    
    # Convert the image from BGR to RGB
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = crop_rgb.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    
    # Determine the team based on the dominant colors
    # Assuming team1 is red and team2 is blue
    team1_color = np.array([255, 0, 0])  # Pure red in RGB
    team2_color = np.array([0, 0, 255])  # Pure blue in RGB
    
    # Calculate the distance of dominant colors to the team colors
    distances_to_team1 = np.linalg.norm(dominant_colors - team1_color, axis=1)
    distances_to_team2 = np.linalg.norm(dominant_colors - team2_color, axis=1)
    
    # Get the minimum distance to classify the color
    min_distance_team1 = np.min(distances_to_team1)
    min_distance_team2 = np.min(distances_to_team2)
    
    if min_distance_team1 < min_distance_team2:
        return "team1"  # Red team
    else:
        return "team2"  # Blue team

def extract_jersey_number(crop):
    # Convert the image to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Apply some preprocessing to improve OCR accuracy
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use pytesseract to extract text from the processed image
    config = "--psm 6"  # Set the Page Segmentation Mode to 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(gray, config=config)
    
    # Use a regular expression to extract numbers from the recognized text
    import re
    numbers = re.findall(r'\d+', text)
    
    if numbers:
        return numbers[0]  # Return the first number found
    else:
        return ""  # Return an empty string if no number is found

def process_frame(frame, trace_points, n_trace_frames):
    results = model(frame)  # Run YOLOv8 inference on the frame
    detections = results[0].boxes  # Access the detections
    
    if len(detections) == 0:
        print("No detections in this frame")
        return frame  # Return the original frame if no detections

    # Prepare detections for Deep SORT
    detection_data = []
    for result in detections:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = float(result.conf)
        class_id = int(result.cls)
        detection_data.append([x1, y1, x2-x1, y2-y1, conf, class_id])
    
    tracks = tracker.update_tracks(detection_data, frame=frame)  # Update Deep SORT with the detections

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()  # Get the bounding box in top-left, bottom-right format
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(track.class_id)
        crop = frame[y1:y2, x1:x2]
        
        if class_id == 32:  # Assuming '32' is the class ID for soccer ball
            trace_points.append((x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2))
            if len(trace_points) > n_trace_frames:
                trace_points.pop(0)
            cv2.circle(frame, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), 5, (0, 255, 0), -1)
            cv2.polylines(frame, [np.array(trace_points, np.int32)], False, (255, 0, 255), 2)
        
        elif class_id in [0, 1]:  # Assuming '0' and '1' are the class IDs for players
            team_color = classify_team_or_referee(crop)
            jersey_number = extract_jersey_number(crop)
            color = (0, 0, 255) if team_color == "red" else (255, 0, 0)
            cv2.circle(frame, (x1 + (x2 - x1) // 2, y2), 10, color, -1)
            cv2.putText(frame, jersey_number, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif class_id == 2:  # Assuming '2' is the class ID for referees
            cv2.circle(frame, (x1 + (x2 - x1) // 2, y2), 10, (0, 255, 255), -1)
    
    return frame

def process_video(input_path, output_path, num_processes, percent_to_process):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = int(total_frames * (percent_to_process / 100))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    trace_points = mp.Manager().list()  # Shared list to store ball positions across processes

    pool = mp.Pool(num_processes)  # Create a pool of workers with a limited number of processes

    frames = []
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    # Process frames in parallel
    process_func = partial(process_frame, trace_points=trace_points, n_trace_frames=n_trace_frames)
    processed_frames = pool.map(process_func, frames)

    for frame in processed_frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_path = 'input.mp4'
    output_video_path = 'output.mp4'
    num_processes = 6  # Adjust this number based on your system's capabilities
    percent_to_process = 5  # Adjust this to the percentage of the video you want to process
    process_video(input_video_path, output_video_path, num_processes, percent_to_process)