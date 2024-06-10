import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from .stub_handler import save_stub, load_stub
from deep_sort_realtime.deepsort_tracker import DeepSort

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
                detections.append({
                    'x': int(x1),
                    'y': int(y1),
                    'w': int(width),
                    'h': int(height),
                    'score': float(score),
                    'class_id': int(class_id)
                })
        return detections

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

class ObjectTracker:
    def __init__(self, max_distance=50, max_skipped_frames=5):
        self.max_distance = max_distance
        self.max_skipped_frames = max_skipped_frames
        self.tracks = {}
        self.next_id = 0
        self.skipped_frames = {}
        self.tracker = DeepSort(max_age=self.max_skipped_frames, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)

    def track_objects(self, detections, frame):
        bboxes = np.array([[det['x'], det['y'], det['x'] + det['w'], det['y'] + det['h'], det['score']] for det in detections])
        tracker_output = self.tracker.update_tracks(bboxes, frame)

        tracked_objects = []
        for track in tracker_output:
            track_id = track.track_id
            bbox = track.to_tlbr()
            tracked_objects.append({
                'object_id': track_id,
                'x': int(bbox[0]),
                'y': int(bbox[1]),
                'w': int(bbox[2] - bbox[0]),
                'h': int(bbox[3] - bbox[1]),
                'score': track.confidence,
                'class_id': track.class_id
            })

        return tracked_objects

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            return load_stub(stub_path)

        object_detector = ObjectDetector('files/models/best.pt')
        detections = object_detector.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            converted_detections = []
            for box, class_id, conf in zip(detection_supervision.xyxy, detection_supervision.class_id, detection_supervision.confidence):
                converted_detections.append({
                    'x': box[0],
                    'y': box[1],
                    'w': box[2] - box[0],
                    'h': box[3] - box[1],
                    'score': conf,
                    'class_id': class_id
                })

            detection_with_tracks = self.track_objects(converted_detections, frames[frame_num])

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection['bbox']
                cls_id = frame_detection['class_id']
                track_id = frame_detection['object_id']

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in converted_detections:
                bbox = frame_detection['bbox']
                cls_id = frame_detection['class_id']

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            save_stub(tracks, stub_path)

        return tracks

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == 'ball':
                        position = self.get_center_of_bbox(bbox)
                    else:
                        position = self.get_foot_position(bbox)
                    tracks[obj][frame_num][track_id]['position'] = position

    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_foot_position(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions