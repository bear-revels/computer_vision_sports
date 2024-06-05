from collections import deque
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.trackers = {}

    def track_objects(self, detections):
        for det in detections:
            object_id = self.match_to_existing_trackers(det)
            if object_id is None:
                object_id = len(self.trackers) + 1
                self.trackers[object_id] = deque(maxlen=50)
            self.trackers[object_id].append(det)
        return self.get_current_tracks()

    def match_to_existing_trackers(self, detection):
        # Implement matching logic (e.g., IoU-based matching)
        pass

    def get_current_tracks(self):
        current_tracks = []
        for object_id, track in self.trackers.items():
            if track:
                # Convert tuple to dictionary
                x, y, w, h, confidence, class_id, team_id = track[-1]
                current_tracks.append({
                    'object_id': object_id,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'confidence': confidence,
                    'class_id': class_id,
                    'team_id': team_id
                })
        return current_tracks