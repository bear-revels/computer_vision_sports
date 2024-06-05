import cv2
import numpy as np

class TeamClassifier:
    def classify_teams(self, detections, frame):
        classified_detections = []
        for (x, y, width, height, confidence, class_id) in detections:
            roi = frame[y:y + height, x:x + width]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hue = np.mean(hsv_roi[:, :, 0])
            team_id = self.get_team_id(avg_hue)
            classified_detections.append((x, y, width, height, confidence, class_id, team_id))
        return classified_detections

    def get_team_id(self, hue):
        if hue < 20:
            return 1  # Team 1
        elif hue > 100:
            return 2  # Team 2
        else:
            return 0  # Referee/Other