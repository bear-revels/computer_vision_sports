import cv2

class VideoAnnotator:
    def __init__(self, output_video_path):
        self.output_video_path = output_video_path
        self.writer = None

    def annotate_frame(self, frame, objects):
        for obj in objects:
            x, y, w, h, team_id = obj['x'], obj['y'], obj['w'], obj['h'], obj['team_id']
            color = (0, 255, 0) if team_id == 1 else (0, 0, 255)
            cv2.ellipse(frame, (x + w // 2, y + h), (w // 2, h // 4), 0, 0, 360, color, 2)
        return frame

    def write_frame(self, frame):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()