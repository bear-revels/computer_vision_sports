import cv2
import polars as pl
import numpy as np
from multiprocessing import Pool, cpu_count
import gc

class DataExporter:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def export_data(self, tracking_data):
        data = []
        max_objects = max(len(frame_data['objects']) for frame_data in tracking_data)

        for frame_data in tracking_data:
            frame_id = frame_data['frame_id']
            row = {'frame': frame_id}
            for i, obj in enumerate(frame_data['objects']):
                row[f'p{i+1}_x'] = obj['x']
                row[f'p{i+1}_y'] = obj['y']
                row[f'p{i+1}_t'] = obj['team_id']
            data.append(row)
        
        df = pl.DataFrame(data)
        df.write_parquet(self.parquet_path)

class VideoProcessor:
    def __init__(self, video_path, output_video_path, parquet_path):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.parquet_path = parquet_path
        self.writer = None
        self.data_exporter = DataExporter(parquet_path)

    def read_frames(self, frame_rate_reduction=2):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate_reduction == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        return frames

    def interpolate_positions(self, positions):
        df_positions = pl.DataFrame(positions, schema={'x': pl.Float64, 'y': pl.Float64})

        # Interpolate missing values
        df_positions = df_positions.with_columns([
            pl.col('x').interpolate(),
            pl.col('y').interpolate()
        ])

        # Backfill any remaining missing values
        df_positions = df_positions.fill_null(strategy='backward')

        interpolated_positions = df_positions.to_numpy()
        return interpolated_positions

    def annotate_frame(self, frame, objects):
        annotated_frame = frame.copy()
        for obj in objects:
            x, y, w, h, team_id, class_id, object_id = int(obj['x']), int(obj['y']), int(obj['w']), int(obj['h']), obj['team_id'], obj['class_id'], obj['object_id']
            if class_id == 0:  # Assuming class_id 0 corresponds to ball
                annotated_frame = self.draw_triangle(annotated_frame, (x, y, x + w, y + h), (0, 255, 0))
            else:
                color = (0, 255, 0) if team_id == 1 else (0, 0, 255)
                annotated_frame = self.draw_ellipse(annotated_frame, (x, y, x + w, y + h), color, object_id)
        return annotated_frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width / 2), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = self.get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_bbox_width(self, bbox):
        return bbox[2] - bbox[0]

    def write_frame(self, frame):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def process_and_export(self, frames, tracking_data, batch_size=20):
        # Interpolate positions for all objects
        try:
            for i in range(len(tracking_data[0]['objects'])):
                object_positions = [{frame_data['frame_id']: frame_data['objects'][i]} for frame_data in tracking_data]
                interpolated_positions = self.interpolate_positions([obj[1] for obj in object_positions if 1 in obj])

                if len(interpolated_positions) != len(tracking_data):
                    raise ValueError(f"Interpolation failed: expected {len(tracking_data)} positions, but got {len(interpolated_positions)}")

                for j, frame_data in enumerate(tracking_data):
                    if 1 in object_positions[j]:
                        frame_data['objects'][i]['x'], frame_data['objects'][i]['y'] = interpolated_positions[j]
        except Exception as e:
            print(f"Interpolation failed: {e}. Skipping interpolation.")

        def annotate_and_write(frames, tracking_data):
            for i, frame in enumerate(frames):
                annotated_frame = self.annotate_frame(frame, tracking_data[i]['objects'])
                self.write_frame(annotated_frame)

        # Use multiprocessing for annotation and writing frames
        with Pool(cpu_count()) as pool:
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_tracking_data = tracking_data[i:i + batch_size]
                pool.apply_async(annotate_and_write, (batch_frames, batch_tracking_data))
                # Clear cache periodically
                if i % (batch_size * 5) == 0:
                    gc.collect()
            pool.close()
            pool.join()

        self.release()
        self.data_exporter.export_data(tracking_data)