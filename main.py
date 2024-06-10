import os
import numpy as np
from utils import (
    VideoProcessor,
    ObjectDetector,
    ObjectTracker,
    CameraMovementEstimator,
    ViewTransformer,
    TeamClassifier,
    save_stub,
    load_stub
)

def main():
    # Input video file path
    input_video_path = 'files/input/08fd33_4.mp4'
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Output paths
    output_parquet_path = f'files/output/csv/{base_name}_coordinates.parquet'
    output_video_path = f'files/output/video/{base_name}_annotated.mp4'
    track_stub_path = f'files/stubs/{base_name}_tracks.pkl'
    camera_movement_stub_path = f'files/stubs/{base_name}_camera_movement.pkl'

    # Initialize components
    video_processor = VideoProcessor(input_video_path, output_video_path, output_parquet_path)
    object_detector = ObjectDetector('files/models/best.pt')
    team_classifier = TeamClassifier()
    object_tracker = ObjectTracker()

    # Process video frames
    frames = video_processor.read_frames(frame_rate_reduction=2)

    # Initialize camera movement estimator and view transformer
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movements = camera_movement_estimator.get_camera_movement(
        frames,
        read_from_stub=True,
        stub_path=camera_movement_stub_path
    )

    view_transformer = ViewTransformer()

    # Get object tracks (use stub if available)
    tracks = object_tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path=track_stub_path
    )

    # Add positions to tracks
    object_tracker.add_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = object_tracker.interpolate_ball_positions(tracks["ball"])

    # Classify teams and add team IDs
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            bbox = track["bbox"]
            temp_id = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            team = team_classifier.classify_teams(
                [{'x': bbox[0], 'y': bbox[1], 'w': bbox[2]-bbox[0], 'h': bbox[3]-bbox[1], 'class_id': 1}],
                frames[frame_num]
            )[0]['team_id']
            tracks["players"][frame_num][player_id]["team_id"] = team

    # Adjust positions based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movements)

    # Transform coordinates to field view
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Process frames and export data
    video_processor.process_and_export(frames, tracks)

if __name__ == "__main__":
    main()