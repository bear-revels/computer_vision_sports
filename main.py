import os
from utils import VideoProcessor, ObjectDetector, TeamClassifier, ObjectTracker, DataExporter, VideoAnnotator

def main():
    # Input video file path
    input_video_path = 'files/input/08fd33_4.mp4'
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Output paths
    output_csv_path = f'files/output/csv/{base_name}_coordinates.csv'
    output_video_path = f'files/output/video/{base_name}_annotated.mp4'

    # Initialize components
    video_processor = VideoProcessor(input_video_path)
    object_detector = ObjectDetector('files/models/best.pt')
    team_classifier = TeamClassifier()
    object_tracker = ObjectTracker()
    data_exporter = DataExporter(output_csv_path)
    video_annotator = VideoAnnotator(output_video_path)
    # homography = Homography()

    # Process video frames
    frames = video_processor.read_frames()
    
    tracking_data = []
    
    for frame_id, frame in enumerate(frames):
        # Detect objects in the frame
        detections = object_detector.detect_objects(frame)
        
        # Classify teams
        classified_detections = team_classifier.classify_teams(detections, frame)
        
        # Track objects
        tracked_objects = object_tracker.track_objects(classified_detections)
        
        # # Apply homography for normalization
        # normalized_objects = homography.normalize_coordinates(tracked_objects)
        
        # Instead of normalized_objects, use tracked_objects directly
        for obj in tracked_objects:
            obj['frame_id'] = frame_id
        
        # Save tracking data
        tracking_data.append({
            'frame_id': frame_id,
            'objects': tracked_objects
        })
        
        # Annotate frame
        annotated_frame = video_annotator.annotate_frame(frame, tracked_objects)
        
        video_annotator.write_frame(annotated_frame)
    
    # Export tracking data to CSV
    data_exporter.export_data(tracking_data)

    # Release the video writer
    video_annotator.release()

if __name__ == "__main__":
    main()