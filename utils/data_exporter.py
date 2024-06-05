import polars as pl

class DataExporter:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def export_data(self, tracking_data):
        data = []
        for frame_data in tracking_data:
            frame_id = frame_data['frame_id']
            for obj in frame_data['objects']:
                data.append([frame_id, obj['object_id'], obj['x'], obj['y'], obj['team_id']])
        df = pl.DataFrame(data, schema=['frame', 'object_id', 'x', 'y', 'team_id'])
        df.write_csv(self.csv_path)