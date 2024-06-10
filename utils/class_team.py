from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self):
        self.team_assigner = TeamAssigner()

    def classify_teams(self, detections, frame):
        # Filter out players from detections
        players = [det for det in detections if det['class_id'] == 1]

        if len(players) > 0:
            self.team_assigner.assign_team_color(frame, players)
            for player in players:
                bbox = (player['x'], player['y'], player['x'] + player['w'], player['y'] + player['h'])
                # Use a temporary unique identifier for each player for team assignment
                temp_id = f"{player['x']}_{player['y']}_{player['w']}_{player['h']}"
                player['temp_id'] = temp_id
                player['team_id'] = self.team_assigner.get_player_team(frame, bbox, temp_id)

        return detections

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None  # Initialize kmeans as None

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        if len(player_detections) < 2:
            # If less than 2 players, assign all to one team arbitrarily
            for player_detection in player_detections:
                player_detection['team_id'] = 1
            self.kmeans = None  # Set kmeans to None if not used
            return

        player_colors = []
        for player_detection in player_detections:
            bbox = (player_detection['x'], player_detection['y'], player_detection['x'] + player_detection['w'], player_detection['y'] + player_detection['h'])
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            # If kmeans was not set, default all to team 1
            self.player_team_dict[player_id] = 1
            return 1

        player_color = self.get_player_color(frame, bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id

        return team_id