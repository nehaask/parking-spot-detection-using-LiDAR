# Re-run the parking spot detection pipeline after environment reset

import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

# ========================================
# Module: Parking Spot Detection Pipeline
# ========================================

class ParkingSpotDetector:
    def __init__(self, base_map_polygons):
        self.base_map = base_map_polygons
        self.cluster_eps = 2.0
        self.min_samples = 8

    def localize_vehicle(self, current_pose, map_transform=None):
        return current_pose

    def filter_ground_points(self, point_cloud):
        return point_cloud[point_cloud[:, 2] > 0.2]

    def cluster_obstacles(self, filtered_points):
        dbscan = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(filtered_points[:, :2])
        centroids = []

        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = filtered_points[labels == label]
            centroid = np.mean(cluster[:, :2], axis=0)
            centroids.append(centroid)

        return np.array(centroids)

    def classify_parking_spots(self, centroids):
        for spot in self.base_map:
            polygon = Polygon(spot["polygon"])
            spot["occupied"] = any(polygon.contains(Point(c[0], c[1])) for c in centroids)
        return self.base_map

    def run_pipeline(self, base_map_polygons, current_point_cloud, vehicle_pose):
        self.base_map = base_map_polygons
        localized_pose = self.localize_vehicle(vehicle_pose)
        filtered_points = self.filter_ground_points(current_point_cloud)
        centroids = self.cluster_obstacles(filtered_points)
        classified_spots = self.classify_parking_spots(centroids)
        return classified_spots, centroids
    

    def visualize_parking_spots(spots, centroids):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot each parking spot
        for spot in spots:
            polygon = np.array(spot["polygon"])
            color = 'red' if spot["occupied"] else 'green'
            ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.4, color=color, label=f"Spot {spot['id']}" if not spot["occupied"] else None)
            ax.plot(polygon[:, 0], polygon[:, 1], 'k--', linewidth=0.5)

        # Plot cluster centroids
        if centroids is not None and len(centroids) > 0:
            centroids = np.array(centroids)
            ax.scatter(centroids[:, 0], centroids[:, 1], color='blue', s=30, label='Detected Centroids')

        ax.set_title("Parking Spot Occupancy Visualization")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.axis("equal")
        ax.legend(loc='upper right')
        plt.grid(True)
        plt.show()


# =====================
# Example Usage Stub
# =====================
def example_usage():
    base_map = [
        {'id': 10, 'polygon': [(1, -30), (6, -30), (6, -28), (1, -28)], 'occupied': False},
        {'id': 9, 'polygon': [(1, -27), (6, -27), (6, -25), (1, -25)], 'occupied': False},
        {'id': 8, 'polygon': [(1, -24), (6, -24), (6, -22), (1, -22)], 'occupied': False},
        {'id': 7, 'polygon': [(1, -21), (6, -21), (6, -19), (1, -19)], 'occupied': False},
        {'id': 6, 'polygon': [(1, -18), (6, -18), (6, -16), (1, -16)], 'occupied': False},
        {'id': 5, 'polygon': [(8, -30), (13, -30), (13, -28), (8, -28)], 'occupied': False},
        {'id': 4, 'polygon': [(8, -27), (13, -27), (13, -25), (8, -25)], 'occupied': False},
        {'id': 3, 'polygon': [(8, -24), (13, -24), (13, -22), (8, -22)], 'occupied': False},
        {'id': 2, 'polygon': [(8, -21), (13, -21), (13, -19), (8, -19)], 'occupied': False},
        {'id': 1, 'polygon': [(8, -18), (13, -18), (13, -16), (8, -16)], 'occupied': False},
    ]


    # Generate clusters that fall inside a few parking spots
    realtime_cloud = np.vstack([
        np.random.normal(loc=[3, -30, 0.3], scale=0.3, size=(50, 3)),  # Spot 10
        np.random.normal(loc=[10, -24, 0.3], scale=0.3, size=(50, 3)), # Spot 3
        np.random.normal(loc=[4, -19, 0.3], scale=0.3, size=(50, 3))   # Spot 6
    ])

    vehicle_pose = (0, 0, 0)

    detector = ParkingSpotDetector(base_map)
    spots, centroids = detector.run_pipeline(base_map, realtime_cloud, vehicle_pose)

    return spots, centroids

spots_detected, detected_centroids = example_usage()
ParkingSpotDetector.visualize_parking_spots(spots_detected, detected_centroids)

