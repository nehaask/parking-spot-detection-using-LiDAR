"""
CARLA Parking Spot Detection System using Semantic LiDAR and DBSCAN Clustering
"""

# ======================
# Environment Setup
# ======================
import glob
import os
import sys
import time
import math
import numpy as np
from sklearn.cluster import KMeans
import carla
from collections import deque


# CARLA PythonAPI setup
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ======================
# Global Configurations
# ======================
# Predefined parking spots in world coordinates (x, y)
PARKING_SPOTS = [
    {
        "id": 1,
        "polygon": [(85.0, 125.0), (90.0, 125.0), (90.0, 130.0), (85.0, 130.0)],
        "occupied": False
    },
    {
        "id": 2,
        "polygon": [(95.0, 125.0), (100.0, 125.0), (100.0, 130.0), (95.0, 130.0)],
        "occupied": False
    }
]

# CARLA semantic segmentation object IDs
OBJECT_IDS = {
    "Ground": 25,
    "ParkingSpace": 150,
    "Vehicle": 10,
}

# ======================
# Core Parking Monitor Class
# ======================
class ParkingMonitor:
    def __init__(self, world, vehicle):
        """Initialize parking monitoring system"""
        self.world = world          # CARLA world reference
        self.vehicle = vehicle      # Player-controlled vehicle
        self.lidar = None           # LiDAR sensor object
        self.debug = world.debug    # CARLA debug drawing interface
        self.lidar_buffer = deque(maxlen=5)  # Automatically maintains a fixed-length buffer

    def point_in_polygon(self, point, polygon):
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        # Check intersections with polygon edges
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            # Calculate intersection point
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def save_point_cloud(self, filename="buffered_lidar.txt"):
        """Save buffered LiDAR points to a text file (x, y, z)"""
        all_points = [pt for frame in self.lidar_buffer for pt in frame]
        
        with open(filename, "w") as f:
            for pt in all_points:
                # Write x, y, z coordinates (flat ground at z=0.5)
                f.write(f"{pt[0]} {pt[1]} 0.5\n")

        print(f"[INFO] Saved {len(all_points)} points to '{filename}'")


    def draw_parking_spots(self):
        """Visualize parking spots with color-coded occupancy status"""
        for spot in PARKING_SPOTS:
            # Choose color based on occupancy (green=available, red=occupied)
            color = carla.Color(0, 255, 0) if not spot["occupied"] else carla.Color(255, 0, 0)
            
            # Convert polygon points to CARLA locations
            points = [carla.Location(x=pt[0], y=pt[1], z=0.5) for pt in spot["polygon"]]
            
            # Draw polygon edges
            for i in range(len(points)):
                self.debug.draw_line(
                    points[i], points[(i+1)%len(points)],
                    thickness=0.2,    # Line thickness in meters
                    color=color,      # Color based on occupancy
                    life_time=0.1     # Duration to keep line visible (seconds)
                )

    def semantic_lidar_callback_old(self, point_cloud):
        """Process LiDAR data and update parking spot status"""
        # Get current vehicle transform for coordinate conversion
        vehicle_transform = self.vehicle.get_transform()
        obstacle_points = []

        # Process each LiDAR detection point
        for detection in point_cloud:
            # Filter out ground points (only keep obstacles)
            if detection.object_tag != OBJECT_IDS["Ground"]:
                # Convert local coordinates to world space
                local_point = detection.point
                world_point = vehicle_transform.transform(
                    carla.Location(local_point.x, local_point.y, local_point.z))
                obstacle_points.append([world_point.x, world_point.y])


        #DEBUG

        # for p in obstacle_points:
        #     self.debug.draw_point(
        #         carla.Location(x=p[0], y=p[1], z=0.5),
        #         size=0.05, color=carla.Color(0, 0, 255), life_time=0.2
        #     )


        # Cluster obstacles using DBSCAN algorithm
        centroids = []
        if obstacle_points:
            X = np.array(obstacle_points)
            
            # DBSCAN parameters (eps in meters, min_samples for cluster formation)
            clustering = DBSCAN(eps=1.5, min_samples=50).fit(X)
            
            # Process detected clusters
            for label in set(clustering.labels_):
                if label != -1:  # Ignore noise points
                    cluster_points = X[clustering.labels_ == label]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
                    
                    # Visualize cluster centroids
                    self.debug.draw_point(
                        carla.Location(x=centroid[0], y=centroid[1], z=1.0),
                        size=0.05,  # Point size in meters
                        color=carla.Color(215, 195, 0),  # Orange color
                        life_time=0.5  # Duration to keep point visible
                    )

        # Update parking spot occupancy status
        for spot in PARKING_SPOTS:
            spot["occupied"] = False
            for centroid in centroids:
                if self.point_in_polygon(centroid, spot["polygon"]):
                    spot["occupied"] = True
                    break  # No need to check other centroids for this spot

        # Console output of current status
        free_spots = [s["id"] for s in PARKING_SPOTS if not s["occupied"]]
        print(f"\n[{time.strftime('%H:%M:%S')}] Free spots: {free_spots}")

   
        vehicle_transform = self.vehicle.get_transform()
        current_frame_points = []

        for detection in point_cloud:
            if detection.object_tag != OBJECT_IDS["Ground"]:
                local_point = detection.point
                world_point = vehicle_transform.transform(
                    carla.Location(local_point.x, local_point.y, local_point.z))
                current_frame_points.append([world_point.x, world_point.y])

        # Add current frame to buffer
        self.lidar_buffer.append(current_frame_points)

        # Merge all points from buffer
        obstacle_points = [pt for frame in self.lidar_buffer for pt in frame]

        # Optional: Visualize all buffered points (blue)
        for p in obstacle_points:
            self.debug.draw_point(
                carla.Location(x=p[0], y=p[1], z=0.5),
                size=0.05, color=carla.Color(0, 0, 255), life_time=0.2
            )

        # Cluster buffered points
        centroids = []
        if obstacle_points:
            X = np.array(obstacle_points)
            clustering = DBSCAN(eps=1.5, min_samples=20).fit(X)

            for label in set(clustering.labels_):
                if label != -1:
                    cluster_points = X[clustering.labels_ == label]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)

                    # Visualize cluster centroids (orange)
                    self.debug.draw_point(
                        carla.Location(x=centroid[0], y=centroid[1], z=1.0),
                        size=0.05, color=carla.Color(215, 195, 0), life_time=0.5
                    )

        # Update parking spot occupancy
        for spot in PARKING_SPOTS:
            spot["occupied"] = False
            for centroid in centroids:
                if self.point_in_polygon(centroid, spot["polygon"]):
                    spot["occupied"] = True
                    break

        free_spots = [s["id"] for s in PARKING_SPOTS if not s["occupied"]]
        print(f"\n[{time.strftime('%H:%M:%S')}] Free spots: {free_spots}")

    def semantic_lidar_callback_kmeans(self, point_cloud):
        """Process LiDAR data and update parking spot status"""
        # Get current vehicle transform for coordinate conversion
        vehicle_transform = self.vehicle.get_transform()
        obstacle_points = []

        # Process each LiDAR detection point
        for detection in point_cloud:
            # Filter out ground points (only keep obstacles)
            if detection.object_tag != OBJECT_IDS["Ground"]:
                # Convert local coordinates to world space
                local_point = detection.point
                world_point = vehicle_transform.transform(
                    carla.Location(local_point.x, local_point.y, local_point.z))
                obstacle_points.append([world_point.x, world_point.y])

        #DEBUG

        # for p in obstacle_points:
        #     self.debug.draw_point(
        #         carla.Location(x=p[0], y=p[1], z=0.5),
        #         size=0.05, color=carla.Color(0, 0, 255), life_time=0.2
        #     )

        # Cluster obstacles using DBSCAN algorithm
        centroids = []
        if obstacle_points:
            X = np.array(obstacle_points)

            # Estimate number of clusters based on number of points (or just fix it)
            # estimated_clusters = max(1, len(X) // 200)  # One cluster per ~200 points
            estimated_clusters = 2
            n_clusters = min(estimated_clusters, 10)    # Limit to 10 max clusters

            if len(X) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X)

                centroids = kmeans.cluster_centers_

                for centroid in centroids:
                    # Visualize cluster centroids
                    self.debug.draw_point(
                        carla.Location(x=centroid[0], y=centroid[1], z=1.0),
                        size=0.05,
                        color=carla.Color(0, 255, 255),  # Cyan for KMeans
                        life_time=0.5
                    )
            else:
                centroids = []


        # Update parking spot occupancy status
        for spot in PARKING_SPOTS:
            spot["occupied"] = False
            for centroid in centroids:
                if self.point_in_polygon(centroid, spot["polygon"]):
                    spot["occupied"] = True
                    break  # No need to check other centroids for this spot

        # Console output of current status
        free_spots = [s["id"] for s in PARKING_SPOTS if not s["occupied"]]
        print(f"\n[{time.strftime('%H:%M:%S')}] Free spots: {free_spots}")

    def setup_lidar(self):
        """Configure and attach semantic LiDAR sensor to vehicle"""
        blueprint_lib = self.world.get_blueprint_library()
        
        # Get LiDAR blueprint and configure parameters
        lidar_bp = blueprint_lib.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', '64')          # Vertical resolution
        lidar_bp.set_attribute('range', '50.0')           # Detection range in meters
        lidar_bp.set_attribute('rotation_frequency', '20')# Scanner rotation speed (Hz)
        lidar_bp.set_attribute('points_per_second', '100000')  # Point density
        
        # Mounting position on vehicle (center, 2.5m height)
        transform = carla.Transform(
            carla.Location(x=0, y=0, z=2.5),
            carla.Rotation(pitch=0, yaw=90, roll=0)  # Forward-facing
        )
        
        # Spawn and activate LiDAR sensor
        self.lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self.semantic_lidar_callback_kmeans(data))

# ======================
# Main Simulation Loop
# ======================
def main():
    actor_list = []  # Track all CARLA actors

    try:
        # Initialize CARLA connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town05')  # Load map
       # Spawn player vehicle
        blueprint_lib = world.get_blueprint_library()

        # Get all vehicle blueprints and select first 5 different models
        vehicle_blueprints = blueprint_lib.filter('vehicle.*')
        selected_blueprints = [vb for vb in vehicle_blueprints][:5]  # First 5 unique vehicles

        # Get all available spawn points
        spawn_points = world.get_map().get_spawn_points()

        # Spawn 5 vehicles at different locations
        for i in range(5):
            try:
                # Cycle through spawn points using modulo to avoid index errors
                vehicle = world.spawn_actor(
                    selected_blueprints[i % len(selected_blueprints)],
                    spawn_points[167 + i]  # Different spawn near parking area
                )
                # Set vehicle to autopilot mode
                vehicle.set_autopilot(True)
                actor_list.append(vehicle)
            except Exception as e:
                print(f"Failed to spawn vehicle {i}: {str(e)}")

        # Initialize parking monitoring system
        monitor = ParkingMonitor(world, vehicle)
        monitor.setup_lidar()

        # Configure spectator camera
        spectator = world.get_spectator()
        
        def follow_vehicle_camera():
            """Maintain top-down view of vehicle"""
            transform = vehicle.get_transform()
            # Position camera 40m above vehicle
            location = transform.location + carla.Location(z=40)
            # Directly downward view (-90 degree pitch)
            rotation = carla.Rotation(pitch=-90)
            spectator.set_transform(carla.Transform(location, rotation))

        # Main simulation loop
        while True:
            monitor.draw_parking_spots()  # Update parking spot visuals
            follow_vehicle_camera()       # Keep camera following vehicle
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.3,  # Maintain forward motion
                steer=0.0      # No steering input
            ))
            time.sleep(0.1)  # Control loop frequency (~10Hz)
            

    finally:
        # Cleanup routine
        print("Cleaning up...")
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        if monitor.lidar.is_alive:
            monitor.lidar.destroy()
        print("Cleanup complete.")

if __name__ == '__main__':
    main()