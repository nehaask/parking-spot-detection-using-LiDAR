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
from sklearn.cluster import DBSCAN
import carla

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
    {'id': 10, 
     'polygon': [(1, -31), (6, -31), (6, -29), (1, -29)], 
     'occupied': False
    },

    {'id': 5, 
      'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)],
      'occupied': False
    }, 

    {'id': 9, 
       'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)], 
       'occupied': False
    },

    {'id': 8,
      'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)],
      'occupied': False
    }, 

    {'id': 7,
      'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)],
      'occupied': False
    }, 

    {'id': 6,
     'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)],
     'occupied': False
    }, 

    {'id': 4,
      'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)],
      'occupied': False
    }, 
    
    {'id': 3, 
     'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)],
     'occupied': False
     },
     
    {'id': 2,
      'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)],
      'occupied': False
    }, 
    
    {'id': 1,
     'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)],
     'occupied': False
    }
]

# CARLA semantic segmentation object IDs
OBJECT_IDS = {
    "Ground": 0,
    "ParkingSpace": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example IDs for parking spaces

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
                    life_time=0.5     # Duration to keep line visible (seconds)
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

        # Cluster obstacles using DBSCAN algorithm
        centroids = []
        if obstacle_points:
            X = np.array(obstacle_points)
            
            # DBSCAN parameters (eps in meters, min_samples for cluster formation)
            clustering = DBSCAN(eps=2.0, min_samples=100).fit(X)
            
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
            for spot in PARKING_SPOTS:
                spot["occupied"] = False  # Reset status
                for centroid in centroids:
                    if self.point_in_polygon(centroid, spot["polygon"]):
                        spot["occupied"] = True
                        break  # Exit early if any centroid found

        # Console output of current status
        free_spots = [s["id"] for s in PARKING_SPOTS if not s["occupied"]]
        print(f"\n[{time.strftime('%H:%M:%S')}] Free spots: {free_spots}")

    def semantic_lidar_callback(self, point_cloud):
        """Process LiDAR data and update parking spot status"""
        vehicle_transform = self.vehicle.get_transform()
        obstacle_points = []

        # Step 1: Convert semantic LiDAR points to world coordinates (filter out ground)
        for detection in point_cloud:
            if detection.object_tag != OBJECT_IDS["Ground"]:
                local_point = detection.point
                world_point = vehicle_transform.transform(
                    carla.Location(local_point.x, local_point.y, local_point.z)
                )
                obstacle_points.append([world_point.x, world_point.y])

        # Step 2: Cluster using DBSCAN
        centroids = []
        if obstacle_points:
            X = np.array(obstacle_points)
            clustering = DBSCAN(eps=2.0, min_samples=100).fit(X)

            for label in set(clustering.labels_):
                if label == -1:
                    continue  # skip noise
                cluster_points = X[clustering.labels_ == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)

                # Optional: visualize centroid in CARLA
                self.debug.draw_point(
                    carla.Location(x=centroid[0], y=centroid[1], z=1.0),
                    size=0.05,
                    color=carla.Color(215, 195, 0),  # yellow-orange
                    life_time=0.5
                )

        # Step 3: Update parking spot status
        for spot in PARKING_SPOTS:
            spot["occupied"] = False
            for centroid in centroids:
                if self.point_in_polygon(centroid, spot["polygon"]):
                    spot["occupied"] = True
                    break

        # Debug print
        free_spots = [s["id"] for s in PARKING_SPOTS if not s["occupied"]]
        print(f"[{time.strftime('%H:%M:%S')}] Free spots: {free_spots}")

    
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
        self.lidar.listen(lambda data: self.semantic_lidar_callback(data))

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
        vehicle_bp = blueprint_lib.filter('vehicle')[5]
        spawn_point = world.get_map().get_spawn_points()[167]  # Predefined parking area
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

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