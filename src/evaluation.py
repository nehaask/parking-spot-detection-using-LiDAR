import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

def load_and_process_pcd(file_path):
    """Load PCD file and convert to numpy array"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def is_point_in_polygon(point, polygon):
    """Check if a given point is within the polygon"""
    p = Point(point[0], point[1])
    return polygon.contains(p)

def apply_dbscan_to_parking_spot(pcd_points, cluster_eps, min_cluster_points):
    """Apply DBSCAN clustering to points inside a specific parking spot"""
    db = DBSCAN(eps=cluster_eps, min_samples=min_cluster_points).fit(pcd_points)
    labels = db.labels_
    return labels

def main():
    # ======================
    # Global Configurations
    # ======================
    # Predefined parking spots in world coordinates (x, y)
    PARKING_SPOTS = [
        {'id': 10, 
         'polygon': Polygon([(1, -31), (6, -31), (6, -29), (1, -29)]), 
         'occupied': False},
        {'id': 5, 
         'polygon': Polygon([(8, -31), (13, -31), (13, -29), (8, -29)]), 
         'occupied': False}, 
        {'id': 9, 
         'polygon': Polygon([(1, -28), (6, -28), (6, -26), (1, -26)]), 
         'occupied': False},
        {'id': 8, 
         'polygon': Polygon([(1, -25), (6, -25), (6, -23), (1, -23)]), 
         'occupied': False}, 
        {'id': 7, 
         'polygon': Polygon([(1, -23), (6, -23), (6, -20), (1, -20)]), 
         'occupied': False},
        {'id': 6, 
         'polygon': Polygon([(1, -20), (6, -20), (6, -18), (1, -18)]), 
         'occupied': False},
        {'id': 4, 
         'polygon': Polygon([(8, -28), (13, -28), (13, -26), (8, -26)]), 
         'occupied': False},
        {'id': 3, 
         'polygon': Polygon([(8, -25), (13, -25), (13, -23), (8, -23)]), 
         'occupied': False},
        {'id': 2, 
         'polygon': Polygon([(8, -23), (13, -23), (13, -21), (8, -21)]), 
         'occupied': False}, 
        {'id': 1, 
         'polygon': Polygon([(8, -20), (13, -20), (13, -18), (8, -18)]), 
         'occupied': False}
    ]

    # Configuration parameters
    PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/accumulated/accumulated_1_10.pcd"
    CLUSTER_EPS = 0.8              # DBSCAN distance threshold for small-scale clustering
    MIN_CLUSTER_POINTS = 100       # Minimum points per cluster (adjusted for smaller scale)
    
    # Processing pipeline
    original_pcd = load_and_process_pcd(PCD_FILE)

    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")

    points = np.asarray(original_pcd.points)
    print(f"Original point cloud size: {points.shape}")

    # Iterate over each parking spot and apply DBSCAN to the points within the polygon
    for spot in PARKING_SPOTS:
        parking_spot_points = []  # Points inside the parking spot polygon
        for idx, point in enumerate(points):
            if is_point_in_polygon(point[:2], spot['polygon']):
                parking_spot_points.append(point)
        
        parking_spot_points = np.array(parking_spot_points)
        if parking_spot_points.shape[0] > MIN_CLUSTER_POINTS:
            # Apply DBSCAN to the points inside the parking spot
            labels = apply_dbscan_to_parking_spot(parking_spot_points, CLUSTER_EPS, MIN_CLUSTER_POINTS)
            unique_labels = set(labels)
            
            # Mark as occupied if there are clusters (excluding noise, label -1)
            if len(unique_labels) > 1:  # More than one cluster means some points were grouped
                spot['occupied'] = True
                print(f"Parking spot {spot['id']} is occupied.")
            else:
                spot['occupied'] = False
                print(f"Parking spot {spot['id']} is empty.")
            
            # Visualization of the parking spot with cluster colors
            colors = plt.get_cmap("tab10")(labels / len(unique_labels))[:, :3]  # Normalize labels for color mapping
            colors[labels == -1] = [0, 0, 0]  # Noise points in black
            
            # Visualize the cluster points
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(parking_spot_points)
            pcd_vis.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd_vis], window_name=f"Parking Spot {spot['id']} - Clusters")
        else:
            print(f"Parking spot {spot['id']} has insufficient points for clustering.")

if __name__ == "__main__":
    main()
