import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def load_and_process_pcd(file_path):
    """Load PCD file and convert to numpy array"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

if __name__ == "__main__":
    # Configuration parameters
    # PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/Test_segmentation/basemap_parking_lot.pcd"
    PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/accumulated/accumulated_1_10.pcd"
    ANNOTATION_COLOR = [1, 0, 0]  # RGB for annotation markers (normalized 0-1)
    CLUSTER_EPS = 0.8              # DBSCAN distance threshold
    MIN_CLUSTER_POINTS = 200      # Minimum points per cluster
    
    # Processing pipeline
    original_pcd = load_and_process_pcd(PCD_FILE)

    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")

    points = np.asarray(original_pcd.points)
    print(f"Original point cloud size: {points.shape}")

    pcd = original_pcd.voxel_down_sample(voxel_size=0.3)
    print(f"Downsampled point cloud size: {np.asarray(pcd.points).shape}")

    db = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_CLUSTER_POINTS).fit(points)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    colors = plt.get_cmap("tab10")(labels / n_clusters)[:, :3]  # Normalize labels for color mapping
    colors[labels == -1] = [0, 0, 0]  # Noise points in black
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="DBSCAN Clustering Result")


