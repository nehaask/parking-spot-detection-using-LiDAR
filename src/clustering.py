import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def load_and_process_pcd(file_path):
    """Load PCD file and convert to numpy array"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def create_parking_rectangle(polygon, color=[0, 0, 1], z=0.2):
    """
    Creates a 3D LineSet rectangle from a 2D parking polygon.
    """
    pts_3d = [(x, y, z) for x, y in polygon]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Connect rectangle edges
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main():
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

    # Configuration parameters
    # PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/Test_segmentation/basemap_parking_lot.pcd"
    # PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/outputs/basemap_30.pcd"
    PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/filtered_basemap.pcd"
    ANNOTATION_COLOR = [1, 0, 0]  # RGB for annotation markers (normalized 0-1)
    CLUSTER_EPS = 1             # DBSCAN distance threshold
    MIN_CLUSTER_POINTS = 300      # Minimum points per cluster
    
    # Processing pipeline
    original_pcd = load_and_process_pcd(PCD_FILE)

    o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")

    points = np.asarray(original_pcd.points)
    print(f"Original point cloud size: {points.shape}")

    pcd = original_pcd.voxel_down_sample(voxel_size=0.1)
    print(f"Downsampled point cloud size: {np.asarray(pcd.points).shape}")

    db = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_CLUSTER_POINTS).fit(points)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    colors = plt.get_cmap("tab10")(labels / n_clusters)[:, :3]  # Normalize labels for color mapping
    colors[labels == -1] = [0, 0, 0]  # Noise points in black
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="DBSCAN Clustering Result")

    cluster_boxes = []
    labels_unique = set(labels)
    labels_unique.discard(-1)  # Remove noise label

    for cluster_id in labels_unique:
        cluster_points = points[labels == cluster_id]
        if cluster_points.shape[0] == 0:
            continue
        
        # Create Open3D point cloud for this cluster
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

        # Compute bounding box (you can use OrientedBoundingBox if preferred)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 1, 0)  # green
        cluster_boxes.append(bbox)

     # Add parking space rectangles to visualization
    for spot in PARKING_SPOTS:
        rect = create_parking_rectangle(spot['polygon'], color=[1, 1, 0])  # yellow
        cluster_boxes.append(rect)
        
    vis_objects = [pcd] + cluster_boxes
    o3d.visualization.draw_geometries(vis_objects, window_name="Clusters + Parking Rectangles")



if __name__ == "__main__":
    main()


