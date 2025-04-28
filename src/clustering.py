import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# ======================
# Global Parking Spot Definitions
# ======================
PARKING_SPOTS = [
    {'id': 10, 'polygon': [(1.2, -31.8), (6.2, -31.8), (6.2, -29.8), (1.2, -29.8)]},
    {'id': 5,  'polygon': [(8.2, -31.8), (13.2, -31.8), (13.2, -29.8), (8.2, -29.8)]},
    {'id': 9,  'polygon': [(1.2, -28.8), (6.2, -28.8), (6.2, -26.8), (1.2, -26.8)]},
    {'id': 4,  'polygon': [(8.2, -28.8), (13.2, -28.8), (13.2, -26.8), (8.2, -26.8)]},
    {'id': 8,  'polygon': [(1.2, -25.8), (6.2, -25.8), (6.2, -23.8), (1.2, -23.8)]},
    {'id': 3,  'polygon': [(8.2, -25.8), (13.2, -25.8), (13.2, -23.8), (8.2, -23.8)]},
    {'id': 7,  'polygon': [(1.2, -22.8), (6.2, -22.8), (6.2, -20.8), (1.2, -20.8)]},
    {'id': 2,  'polygon': [(8.2, -22.8), (13.2, -22.8), (13.2, -20.8), (8.2, -20.8)]},
    {'id': 6,  'polygon': [(1.2, -19.8), (6.2, -19.8), (6.2, -17.8), (1.2, -17.8)]},
    {'id': 1,  'polygon': [(8.2, -19.8), (13.2, -19.8), (13.2, -17.8), (8.2, -17.8)]}
]

# ======================
# Helper Functions
# ======================

def load_pcd(file_path):
    """Load a PCD file."""
    return o3d.io.read_point_cloud(file_path)

def create_parking_rectangle(polygon, color=[0, 0, 1], z=0.2):
    """Create a 3D rectangle LineSet for a parking polygon."""
    pts_3d = [(x, y, z) for x, y in polygon]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def run_dbscan_and_visualize(
    input_pcd_file,
    cluster_eps=1,
    min_cluster_points=300,
    save_clustered=True,
    output_dir="clustered_outputs",
  # ✅ Correct default string
    downsample_voxel=0.3,
    visualize=False
):
    """
    Run DBSCAN clustering on a given PCD and visualize/save the result.
    Automatically names saved clustered PCD based on input filename.
    """
    # Load and optionally downsample point cloud
    pcd = load_pcd(input_pcd_file)
    points = np.asarray(pcd.points)

    print(f"Running DBSCAN on {input_pcd_file} with {points.shape[0]} points.")

    if downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
        print(f"Downsampled point cloud to {np.asarray(pcd.points).shape}")

    # Clustering
    db = DBSCAN(eps=cluster_eps, min_samples=min_cluster_points).fit(points)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    # Coloring points by cluster
    colors = plt.get_cmap("tab10")(labels / n_clusters)[:, :3]
    colors[labels == -1] = [0, 0, 0]  # Noise in black
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save clustered point cloud
    if save_clustered:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(input_pcd_file)
        output_filename = os.path.join(output_dir, f"clustered_{basename}")
        o3d.io.write_point_cloud(output_filename, pcd)
        print(f"✅ Clustered PCD saved to: {output_filename}")

    # Visualization
    if visualize:
        vis_objects = [pcd]
        labels_unique = set(labels)
        labels_unique.discard(-1)

        # Add bounding boxes for clusters
        for cluster_id in labels_unique:
            cluster_points = points[labels == cluster_id]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (0, 1, 0)  # Green
            vis_objects.append(bbox)

        # Add parking rectangles
        for spot in PARKING_SPOTS:
            rect = create_parking_rectangle(spot['polygon'], color=[1, 0.5, 0])
            vis_objects.append(rect)

        o3d.visualization.draw_geometries(vis_objects, window_name="Clusters + Parking Spots")

    return pcd

# ======================
# Standalone Execution
# ======================

if __name__ == "__main__":
    # Example: manual run
    PCD_FILE = "filtered_outputs/filtered_basemap_0030.pcd"
    clustered_dir = "clustered_outputs"
    run_dbscan_and_visualize(
        input_pcd_file=PCD_FILE,
        cluster_eps=1,
        min_cluster_points=300,
        save_clustered=True,
        output_dir=clustered_dir,
        visualize=True
    )
