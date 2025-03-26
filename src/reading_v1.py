from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d

# Load point cloud
pcd = o3d.io.read_point_cloud("neha/lidar_scan.ply")
# points = np.asarray(pcd.points)

# Use DBSCAN for clustering objects
# db = DBSCAN(eps=0.5, min_samples=10).fit(points)
# labels = db.labels_

# # Color each cluster differently
# colors = np.random.rand(len(set(labels)), 3)
# pcd.colors = o3d.utility.Vector3dVector(colors[labels])

# Visualize segmented point cloud
o3d.visualization.draw_geometries([pcd])
