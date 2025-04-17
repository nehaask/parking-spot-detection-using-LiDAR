import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import DBSCAN


# Load the point cloud from the saved file
filename = '/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/output_parking_lot/basemap.pcd'
pcd_all = o3d.t.io.read_point_cloud(filename)

# Access the 'object_tag' attribute
object_tag_all = pcd_all.point["object_tag"].numpy() 

car_markings = np.where(object_tag_all == 14)[0]
road_markings = np.where(object_tag_all == 24)[0]

# clustering
clustering = DBSCAN(eps=0.5, min_samples=10).fit(car_markings.reshape(1, -1))
labels = clustering.labels_
unique_labels = set(labels)

number_of_clusters = len(unique_labels)

print(f"Number of clusters: {number_of_clusters}")

# Assuming 'pcd_all' is the CUDA-based PointCloud
# Create a color map
colors = plt.get_cmap("tab20")(labels / (number_of_clusters if number_of_clusters > 0 else 1))
colors[labels < 0] = 0  # Assign color for noise

# Convert the colors from RGBA to RGB (because Open3D expects RGB)
pcd_all.point["colors"] = o3d.core.Tensor(colors[:, :3], o3d.core.float32)

# Visualize the point cloud with the assigned colors
o3d.visualization.draw_geometries([pcd_all], window_name="DBSCAN Clustering", width=800, height=600)

# # Step 4: Determine parking occupancy
# occupied = (labels >= 0).sum()
# print(f"Estimated occupied parking slots: {occupied}")

