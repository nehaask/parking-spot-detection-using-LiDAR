# import numpy as np
# import open3d as o3d
# from sklearn.cluster import DBSCAN
# from shapely.geometry import Point, Polygon


# def load_and_process_pcd(file_path):
#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points)
#     return pcd, points


# def create_parking_rectangle(polygon, color=[1, 0, 0], z=0.2):
#     pts_3d = [(x, y, z) for x, y in polygon]
#     lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(pts_3d)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
#     return line_set


# def extract_points_in_polygon(points, polygon):
#     """
#     Extracts points from the point cloud that are within a 2D polygon (parking spot).
#     """
#     poly = Polygon(polygon)
#     return np.array([pt for pt in points if poly.contains(Point(pt[0], pt[1]))])


# def main():
#     # File path to your PCD file
#     # PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/outputs/basemap_30.pcd"
#     PCD_FILE = "/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/filtered_output.pcd"
    
#     # DBSCAN parameters
#     EPS = 1.5
#     MIN_POINTS = 300

#     # Predefined parking spots (rectangles)
#     PARKING_SPOTS = [
#         {'id': 1, 'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)]},
#         {'id': 2, 'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)]},
#         {'id': 3, 'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)]},
#         {'id': 4, 'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)]},
#         {'id': 5, 'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)]},
#         {'id': 6, 'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)]},
#         {'id': 7, 'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)]},
#         {'id': 8, 'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)]},
#         {'id': 9, 'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)]},
#         {'id': 10, 'polygon': [(1, -31), (6, -31), (6, -29), (1, -29)]}
#     ]

#     # Load point cloud
#     pcd, points = load_and_process_pcd(PCD_FILE)

#     parking_lines = []
#     car_clusters = []

#     for spot in PARKING_SPOTS:
#         # Create parking spot rectangle
#         rect = create_parking_rectangle(spot['polygon'])
#         parking_lines.append(rect)

#         # Extract points within the parking spot rectangle
#         inside_points = extract_points_in_polygon(points, spot['polygon'])

#         if inside_points.shape[0] == 0:
#             continue

#         # Run DBSCAN clustering within the parking rectangle
#         clustering = DBSCAN(eps=EPS, min_samples=MIN_POINTS).fit(inside_points[:, :2])
#         labels = clustering.labels_

#         # Create bounding boxes for detected clusters
#         for cid in set(labels):
#             if cid == -1:
#                 continue
#             cluster_pts = inside_points[labels == cid]
#             cluster_pcd = o3d.geometry.PointCloud()
#             cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
#             bbox = cluster_pcd.get_axis_aligned_bounding_box()
#             bbox.color = (1, 0, 0)  # Red for detected cars
#             car_clusters.append(bbox)

#     # Visualize point cloud with parking rectangles and detected car clusters
#     o3d.visualization.draw_geometries([pcd] + parking_lines + car_clusters,
#                                       window_name="Detected Cars in Parking Spots")


# if __name__ == "__main__":
#     main()

import open3d as o3d
import numpy as np

spots = [
    {'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)]},
    {'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)]},
    {'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)]},
    {'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)]},
    {'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)]},
    {'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)]},
    {'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)]},
    {'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)]},
    {'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)]}
]

def polygon_to_bounds(polygon, z_min=-5, z_max=5):
    x_vals = [p[0] for p in polygon]
    y_vals = [p[1] for p in polygon]
    return (min(x_vals), max(x_vals), min(y_vals), max(y_vals), z_min, z_max)

def extract_blocks_from_pcd(pcd_path, polygons, output_path, z_min=-5, z_max=5):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    pcd = pcd.select_by_index(np.where(points[:,2] > 0.5)[0])
    points = np.asarray(pcd.points)

    combined_mask = np.zeros(points.shape[0], dtype=bool)
    for spot in polygons:
        bounds = polygon_to_bounds(spot['polygon'], z_min, z_max)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        combined_mask |= mask

    filtered_points = points[combined_mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[combined_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, filtered_pcd)
    print(f"Filtered PCD saved to: {output_path}")

# Example usage
extract_blocks_from_pcd("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/outputs/basemap_30.pcd", spots, "filtered_output_new.pcd", z_min=-2, z_max=2)
