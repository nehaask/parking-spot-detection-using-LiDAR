import open3d as o3d
import numpy as np

# spots = [
#     {'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)]},
#     {'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)]},
#     {'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)]},
#     {'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)]},
#     {'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)]},
#     {'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)]},
#     {'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)]},
#     {'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)]},
#     {'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)]}
# ]
spots = [
    {'id': 10, 'polygon': [(1.2, -30.8), (6.2, -30.8), (6.2, -28.8), (1.2, -28.8)], 'occupied': False},
    {'id': 5, 'polygon': [(8.2, -30.8), (13.2, -30.8), (13.2, -28.8), (8.2, -28.8)], 'occupied': False},
    {'id': 9, 'polygon': [(1.2, -27.8), (6.2, -27.8), (6.2, -25.8), (1.2, -25.8)], 'occupied': False},
    {'id': 8, 'polygon': [(1.2, -24.8), (6.2, -24.8), (6.2, -22.8), (1.2, -22.8)], 'occupied': False},
    {'id': 7, 'polygon': [(1.2, -22.8), (6.2, -22.8), (6.2, -19.8), (1.2, -19.8)], 'occupied': False},
    {'id': 6, 'polygon': [(1.2, -19.8), (6.2, -19.8), (6.2, -17.8), (1.2, -17.8)], 'occupied': False},
    {'id': 4, 'polygon': [(8.2, -27.8), (13.2, -27.8), (13.2, -25.8), (8.2, -25.8)], 'occupied': False},
    {'id': 3, 'polygon': [(8.2, -24.8), (13.2, -24.8), (13.2, -22.8), (8.2, -22.8)], 'occupied': False},
    {'id': 2, 'polygon': [(8.2, -22.8), (13.2, -22.8), (13.2, -20.8), (8.2, -20.8)], 'occupied': False},
    {'id': 1, 'polygon': [(8.2, -19.8), (13.2, -19.8), (13.2, -17.8), (8.2, -17.8)], 'occupied': False}
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
