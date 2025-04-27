import open3d as o3d
import numpy as np

spots = [
    {'id': 10, 'polygon': [(1.2, -31.8), (6.2, -31.8), (6.2, -29.8), (1.2, -29.8)], 'occupied': False},
    {'id': 5,  'polygon': [(8.2, -31.8), (13.2, -31.8), (13.2, -29.8), (8.2, -29.8)], 'occupied': False},

    {'id': 9,  'polygon': [(1.2, -28.8), (6.2, -28.8), (6.2, -26.8), (1.2, -26.8)], 'occupied': False},
    {'id': 4,  'polygon': [(8.2, -28.8), (13.2, -28.8), (13.2, -26.8), (8.2, -26.8)], 'occupied': False},

    {'id': 8,  'polygon': [(1.2, -25.8), (6.2, -25.8), (6.2, -23.8), (1.2, -23.8)], 'occupied': False},
    {'id': 3,  'polygon': [(8.2, -25.8), (13.2, -25.8), (13.2, -23.8), (8.2, -23.8)], 'occupied': False},

    {'id': 7,  'polygon': [(1.2, -22.8), (6.2, -22.8), (6.2, -20.8), (1.2, -20.8)], 'occupied': False},
    {'id': 2,  'polygon': [(8.2, -22.8), (13.2, -22.8), (13.2, -20.8), (8.2, -20.8)], 'occupied': False},

    {'id': 6,  'polygon': [(1.2, -19.8), (6.2, -19.8), (6.2, -17.8), (1.2, -17.8)], 'occupied': False},
    {'id': 1,  'polygon': [(8.2, -19.8), (13.2, -19.8), (13.2, -17.8), (8.2, -17.8)], 'occupied': False}
]

def create_bbox_lineset(bounds, color=(0, 1, 0)):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Define the 8 vertices of the bounding box
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    # Define the lines connecting the vertices
    lines = [
        [0,1], [1,2], [2,3], [3,0],  # Bottom face
        [4,5], [5,6], [6,7], [7,4],  # Top face
        [0,4], [1,5], [2,6], [3,7]   # Vertical connections
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set


def visualize_parking_spots(pcd, polygons, z_min=-2, z_max=2):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the point cloud
    vis.add_geometry(pcd)
    
    # Add parking spot bounding boxes
    for spot in polygons:
        bounds = polygon_to_bounds(spot['polygon'], z_min, z_max)
        bbox = create_bbox_lineset(bounds)
        vis.add_geometry(bbox)
    
    # Configure view settings
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2
    
    vis.run()
    vis.destroy_window()


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

    # visualize_parking_spots(filtered_pcd, polygons, z_min, z_max)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[combined_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, filtered_pcd)
    print(f"Filtered PCD saved to: {output_path}")

# Example usage
extract_blocks_from_pcd("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/outputs/basemap_30.pcd", spots, "filtered_basemap_new.pcd", z_min=-2, z_max=2)
