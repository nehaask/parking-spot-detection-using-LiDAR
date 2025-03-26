import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import json

def load_basemap(basemap_path):
    """Load the basemap point cloud"""
    print(f"Loading basemap from {basemap_path}")
    return o3d.io.read_point_cloud(basemap_path)

def preprocess_point_cloud(pcd):
    """Preprocess the point cloud to prepare for DBSCAN"""
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    # Ground plane segmentation using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=1000)
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    
    print(f"Ground plane equation: {plane_model}")
    print(f"Found {len(inliers)} ground points out of {len(points)} total points")
    
    # Filter points by height (assuming Z is up)
    ground_points = np.asarray(ground_pcd.points)
    
    return ground_points, ground_pcd, non_ground_pcd, plane_model

def apply_dbscan(points, eps=0.5, min_samples=10):
    """Apply DBSCAN to cluster points"""
    # Project points to 2D (ignore height) for better clustering of ground points
    points_2d = points[:, :2]  # Only X and Y coordinates
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(points_2d)
    
    print(f"Found {len(np.unique(labels))-1} clusters")  # -1 because -1 is noise
    
    return labels

def identify_parking_spots(ground_points, labels, plane_model, min_size=5, max_size=30):
    """Identify potential parking spots from clusters"""
    unique_labels = np.unique(labels)
    parking_spots = []
    
    # Skip label -1 which is noise
    for label in unique_labels:
        if label == -1:
            continue
        
        # Get points in this cluster
        cluster_points = ground_points[labels == label]
        
        # Calculate bounding box
        min_bounds = np.min(cluster_points, axis=0)
        max_bounds = np.max(cluster_points, axis=0)
        size = max_bounds - min_bounds
        
        # Filter by size (typical parking spot is about 2.5m x 5m)
        if (size[0] > min_size and size[0] < max_size and 
            size[1] > min_size and size[1] < max_size):
            
            # Calculate center
            center = (min_bounds + max_bounds) / 2
            
            # Calculate orientation - assume aligned with longest dimension
            if size[0] > size[1]:
                orientation = 0  # Aligned with X-axis
            else:
                orientation = 90  # Aligned with Y-axis
            
            parking_spots.append({
                'center': center.tolist(),
                'size': size.tolist(),
                'orientation': orientation,
                'num_points': len(cluster_points)
            })
    
    print(f"Identified {len(parking_spots)} potential parking spots")
    return parking_spots

def visualize_results(ground_pcd, non_ground_pcd, parking_spots, output_dir):
    """Visualize the results"""
    # Create visualization geometry
    vis_geometries = []
    
    # Add ground and non-ground points
    ground_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for ground
    non_ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-ground
    
    vis_geometries.append(ground_pcd)
    vis_geometries.append(non_ground_pcd)
    
    # Add bounding boxes for parking spots
    for spot in parking_spots:
        center = spot['center']
        size = spot['size']
        
        # Create a box for each parking spot
        box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=0.2)
        box.translate([center[0] - size[0]/2, center[1] - size[1]/2, center[2]])
        box.paint_uniform_color([1.0, 0.0, 0.0])  # Red for parking spots
        
        vis_geometries.append(box)
    
    # Save visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in vis_geometries:
        vis.add_geometry(geom)
    
    # Set view
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = 1.0
    vis.run()
    
    # Save screenshot
    image_path = os.path.join(output_dir, "parking_spots_visualization.png")
    vis.capture_screen_image(image_path)
    print(f"Saved visualization to {image_path}")
    
    vis.destroy_window()

def create_annotated_map(parking_spots, output_dir):
    """Create an annotated map of parking spots as JSON"""
    map_data = {
        'parking_spots': parking_spots,
        'metadata': {
            'num_spots': len(parking_spots),
            'creation_date': str(datetime.datetime.now())
        }
    }
    
    # Save to JSON
    map_path = os.path.join(output_dir, "parking_spots_map.json")
    with open(map_path, 'w') as f:
        json.dump(map_data, f, indent=2)
    
    print(f"Saved annotated map to {map_path}")
    
    return map_data

def visualize_annotated_map(parking_spots, output_dir):
    """Create a 2D visualization of the parking spot map"""
    plt.figure(figsize=(12, 10))
    
    # Plot each parking spot as a rectangle
    for i, spot in enumerate(parking_spots):
        center = spot['center']
        size = spot['size']
        
        # Create rectangle
        rect = plt.Rectangle(
            (center[0] - size[0]/2, center[1] - size[1]/2),
            size[0], size[1],
            linewidth=1, edgecolor='r', facecolor='none', alpha=0.7
        )
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(center[0], center[1], f"Spot {i+1}", 
                 ha='center', va='center', fontsize=8)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Parking Spot Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Save to file
    map_viz_path = os.path.join(output_dir, "parking_spots_2d_map.png")
    plt.savefig(map_viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D map visualization to {map_viz_path}")
    
    plt.close()

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load basemap point cloud
    basemap = load_basemap(args.basemap_path)
    
    # Preprocess the point cloud
    ground_points, ground_pcd, non_ground_pcd, plane_model = preprocess_point_cloud(basemap)
    
    # Apply DBSCAN to cluster ground points
    labels = apply_dbscan(ground_points, eps=args.eps, min_samples=args.min_samples)
    
    # Identify parking spots
    parking_spots = identify_parking_spots(
        ground_points, labels, plane_model, 
        min_size=args.min_size, max_size=args.max_size
    )
    
    # Create annotated map
    map_data = create_annotated_map(parking_spots, args.output_dir)
    
    # Visualize results
    visualize_results(ground_pcd, non_ground_pcd, parking_spots, args.output_dir)
    visualize_annotated_map(parking_spots, args.output_dir)
    
    print("Parking spot detection complete!")

if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description='Detect parking spots using DBSCAN')
    parser.add_argument('--basemap_path', type=str, required=True, default='src/pcds/basemap.pcd',
                        help='Path to the basemap PCD file')
    parser.add_argument('--output_dir', type=str, default='output/', 
                        help='Directory to save results')
    parser.add_argument('--eps', type=float, default=0.5, 
                        help='DBSCAN eps parameter (cluster proximity)')
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='DBSCAN min_samples parameter')
    parser.add_argument('--min_size', type=float, default=2.0, 
                        help='Minimum size of parking spot in meters')
    parser.add_argument('--max_size', type=float, default=6.0, 
                        help='Maximum size of parking spot in meters')
    
    args = parser.parse_args()
    main(args)