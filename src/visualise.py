import os
import numpy as np
import open3d as o3d
import glob
from matplotlib import cm
import time

# Directory containing your PCD files
PCD_DIR = "src/pcds"  # Replace with your actual path

# Get all PCD files and sort them numerically
pcd_files = sorted(glob.glob(os.path.join(PCD_DIR, "*.pcd")), 
                   key=lambda x: int(os.path.basename(x).split('.')[0]))

def visualize_single_frame(pcd_path, show_coordinate_frame=True):
    """
    Visualize a single point cloud frame with semantic coloring
    """
    print(f"Visualizing: {pcd_path}")
    
    # Load the point cloud
    pcd = o3d.t.io.read_point_cloud(pcd_path)
    pcd_legacy = pcd.to_legacy()
    
    # Color points by object tag (semantic information)
    if "object_tag" in pcd.point:
        # Get object tags
        tags = pcd.point["object_tag"].numpy().flatten()
        unique_tags = np.unique(tags)
        
        # Create a colormap using matplotlib's tab20 for distinct colors
        cmap = cm.get_cmap('tab20', len(unique_tags))
        colors = np.zeros((len(tags), 3))
        
        # Assign a color to each unique semantic tag
        for i, tag in enumerate(unique_tags):
            mask = tags == tag
            tag_color = cmap(i)[:3]  # Get RGB (exclude alpha)
            colors[mask] = tag_color
            
            # Print semantic class information (useful for reference)
            count = np.sum(mask)
            percentage = (count / len(tags)) * 100
            print(f"Tag {tag}: {count} points ({percentage:.1f}%)")
            
        pcd_legacy.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    geometries = [pcd_legacy]
    
    # Add coordinate frame for reference (optional)
    if show_coordinate_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        geometries.append(coord_frame)
    
    o3d.visualization.draw_geometries(geometries, 
                                     window_name=f"Frame: {os.path.basename(pcd_path)}",
                                     width=1200, 
                                     height=800,
                                     point_show_normal=False,
                                     zoom=0.5)

def create_interactive_animation(pcd_files, start_idx=0, end_idx=None, loop=False):
    """
    Create an interactive animation where you can control playback
    """
    if end_idx is None:
        end_idx = len(pcd_files)
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Animation", width=1200, height=800)
    
    # Add controls to view
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 1.0
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)
    
    # Prepare the first frame
    current_idx = [start_idx]  # Using list for nonlocal access in callbacks
    pcd = o3d.t.io.read_point_cloud(pcd_files[current_idx[0]])
    pcd_legacy = pcd.to_legacy()
    
    # Color the first frame
    if "object_tag" in pcd.point:
        tags = pcd.point["object_tag"].numpy().flatten()
        unique_tags = np.unique(tags)
        cmap = cm.get_cmap('tab20', max(20, len(unique_tags)))
        
        colors = np.zeros((len(tags), 3))
        for i, tag in enumerate(unique_tags):
            mask = tags == tag
            colors[mask] = cmap(i % 20)[:3]
        
        pcd_legacy.colors = o3d.utility.Vector3dVector(colors)
    
    # Add geometry and set view
    vis.add_geometry(pcd_legacy)
    
    # Define callback for updating frames
    def update_frame(vis, action, mods):
        # Remove previous frame
        vis.remove_geometry(pcd_legacy)
        
        # Load new frame
        pcd = o3d.t.io.read_point_cloud(pcd_files[current_idx[0]])
        pcd_legacy = pcd.to_legacy()
        
        # Color points
        if "object_tag" in pcd.point:
            tags = pcd.point["object_tag"].numpy().flatten()
            unique_tags = np.unique(tags)
            cmap = cm.get_cmap('tab20', max(20, len(unique_tags)))
            
            colors = np.zeros((len(tags), 3))
            for i, tag in enumerate(unique_tags):
                mask = tags == tag
                colors[mask] = cmap(i % 20)[:3]
            
            pcd_legacy.colors = o3d.utility.Vector3dVector(colors)
        
        # Add updated frame
        vis.add_geometry(pcd_legacy)
        
        # Update window title with frame info
        frame_num = os.path.basename(pcd_files[current_idx[0]]).split('.')[0]
        vis.get_window().set_title(f"Frame: {frame_num} ({current_idx[0]+1}/{len(pcd_files)})")
        
        return False
    
    # Register key callbacks
    def next_frame(vis):
        current_idx[0] = (current_idx[0] + 1) % len(pcd_files) if loop else min(current_idx[0] + 1, end_idx - 1)
        update_frame(vis, None, None)
        return False
    
    def prev_frame(vis):
        current_idx[0] = (current_idx[0] - 1) % len(pcd_files) if loop else max(current_idx[0] - 1, start_idx)
        update_frame(vis, None, None)
        return False
    
    def play_frames(vis):
        for i in range(current_idx[0], end_idx):
            current_idx[0] = i
            update_frame(vis, None, None)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.5)  # Control speed - adjust as needed
        return False
    
    # Register keyboard callbacks
    vis.register_key_callback(ord('D'), next_frame)  # Press D for next frame
    vis.register_key_callback(ord('A'), prev_frame)  # Press A for previous frame
    vis.register_key_callback(ord('P'), play_frames)  # Press P to play animation
    
    print("\nControls:")
    print("  A: Previous frame")
    print("  D: Next frame")
    print("  P: Play animation")
    print("  Q: Quit")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def colorize_by_height(pcd_path):
    """
    Visualize a point cloud with height-based coloring
    """
    pcd = o3d.t.io.read_point_cloud(pcd_path)
    pcd_legacy = pcd.to_legacy()
    
    # Get points
    points = np.asarray(pcd_legacy.points)
    
    # Color based on height (z-value)
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    
    # Normalize heights to [0, 1] range
    normalized_z = (points[:, 2] - min_z) / (max_z - min_z)
    
    # Apply colormap (jet or viridis are good options for height)
    cmap = cm.get_cmap('viridis')
    colors = cmap(normalized_z)[:, :3]  # RGB values
    
    pcd_legacy.colors = o3d.utility.Vector3dVector(colors)
    
    # Add axis for scale reference
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    o3d.visualization.draw_geometries([pcd_legacy, axis],
                                     window_name=f"Height Map: {os.path.basename(pcd_path)}",
                                     width=1200, 
                                     height=800)

# Example usage
if __name__ == "__main__":
    # Make sure there are PCD files to visualize
    if not pcd_files:
        print(f"No PCD files found in {PCD_DIR}")
        exit(1)
    
    print(f"Found {len(pcd_files)} PCD files")
    
    # 1. Visualize single frame with semantic coloring
    visualize_single_frame(pcd_files[0])
    
    # 2. Create an interactive animation
    create_interactive_animation(pcd_files, start_idx=0, end_idx=None, loop=True)
    
    # 3. Visualize with height-based coloring
    colorize_by_height(pcd_files[0])