import open3d as o3d
import os
from pathlib import Path

# Paths
# input_dir = Path("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/basemaps/output_parking_lot/pcds")
input_dir = Path("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/output/pcds")
output_dir = Path("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/src/accumulated")
output_dir.mkdir(parents=True, exist_ok=True)

# Helper to read multiple point clouds and merge
def accumulate_point_clouds(file_list):
    combined_pcd = o3d.geometry.PointCloud()
    for file_path in file_list:
        pcd = o3d.io.read_point_cloud(str(file_path))
        combined_pcd += pcd
    return combined_pcd

# Get sorted list of all pcd files
pcd_files = sorted(input_dir.glob("*.pcd"), key=lambda x: int(x.stem))

# Process in chunks of 10
chunk_size = 100
for i in range(0, len(pcd_files), chunk_size):
    chunk = pcd_files[i:i+chunk_size]
    print(f"Processing frames: {chunk[0].name} to {chunk[-1].name}")
    
    accumulated_pcd = accumulate_point_clouds(chunk)

    # Optional: downsample for faster viewing
    downsampled_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.2)

    # Visualize
    o3d.visualization.draw_geometries([downsampled_pcd])

    # Save accumulated frame
    out_file = output_dir / f"accumulated_{i+1}_{i+len(chunk)}.pcd"
    o3d.io.write_point_cloud(str(out_file), accumulated_pcd)
    print(f"Saved: {out_file.name}")
