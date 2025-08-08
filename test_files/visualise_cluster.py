import open3d as o3d
pcd = o3d.io.read_point_cloud("/home/nk4349/Desktop/Carla/PythonAPI/examples/important/buffered_lidar.txt", format='xyz')
o3d.visualization.draw_geometries([pcd])
