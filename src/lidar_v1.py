import carla
import numpy as np
import open3d as o3d  # Install using: pip install open3d

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Spawn a vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model 3
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Attach LiDAR sensor
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '500000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '100')

    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Function to process LiDAR data
    def process_lidar(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)  # x, y, z, intensity

        # Convert to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use only x, y, z

        # Save as .ply file
        o3d.io.write_point_cloud("lidar_scan.ply", pcd)
        print("Saved LiDAR scan as 'lidar_scan.ply'.")

    # Attach LiDAR processing function
    lidar_sensor.listen(lambda data: process_lidar(data))

    try:
        import time
        time.sleep(5)  # Run for 5 seconds before stopping
    finally:
        lidar_sensor.destroy()
        vehicle.destroy()
        print("Cleaned up actors")

if __name__ == '__main__':
    main()
