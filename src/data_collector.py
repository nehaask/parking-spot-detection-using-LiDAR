import carla
import open3d as o3d
import numpy as np
import os
from queue import Queue

# LiDAR callback queue
lidar_queue = Queue()

def semantic_lidar_callback(data):
    point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    x, y, z = point_cloud['x'], point_cloud['y'], point_cloud['z']
    points = np.array([x, y, z]).T
    lidar_queue.put(points)

def main():
    output_dir = "output/pcds"
    os.makedirs(output_dir, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world('Town05')
    world = client.get_world()
    actor_list = []

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.1
    settings.synchronous_mode = True
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

    # Two custom spawn points
    spawn_point1 = carla.Transform(carla.Location(x=11, y=-30, z=0.5), carla.Rotation(yaw=-180))
    spawn_point2 = carla.Transform(carla.Location(x=11, y=-24, z=0.5), carla.Rotation(yaw=-180))

    vehicle1 = world.spawn_actor(vehicle_bp, spawn_point1)
    vehicle2 = world.spawn_actor(vehicle_bp, spawn_point2)
    actor_list.extend([vehicle1, vehicle2])

    vehicle1.set_autopilot(True)

    spectator_bp = world.get_spectator()
    spectator_bp.set_transform(spawn_point1)

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('points_per_second', '1310720')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov', '21.2')
    lidar_bp.set_attribute('lower_fov', '-21.2')

    lidar_transform = carla.Transform(carla.Location(z=1))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle1)
    lidar.listen(lambda data: semantic_lidar_callback(data))

    frame = 1
    try:
        while frame <= 20:
            world.tick()
            points = lidar_queue.get(timeout=5)

            # Save point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(f"{output_dir}/{frame}.pcd", pcd)
            print(f"Saved frame {frame}.pcd")
            frame += 1

    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        lidar.destroy()
        print("Simulation ended.")

if __name__ == '__main__':
    main()
