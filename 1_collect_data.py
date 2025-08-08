import carla
import open3d as o3d
import numpy as np
import os
import time
from queue import Queue

# LiDAR callback queue
lidar_queue = Queue()

def semantic_lidar_callback(world, data):
    point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    snapshot = world.get_snapshot()
    timestamp_seconds = snapshot.timestamp.elapsed_seconds

    x = point_cloud['x']
    y = point_cloud['y']
    z = point_cloud['z']
    ObjIdx = point_cloud['ObjIdx']
    ObjTag = point_cloud['ObjTag']

    points = np.array([x, y, z]).T
    lidar_queue.put((data.frame, points, ObjIdx, ObjTag, data.transform, timestamp_seconds))

def main():
    output_dir = "outputs_test/pcds"
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

    spawn_point = carla.Transform(carla.Location(x=15, y=-30, z=0.5), carla.Rotation(yaw=90))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)

    spectator = world.get_spectator()
    
    def follow_vehicle_camera():
        transform = vehicle.get_transform()
        location = transform.location + carla.Location(z=40)
        rotation = carla.Rotation(pitch=-90)
        spectator.set_transform(carla.Transform(location, rotation))

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('points_per_second', '1310720')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov', '21.2')
    lidar_bp.set_attribute('lower_fov', '-21.2')

    lidar_transform = carla.Transform(
        carla.Location(x=0, y=0, z=3.5),
        carla.Rotation(pitch=0, yaw=90, roll=0)
    )

    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(lambda data: semantic_lidar_callback(world, data))
    
    current_frame = 1
    end = 10 * 10  # 10 seconds at 10 Hz

    try:
        while current_frame <= end:
            world.tick()
            print(f"Collecting Frame: {current_frame}")

            s_car = lidar_queue.get(True, 5)
            frame, pcl, object_id, object_tag, transform, timestamp = s_car

            points = pcl[:, 0:3]

            follow_vehicle_camera()
            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
            time.sleep(0.1)

            # Transform points to world coordinates
            points_t = np.append(points, np.ones((points.shape[0], 1)), axis=1)
            points_t = np.dot(transform.get_matrix(), points_t.T).T[:, :-1]

            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points_t, o3d.core.float32)
            pcd.point["object_id"] = o3d.core.Tensor(object_id.reshape((-1, 1)), o3d.core.uint32)
            pcd.point["object_tag"] = o3d.core.Tensor(object_tag.reshape((-1, 1)), o3d.core.uint32)

            filename = f"{output_dir}/{current_frame:04d}.pcd"
            o3d.t.io.write_point_cloud(filename, pcd, write_ascii=False)

            current_frame += 1

    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        lidar.destroy()

if __name__ == '__main__':
    main()
