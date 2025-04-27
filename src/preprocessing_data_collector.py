import carla
import open3d as o3d
import numpy as np
import os
import time
from queue import Queue

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

# LiDAR callback queue
lidar_queue = Queue()

def semantic_lidar_callback(world, data):
    point_cloud =  np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    
    print(point_cloud.shape)

    snapshot = world.get_snapshot()
    timestamp_seconds = snapshot.timestamp.elapsed_seconds

    x = point_cloud['x']
    y = point_cloud['y']
    z = point_cloud['z']
    CosAngle = point_cloud['CosAngle']
    ObjIdx = point_cloud['ObjIdx']
    ObjTag = point_cloud['ObjTag']

    # Construct the points array
    points = np.array([x, y, z]).T
    lidar_queue.put((data.frame, points, ObjIdx, ObjTag, data.transform, timestamp_seconds))

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
    spawn_point = carla.Transform(carla.Location(x=15, y=-30, z=0.5), carla.Rotation(yaw=90))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)

    # Configure spectator camera
    spectator = world.get_spectator()
    
    def follow_vehicle_camera():
        """Maintain top-down view of vehicle"""
        transform = vehicle.get_transform()
        # Position camera 40m above vehicle
        location = transform.location + carla.Location(z=40)
        # Directly downward view (-90 degree pitch)
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
            carla.Rotation(pitch=0, yaw=90, roll=0)  # Forward-facing
        )
    
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(lambda data: semantic_lidar_callback(world, data))
    current_frame = 1
    end = 10 * 10
    batch_size = 10

    points_all = []
    object_id_all = []
    object_tag_all = []

    try:
        while current_frame:
            world.tick()
            print(current_frame)

            s_car = lidar_queue.get(True, 5)
            frame, pcl, object_id, object_tag, transform, timestamp = s_car

            points = pcl[:, 0:3]

            follow_vehicle_camera()       # Keep camera following vehicle
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.5,  # Maintain forward motion
                steer=0.0      # No steering input
            ))
            time.sleep(0.1)  # Control loop frequency (~10Hz)

            # Transform points
            points_t = np.append(points, np.ones((points.shape[0], 1)), axis=1)
            points_t = np.dot(transform.get_matrix(), points_t.T).T[:, :-1]

            # Accumulate for basemap every 10 frames
            points_all.append(points_t)
            object_id_all.append(object_id)
            object_tag_all.append(object_tag)

            # Save individual frame's point cloud
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
            filename = f"outputs/pcds/{current_frame}.pcd"
            o3d.t.io.write_point_cloud(filename, pcd, write_ascii=False)

            # Save pose and timestamp
            t = [current_frame, transform.location.x, transform.location.y, transform.location.z,
                transform.rotation.yaw, transform.rotation.pitch, transform.rotation.roll]

            # Every 10 frames, compute and reset
            if current_frame % batch_size == 0:
                print(f"Saving basemap batch ending at frame {current_frame}")
                combined_points = np.concatenate(points_all, axis=0)
                combined_ids = np.concatenate(object_id_all)
                combined_tags = np.concatenate(object_tag_all)

                pcd_all = o3d.t.geometry.PointCloud()
                pcd_all.point["positions"] = o3d.core.Tensor(combined_points, o3d.core.float32)
                pcd_all.point["object_id"] = o3d.core.Tensor(combined_ids.reshape((-1, 1)), o3d.core.uint32)
                pcd_all.point["object_tag"] = o3d.core.Tensor(combined_tags.reshape((-1, 1)), o3d.core.uint32)

                filename = f"outputs/basemap_{current_frame}.pcd"
                o3d.t.io.write_point_cloud(filename, pcd_all, write_ascii=False)
                print("Basemap saved")

                # Reset accumulators for next batch
                points_all.clear()
                object_id_all.clear()
                object_tag_all.clear()

            if current_frame == end:
                world.apply_settings(original_settings)
                break

            current_frame += 1

    finally:
        world.apply_settings(original_settings)

        for actor in actor_list:
            actor.destroy()

        lidar.destroy()


if __name__ == '__main__':
    main()
