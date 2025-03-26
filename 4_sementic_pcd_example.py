import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
from queue import Queue, Empty
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla

"""
# LiDAR configuration (OS1-128)
fps = 10
lidar_range = 120
lidar_channels = 128
points_per_cloud = 262144
lidar_upper_fov = 22.5
lidar_lower_fov = -22.5
"""

# LiDAR configuration (OS0-64)
fps = 10
lidar_range = 50
lidar_channels = 128
points_per_cloud = 200000
lidar_upper_fov = 45
lidar_lower_fov = -45

lidar_sensorQueue = Queue()


def lidar_car_callback(data):
    point_cloud = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    print(point_cloud.shape)

    # point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
    # print(point_cloud.shape)

    lidar_sensorQueue.put((data.frame, point_cloud, data.transform))


def semantic_lidar_callback(data):
    point_cloud =  np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    
    print(point_cloud.shape)

    x = point_cloud['x']
    y = point_cloud['y']
    z = point_cloud['z']
    CosAngle = point_cloud['CosAngle']
    ObjIdx = point_cloud['ObjIdx']
    ObjTag = point_cloud['ObjTag']

    # Construct the points array
    points = np.array([x, y, z]).T
    # point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 6), 6))

    lidar_sensorQueue.put((data.frame, points, ObjIdx, ObjTag, data.transform))


def generate_lidar_bp(arg, blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""

    if arg.semantic:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
            #lidar_bp.set_attribute('dropoff_general_rate', '0.45')
            #lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
            #lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
            #lidar_bp.set_attribute('noise_stddev', '0.2')
        else:
            lidar_bp.set_attribute('dropoff_general_rate',arg.dropoff)
            lidar_bp.set_attribute('noise_stddev', arg.noise)

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_cloud*arg.fps))
    lidar_bp.set_attribute('rotation_frequency', str(arg.fps))

    return lidar_bp


def main(arg):
    """Main function of the script"""

    if not os.path.exists(os.path.join(arg.output, "pcds")):
        os.makedirs(os.path.join(arg.output, "pcds"))

    # load client and world
    client = carla.Client(arg.host, arg.port)
    # client.set_timeout(2.0)
    client.load_world(arg.map)
    world = client.get_world()

    try:
        #Set configs
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        settings.fixed_delta_seconds = 1. / arg.fps
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)
        
        # filter blueprints of cars
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [x for x in vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith(('isetta','carlacola','cybertruck','t2'))]
        
        # creating custom route
        spawn_points = world.get_map().get_spawn_points()
        # [X, Y] = [start route at spawn point X, end point at spawn point Y]
        route_indices = [[57,135],[138,86],[137,101], [135,100], [136,41],[154,42],[295,11],[294,12]] #manually selected route points from spawn points

        route_points=[]

        for route in route_indices:
            points= []
            for ind in route:
                points.append(spawn_points[ind])
            route_points.append(points)
    
        
        # basemap = 1 #variale to spawn only one vehicle in map
        for i, route in enumerate(route_points):
            # if basemap==1:
            vehicle = world.spawn_actor(vehicle_blueprints[i], route[0])
            traffic_manager.set_path(vehicle, [transform.location for transform in route[1:]])
            vehicles.append(vehicle)
                # basemap = basemap+1; 

        # set autopilot
                # basemap = 1 #variale to spawn only one vehicle in map

        for vehicle in vehicles:
            vehicle.set_autopilot(True)

        # create car lidar sensor on primary vehicle and registers callback
        ego_vehicle=vehicles[0]
        lidar_car_bp = generate_lidar_bp(arg, blueprint_library)
        hp = max(ego_vehicle.bounding_box.extent.x,ego_vehicle.bounding_box.extent.y)*np.tan(np.radians(-arg.lower_fov))
        lidar_car_transform = carla.Transform(carla.Location(z=2*ego_vehicle.bounding_box.extent.z+hp))
        lidar_car= world.spawn_actor(lidar_car_bp, lidar_car_transform, attach_to=ego_vehicle)
        sensors.append(lidar_car)
        if arg.semantic:
            lidar_car.listen(lambda data : semantic_lidar_callback(data))
        else:
            lidar_car.listen(lambda data : lidar_car_callback(data))

        # create Spectator
        spectator = world.get_spectator()
        spectator_point = carla.Location(spawn_points[135].location.x, spawn_points[135].location.y ,5)
        spectator.set_transform(carla.Transform(spectator_point , carla.Rotation(0,0,0))) 

        current_frame = 1
        end = arg.t * arg.fps


        while True:
            world.tick()
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    # world.hud.notification("Traffic light changed! Good to go!")
                    traffic_light.set_state(carla.TrafficLightState.Green)


            print(current_frame)
            snapshot = world.get_snapshot()
            timestamp_seconds = snapshot.timestamp.elapsed_seconds
            print(timestamp_seconds)

            
            #elements in the queue
            for i in range(lidar_sensorQueue.qsize()):
                print(lidar_sensorQueue.get())
            # Reading data from vehicle Queuq
            s_car= lidar_sensorQueue.get(True,5)
            spectator.set_transform(vehicle.get_transform())

            frame = s_car[0]
            pcl = s_car[1]
            object_id = s_car[2]
            object_tag = s_car[3]
            transform = s_car[4]
            
            # # Separating points and intensities
            points = pcl[:, 0:3]
            # cosAngle = pcl[:,3]
            # object_id = pcl[:,4]
            # object_tag = pcl[:,5]

             # Creating collevtive point cloud
            # transform points
            points_t = np.append(points, np.ones((points.shape[0], 1)), axis=1)
            points_t = np.dot(transform.get_matrix(), points_t.T).T
            points_t = points_t[:, :-1]
            if current_frame ==1:
                points_all = points_t
            else:
                points_all = np.append(points_all,points_t, axis=0)

            
            #Saving individual point cloud
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
            # pcd.point["cosAngle"] = o3d.core.Tensor(cosAngle.reshape((-1,1)), o3d.core.float32)
            pcd.point["object_id"] = o3d.core.Tensor(object_id.reshape((-1,1)), o3d.core.uint32)
            pcd.point["object_tag"] = o3d.core.Tensor(object_tag.reshape((-1,1)), o3d.core.uint32)
            filename =  arg.output+'/pcds/{}.pcd'.format(current_frame)
            o3d.t.io.write_point_cloud(filename, pcd, write_ascii=True)
            
                
            if current_frame==end:
                print("saving basemamp")
                filename =  args.output +'/basemap.pcd'
                # pcd_all = o3d.t.geometry.PointCloud()
                # pcd_all.point["positions"] = o3d.core.Tensor(points_all, o3d.core.float32)
                # o3d.t.io.write_point_cloud(filename, pcd_all, write_ascii=False)
                pcd_all = o3d.geometry.PointCloud()
                pcd_all.points = o3d.utility.Vector3dVector(points_all)
                o3d.io.write_point_cloud(filename, pcd_all, write_ascii=False)
                print("done") 
            
                world.apply_settings(original_settings)
                traffic_manager.set_synchronous_mode(False)
 
                break

            current_frame+=1

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        # for vehicle in vehicles:
        #     vehicle.destroy()
        # for sensor in sensors:
        #     sensor.destroy()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        metavar='M',
        default='Town05',
        type=str,
        help='Map name (default: Town05')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--dropoff',
        default='0.45',
        type=str,
        help='dropoff value')
    argparser.add_argument(
        '--noise',
        default='0.0',
        type=str,
        help='noise value')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
   
    argparser.add_argument(
        '--fps',
        default=10.0,
        type=float,
        help='frames per second, define the fixed simulation time-steps. (default: 10fps)')
    argparser.add_argument(
        '--upper-fov',
        default=45.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-45.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=128.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=200.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-cloud',
        default=100000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    
    argparser.add_argument(
        '-t',
        default=10,
        type=int,
        help='number of seconds to run simulation')
   
    argparser.add_argument(
        '--seed',
        default=int(time.time()),
        type=int,
        help='Random seed for reproducibility (default: time.time())')
    argparser.add_argument(
        '--output',
        help='specify the path of output folder')
    args = argparser.parse_args()

    try:
        vehicles=[]
        sensors=[]
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
