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
lidar_range = 170
lidar_channels = 128 / 64 / 32 
points_per_cloud = 524288 / 262144 / 131072
lidar_upper_fov = 21.2
lidar_lower_fov = -21.2



#LiDAR configuration (OS0-64)
fps = 10
lidar_range = 50
lidar_channels = 64
points_per_cloud = 131072
lidar_upper_fov = 45
lidar_lower_fov = -45

"""

lidar_car_sensorQueue = Queue()

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
    # point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 6), 6))

    lidar_car_sensorQueue.put((data.frame, points, ObjIdx, ObjTag, data.transform, timestamp_seconds))

def lidar_car_callback(world, data):
    point_cloud = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

    snapshot = world.get_snapshot()
    timestamp_seconds = snapshot.timestamp.elapsed_seconds
    lidar_car_sensorQueue.put((data.frame, point_cloud, data.transform, timestamp_seconds))
    
def generate_lidar_bp(arg, blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')

    if arg.lidar_noise:
        # lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        # lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        # lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        #lidar_bp.set_attribute('dropoff_general_rate', '0.45')
        #lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
        #lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
        lidar_bp.set_attribute('noise_stddev', arg.lidar_noise_std)
       

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_cloud*arg.fps))
    lidar_bp.set_attribute('rotation_frequency', str(arg.fps))

    return lidar_bp


def main(arg):
    """Main function of the script"""

    # make output folders
    if not os.path.exists(os.path.join(arg.output, "pcds")):
        os.makedirs(os.path.join(arg.output, "pcds"))


    # load client and world
    client = carla.Client(arg.host, arg.port)
    # client.set_timeout(20.0)
    client.load_world(arg.map)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(8000)

    try:
        #Set configs
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

        spawn_points_lane_1 =[237, 46, 220, 188, 246, 210, 112, 114, 163, 206, 178, 172]
        spawn_points_lane_2 =[238, 47, 221, 187, 245, 211, 113, 118, 162, 204, 177, 171]
        spawn_points_lane_2_before_round_abt =[238, 47, 221, 187]
        spawn_points_lane_4 = [129, 240, 0, 122, 218, 232, 230, 212, 42, 40, 38, 160, 202, 179]
        spawn_points_lane_4_before_round_abt = [129, 240, 0, 122, 218, 232]
        spawn_points_connecing_lane = [130, 133, 134, 132, 241, 242, 48, 136, 228, 229, 165, 69, 164, 67, 207, 167, 168]

        route_1=[131, 233, 180]
        parking_lot = [167, 176, 203]

        # vehicle_locations= [spawn_points_lane_1, spawn_points_lane_2, spawn_points_lane_4, spawn_points_connecing_lane]
        # vehicle_locations= [spawn_points_lane_2_before_round_abt , spawn_points_lane_4_before_round_abt]
        vehicle_locations= spawn_points_lane_1
        route=parking_lot
        
        spawn_points = world.get_map().get_spawn_points()

        ## UNCOMMENT HERE TO DISPLAY SPAWN POINTS

        # text_color = carla.Color(255, 255, 0) # Yellow color for text
        # # Display each spawn point's index number
        # for i, spawn_point in enumerate(spawn_points):
        # # Draw the index number above each spawn point
        #     world.debug.draw_string(
        #     spawn_point.location + carla.Location(z=1.0), # Slightly above the point for visibility
        #     str(i), # Display the index of the spawn point
        #     draw_shadow=True,
        #     color=text_color,
        #     life_time=100, # 0 means the text persists until reset
        #     persistent_lines=True # Makes the text persistent
        #     )


        ## TILL HERE

        route_points=[]
        for ind in route:
            route_points.append(spawn_points[ind])
    
        vehicle = world.spawn_actor(vehicle_blueprints[0], route_points[0])
        vehicles.append(vehicle)
        traffic_manager.set_path(vehicle, [transform.location for transform in route_points[1:]])

        # set autopilot
        vehicle.set_autopilot(True)
        # vehicle.set_simulate_physics(True)

        if arg.npc:
            for loc in vehicle_locations:
                loc = spawn_points[loc]
                temp = world.try_spawn_actor(random.choice(vehicle_blueprints), loc)
                vehicles.append(temp)

        ## below code add vehicles in the empty spaces between spawn points

        # if arg.fillGap:
        #     while temp is not None:
        #         loc.location.y -= temp.bounding_box.extent.x *3
        #         if loc.location.y > pre_y: 
        #             temp = world.try_spawn_actor(random.choice(vehicle_blueprints), loc)
        #             # if temp is not None:
        #             #     vehicles.append(temp)
        #         else:
        #             pre_y = spawn_points[loc].location.y
        #             break


        # create car lidar sensor and registers callback
        lidar_car_bp = generate_lidar_bp(arg, blueprint_library)
        hp = max(vehicle.bounding_box.extent.x,vehicle.bounding_box.extent.y)*np.tan(np.radians(-arg.lower_fov))
        lidar_car_transform = carla.Transform(carla.Location(z=2*vehicle.bounding_box.extent.z+hp))
        lidar_car= world.spawn_actor(lidar_car_bp, lidar_car_transform, attach_to=vehicle)
        sensors.append(lidar_car)
        lidar_car.listen(lambda data : semantic_lidar_callback(world, data))

        # create Spectator
        spectator = world.get_spectator()
        spectator.set_transform(route_points[0]) 
        # spectator.set_transform(carla.Transform(vehicle.location , vehicle.rotation)) #+ vehicle.location(20,0,10)

        current_frame = 1
        end = arg.t * arg.fps

        while True:
            world.tick()
            if vehicle.is_at_traffic_light():
               traffic_light = vehicle.get_traffic_light()
               if traffic_light.get_state() == carla.TrafficLightState.Red:
                   #world.hud.notification("Traffic light changed! Good to go!")
                   traffic_light.set_state(carla.TrafficLightState.Green)


            print(current_frame)

            # Reading data from vehicle lidar Queuq
            s_car= lidar_car_sensorQueue.get(True,5)

            frame = s_car[0]
            pcl = s_car[1]
            object_id = s_car[2]
            object_tag = s_car[3]
            transform = s_car[4]
            timestamp = s_car[5]

            # # Separating points and intensities
            points = pcl[:, 0:3]
            # cosAngle = pcl[:,3]
            # object_id = pcl[:,4]
            # object_tag = pcl[:,5]

            spectator.set_transform(vehicle.get_transform())

            # transform points
            points_t = np.append(points, np.ones((points.shape[0], 1)), axis=1)
            points_t = np.dot(transform.get_matrix(), points_t.T).T
            points_t = points_t[:, :-1]

            pcd_t = o3d.t.geometry.PointCloud()
            pcd_t.point["positions"] = o3d.core.Tensor(points_t, o3d.core.float32)
            pcd_t.point["object_id"] = o3d.core.Tensor(object_id.reshape((-1,1)), o3d.core.uint32)
            pcd_t.point["object_tag"] = o3d.core.Tensor(object_tag.reshape((-1,1)), o3d.core.uint32)

            if (arg.basemap):
                if current_frame ==1:
                    points_all = points_t
                    object_id_all = object_id
                    object_tag_all = object_tag

                else:
                    points_all = np.append(points_all,points_t, axis=0)
                    object_id_all =  np.append(object_id_all,object_id)
                    object_tag_all =  np.append(object_tag_all,object_tag)

            # Saving individual point cloud
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points, o3d.core.float32)
            # pcd.point["intensity"] = o3d.core.Tensor(intensity.reshape((-1,1)), o3d.core.float32)
            filename =  arg.output+'/pcds/{}.pcd'.format(current_frame)
            o3d.t.io.write_point_cloud(filename, pcd, write_ascii=False)


            # Saving pose
            t = [current_frame, transform.location.x,transform.location.y,transform.location.z, transform.rotation.yaw,transform.rotation.pitch,transform.rotation.roll]
            
            content_pose = ', '.join(map(str,t))
            if current_frame ==1:
                file = open(arg.output+"/pose.txt", "w")
                file2 = open(arg.output+"/timestamp.txt", "w")

            else:
                file = open(arg.output+"/pose.txt", "a")
                file2 = open(arg.output+"/timestamp.txt", "a")

            file.write(content_pose +'\n')
            file2.write(str(timestamp)+'\n')
            file.close()
            file2.close()

            if current_frame==end:
                if (arg.basemap):
                    print("saving basemamp")
                    filename =  args.output +'/basemap.pcd'
                    pcd_all = o3d.t.geometry.PointCloud()
                    pcd_all.point["positions"] = o3d.core.Tensor(points_all, o3d.core.float32)
                    pcd_all.point["object_id"] = o3d.core.Tensor(object_id_all.reshape((-1,1)), o3d.core.uint32)
                    pcd_all.point["object_tag"] = o3d.core.Tensor(object_tag_all.reshape((-1,1)), o3d.core.uint32)
                    o3d.t.io.write_point_cloud(filename, pcd_all, write_ascii=False)
                    print("done")
                world.apply_settings(original_settings)
                traffic_manager.set_synchronous_mode(False)
                break
        
            current_frame+=1

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        for vehicle in vehicles:
            vehicle.destroy()
        for sensor in sensors:
            sensor.destroy()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    
    ## General Parameters
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
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '-m', '--map',
        metavar='M',
        default='Town05',
        type=str,
        help='Map name (default: Town03)')
    argparser.add_argument(
        '--fps',
        default=10.0,
        type=float,
        help='frames per second, define the fixed simulation time-steps. (default: 10fps)')
    argparser.add_argument(
        '-t',
        default=55,#55 to cover the complere route, 20 to go just to round about
        type=int,
        help='number of seconds to run simulation')
    argparser.add_argument(
        '--output',
        help='specify the path of output folder')
    argparser.add_argument(
        '--basemap',
        action='store_true',
        help='creates and save basemap')
    argparser.add_argument(
        '--npc',
        action='store_true',
        help='to spawn npc')
    argparser.add_argument(
        '--lanes',
        default= 0 ,
        type=int,
        help='add vehicles in lanes other than the ego vehicle')
    argparser.add_argument(
        '--fillGap',
        action='store_true',
        help='add vehicles in gaps between spawn points')

    ## LIDAR Parameters
    argparser.add_argument(
        '--upper-fov',
        default=21.2,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-21.2,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=32.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=170.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-cloud',
        default=131072,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    #524288 / 262144 / 131072
    argparser.add_argument(
        '--lidar_noise',
        action='store_true',
        help='add noise to lidar measuremnt')
    argparser.add_argument(
        '--lidar_noise_std',
        default='0.1',
        type=str,
        help='standard deviation of lidar noise')
    
    ## IMU parameters
    argparser.add_argument(
        '--imu_noise',
        action='store_true',
        help='add noise to imu measuremnt')
    argparser.add_argument(
        '--imu_noise_std',
        default='0.1',
        type=str,
        help='standard deviation of imu noise')
    argparser.add_argument(
        '--imu_bias',
        action='store_true',
        help='add bias to imu measuremnt')
    argparser.add_argument(
        '--imu_bias_std',
        default='0.1',
        type=str,
        help='standard deviation of imu noise')
    
    args = argparser.parse_args()

    try:
        vehicles=[]
        sensors=[]
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')

