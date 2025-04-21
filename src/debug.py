import carla
import random

# Step 1: Define parking spots
parking_spots = [
    {'id': 10, 'polygon': [(1, -31), (6, -31), (6, -29), (1, -29)], 'occupied': False},
    {'id': 5,  'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)], 'occupied': False},
    {'id': 9,  'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)], 'occupied': False},
    {'id': 8,  'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)], 'occupied': False},
    {'id': 7,  'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)], 'occupied': False},
    {'id': 6,  'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)], 'occupied': False},
    {'id': 4,  'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)], 'occupied': False},
    {'id': 3,  'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)], 'occupied': False},
    {'id': 2,  'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)], 'occupied': False},
    {'id': 1,  'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)], 'occupied': False}
]

# Step 2: Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Step 3: Choose an unoccupied spot
available_spots = [spot for spot in parking_spots if not spot['occupied']]
spot = random.choice(available_spots)

# Step 4: Compute center of the polygon
poly = spot['polygon']
center_x = sum(p[0] for p in poly) / len(poly)
center_y = sum(p[1] for p in poly) / len(poly)

# Set z to a reasonable height above the ground
spawn_location = carla.Location(x=center_x, y=center_y, z=0.5)

# Face the car toward the road (adjust yaw based on orientation of parking slot)
spawn_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # or 0.0 / 90.0 etc. as needed
spawn_transform = carla.Transform(spawn_location, spawn_rotation)

# Step 5: Choose vehicle blueprint
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*model3')[0]  # Tesla Model 3, for example

# Step 6: Spawn vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

if vehicle:
    print(f"Vehicle spawned at spot ID {spot['id']}!")
    spot['occupied'] = True
else:
    print("Failed to spawn vehicle.")
