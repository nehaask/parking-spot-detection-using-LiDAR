
# ======================
# Environment Setup
# ======================
import glob
import os
import sys
import time
import math
import numpy as np
from sklearn.cluster import DBSCAN
import carla
import matplotlib.pyplot as plt
import random

# CARLA PythonAPI setup
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ======================
# Global Configurations
# ======================
# Predefined parking spots in world coordinates (x, y)
parking_spots = [
    {'id': 10, 
     'polygon': [(1, -31), (6, -31), (6, -29), (1, -29)], 
     'occupied': False
    },

    {'id': 5, 
      'polygon': [(8, -31), (13, -31), (13, -29), (8, -29)],
      'occupied': False
    }, 

    {'id': 9, 
       'polygon': [(1, -28), (6, -28), (6, -26), (1, -26)], 
       'occupied': False
    },

    {'id': 8,
      'polygon': [(1, -25), (6, -25), (6, -23), (1, -23)],
      'occupied': False
    }, 

    {'id': 7,
      'polygon': [(1, -23), (6, -23), (6, -20), (1, -20)],
      'occupied': False
    }, 

    {'id': 6,
     'polygon': [(1, -20), (6, -20), (6, -18), (1, -18)],
     'occupied': False
    }, 

    {'id': 4,
      'polygon': [(8, -28), (13, -28), (13, -26), (8, -26)],
      'occupied': False
    }, 
    
    {'id': 3, 
     'polygon': [(8, -25), (13, -25), (13, -23), (8, -23)],
     'occupied': False
     },
     
    {'id': 2,
      'polygon': [(8, -23), (13, -23), (13, -21), (8, -21)],
      'occupied': False
    }, 
    
    {'id': 1,
     'polygon': [(8, -20), (13, -20), (13, -18), (8, -18)],
     'occupied': False
    }
]

# CARLA semantic segmentation object IDs
OBJECT_IDS = {
    "Ground": 0,
    "ParkingSpace": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example IDs for parking spaces

}

# Initialize CARLA connection
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town05')  # Load map

# Spawn player vehicle
blueprint_lib = world.get_blueprint_library()
vehicle_bp = blueprint_lib.filter('vehicle')[5]
# spawn_point = world.get_map().get_spawn_points()[167]  # Predefined parking area
# spawn_point = carla.Transform(carla.Location(x=1, y=-30, z=0.3), carla.Rotation(yaw=-180))
# vehicle = world.spawn_actor(vehicle_bp, spawn_point)


# --- Find spot with ID 8 ---
spot = next((s for s in parking_spots if s['id'] == 3), None)

if spot is None:
    print("Spot ID 3 not found!")
else:
    poly = spot['polygon']
    center_x = sum(p[0] for p in poly) / len(poly)
    center_y = sum(p[1] for p in poly) / len(poly)

    print(f"Spot ID 3 center: ({center_x}, {center_y})")

    spawn_location = carla.Location(x=center_x, y=center_y, z=0.3)
    spawn_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # adjust based on orientation
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)

    if vehicle:
        print("✅ Vehicle spawned successfully at Spot ID 8!")
        spot['occupied'] = True
    else:
        print("❌ Vehicle spawn failed. Try adjusting Z or checking collision.")
