import open3d as o3d

# Load the point cloud using Open3D's tensor-based API
# pcd = o3d.t.io.read_point_cloud("/home/nk4349/Desktop/Carla/PythonAPI/examples/parking-spot-detection/Test_segmentation/basemap_txt.asc")

# The point attributes are stored in a dictionary
# print(pcd)
# print(list(pcd.point["spot"]))

# print("Available point fields:")
# for key in pcd.point.attribute_names:
#     print(f"- {key}: shape={pcd.point[key].shape}, dtype={pcd.point[key].dtype}")

OBJECT_IDS = {
    "Ground": 0,
    "ParkingSpace": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example
    "Vehicle": 25
}

# Path to your .asc file
asc_path = "/Users/nehask/Desktop/capstone/parking-spot-detection/annotations/basemap_10_annotated.asc"

import csv
from collections import defaultdict


# Group (x, y) points by `spot`
spot_points = defaultdict(list)

with open(asc_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].startswith("//"):
            continue  # Skip header/comments

        try:
            x, y, z, r, g, b, object_tag, object_id, spot, index = map(int, row)
        except ValueError:
            continue  # Skip malformed lines

        if spot != 0:
            spot_points[spot].append((x, y))

# Print the grouped points for each spot
for spot_id, points in spot_points.items():
    print(f"Spot {spot_id}")

# Generate PARKING_SPOTS list from bounding boxes
PARKING_SPOTS = []

for spot_id, points in spot_points.items():
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    polygon = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax)
    ]

    PARKING_SPOTS.append({
        "id": spot_id,
        "polygon": polygon,
        "occupied": False     # Since points exist for this spot
    })


print("Annotated PARKING_SPOTS:", PARKING_SPOTS)