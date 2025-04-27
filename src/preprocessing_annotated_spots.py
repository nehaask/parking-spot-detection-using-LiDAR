import open3d as o3d

# Path to your .asc file
asc_path = "annotations/basemap_10_annotated.asc"

import csv
from collections import defaultdict

spot_points = defaultdict(list)
MARGIN = 0.2  # Margin to add to the bounding box

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
        (xmin+MARGIN, ymin+MARGIN),
        (xmax+MARGIN, ymin+MARGIN),
        (xmax+MARGIN, ymax+MARGIN),
        (xmin+MARGIN, ymax+MARGIN)
    ]

    PARKING_SPOTS.append({
        "id": spot_id, # Unique identifier for the parking spot
        "polygon": polygon, # List of tuples representing the corners of the bounding box
        "occupied": False # Placeholder for occupancy status
    })


print("Annotated PARKING_SPOTS:", PARKING_SPOTS)