## ALL FILES 

- [scan_semantic_carla_parking_lot.py](scan_semantic_carla_parking_lot.py) 

    Code to run autopilot run a Semantic LiDAR around a block that has the parking lot. Autopilot runs for 55 seconds and generates the main basemap.pcd

- [preprocessing_annotated_spots.py](preprocessing_annotated_spots.py)

    Code to generate the parking space annotations from the pointwise PCD generated from the manually annotated basemap. Outputs a dictionary PARKING_SPOTS

    PARKING_SPOTS = {
        "id": Unique identifier for the parking spot
        "polygon": List of tuples representing the corners of the bounding box
        "occupied": Boolean Placeholder for occupancy status
        }

- [preprocessing_data_collector.py](preprocessing_data_collector.py)

    Drives in the CARLA parking lot for 100 frames at a speed 0.3
    Transforms from vehicle coordinates to world coordinates; stores pointwise (found in outputs/pcds/)
    Accumulates these points and concatenates every 20 frames; stores combined basemap (found in outputs/basemap_{last_frame})

- [extracting_annonated.py](extracting_annonated.py)

    Extracting the 3D point cloud data of just the annotated spaces; Outputs a [basemap_spaces.pcd](outputs/filtered_basemap.pcd) 
    Ground Point Removal performed here to include only pcd above the ground

- [clustering.py](clustering.py)

    DBSCAN Clustering on the annotated spaces - Displays visualisation of the parking lot



## order to run files 

1) data collector - collects pointwise pcd from carla; reading pointwise point clouds and accumulating as single pcds

2) extracting_annotated.py - gets pcd of just the annotated spacesfrom the full basemap; filters ground points

3) clustering - returns clusters given the accumulated files

carla_code - overlay on carla






