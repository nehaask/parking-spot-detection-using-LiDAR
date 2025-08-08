import open3d as o3d
import numpy as np
import os
import time
from extracting_annonated import extract_blocks_from_pcd, spots
from test_files.clustering import run_dbscan_and_visualize

def main():
    input_dir = "outputs_test/pcds"
    output_dir = "accumulated_outputs"
    filtered_dir = "filtered_outputs"
    clustered_dir = "clustered_outputs"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(clustered_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pcd')])

    batch_size = 10
    points_batch = []
    object_id_batch = []
    object_tag_batch = []
    latitude_batch = []
    longitude_batch = []
    altitude_batch = []

    overall_start_time = time.time()

    for idx, frame_file in enumerate(frame_files):
        filepath = os.path.join(input_dir, frame_file)
        pcd = o3d.t.io.read_point_cloud(filepath)

        points = pcd.point["positions"].numpy()
        object_id = pcd.point["object_id"].numpy()
        object_tag = pcd.point["object_tag"].numpy()
        latitude = pcd.point["latitude"].numpy()
        longitude = pcd.point["longitude"].numpy()
        altitude = pcd.point["altitude"].numpy()

        points_batch.append(points)
        object_id_batch.append(object_id)
        object_tag_batch.append(object_tag)
        latitude_batch.append(latitude)
        longitude_batch.append(longitude)
        altitude_batch.append(altitude)

        if (idx + 1) % batch_size == 0:
            batch_start_time = time.time()

            batch_idx = (idx + 1)
            print(f"\n🔵 Saving accumulated basemap for frames {batch_idx - batch_size + 1} to {batch_idx}")

            combined_points = np.vstack(points_batch)
            combined_ids = np.vstack(object_id_batch)
            combined_tags = np.vstack(object_tag_batch)
            combined_lat = np.vstack(latitude_batch)
            combined_lon = np.vstack(longitude_batch)
            combined_alt = np.vstack(altitude_batch)

            # Save the accumulated batch
            pcd_all = o3d.t.geometry.PointCloud()
            pcd_all.point["positions"] = o3d.core.Tensor(combined_points, o3d.core.float32)
            pcd_all.point["object_id"] = o3d.core.Tensor(combined_ids, o3d.core.uint32)
            pcd_all.point["object_tag"] = o3d.core.Tensor(combined_tags, o3d.core.uint32)
            pcd_all.point["latitude"] = o3d.core.Tensor(combined_lat, o3d.core.float64)
            pcd_all.point["longitude"] = o3d.core.Tensor(combined_lon, o3d.core.float64)
            pcd_all.point["altitude"] = o3d.core.Tensor(combined_alt, o3d.core.float64)

            filename = f"{output_dir}/basemap_{batch_idx:04d}.pcd"
            o3d.t.io.write_point_cloud(filename, pcd_all, write_ascii=False)
            print(f"✅ Basemap saved: {filename}")

            # --- Step 1: Extract only annotated parking spaces ---
            filtered_pcd_filename = f"{filtered_dir}/filtered_basemap_{batch_idx:04d}.pcd"
            extract_blocks_from_pcd(
                filename,
                spots,
                filtered_pcd_filename,
                z_min=-2,
                z_max=2
            )

            # --- Step 2: Run DBSCAN clustering on extracted data ---
            run_dbscan_and_visualize(
                filtered_pcd_filename,
                save_clustered=True,
                output_dir=clustered_dir
            )

            batch_end_time = time.time()
            batch_elapsed_time = batch_end_time - batch_start_time
            print(f"🕒 Batch {batch_idx//batch_size} processed in {batch_elapsed_time:.2f} seconds.")

            # Reset for next batch
            points_batch.clear()
            object_id_batch.clear()
            object_tag_batch.clear()
            latitude_batch.clear()
            longitude_batch.clear()
            altitude_batch.clear()


    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f"\n🏁 All batches completed in {overall_elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    main()
