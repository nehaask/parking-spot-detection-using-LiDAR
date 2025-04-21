# Let's generate a Python function that would allow the user to create a heatmap-style debug visualization
# for the obstacle points and detected clusters. This visualization can help in tuning DBSCAN and debugging.

# We'll create a simple matplotlib plot to show:
# - All obstacle points (e.g., small gray dots)
# - Clustered points (different colors)
# - Cluster centroids (large red dots)
# - Parking spots (as boxes)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import DBSCAN

def generate_debug_heatmap(obstacle_points, parking_spots, eps=2.0, min_samples=100):
    X = np.array(obstacle_points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points as gray
    ax.scatter(X[:, 0], X[:, 1], s=5, c='gray', label='Obstacle Points', alpha=0.4)

    # Plot each cluster in a different color
    labels = set(clustering.labels_)
    colors = plt.cm.get_cmap("tab10", len(labels))

    for label in labels:
        if label == -1:
            continue  # noise
        cluster_points = X[clustering.labels_ == label]
        centroid = np.mean(cluster_points, axis=0)
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {label}')
        ax.scatter(*centroid, color='red', s=100, edgecolor='black', label='Centroid' if label == 0 else "")

    # Plot parking spots
    for spot in parking_spots:
        poly = np.array(spot["polygon"])
        rect = patches.Polygon(poly, closed=True,
                               linewidth=2,
                               edgecolor='green' if not spot['occupied'] else 'red',
                               facecolor='none')
        ax.add_patch(rect)
        cx, cy = np.mean(poly, axis=0)
        ax.text(cx, cy, f"ID {spot['id']}", ha='center', va='center', fontsize=8, color='black')

    ax.set_title("Obstacle Points, Clusters, and Parking Spots")
    ax.set_xlabel("X (world coords)")
    ax.set_ylabel("Y (world coords)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Dummy call example (would need to be replaced with real data during runtime)
generate_debug_heatmap(obstacle_points, PARKING_SPOTS)

