# USAGE: python script.py data.csv 10:20 --interval 30

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

def load_data(filename):
    """Loads the CSV data and extracts relevant columns dynamically."""
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Identify column indices
        timestamp_idx = header.index("timestamp")
        x_idx = header.index("x")
        y_idx = header.index("y")
        cluster_indices = [i for i, col in enumerate(header) if col.startswith("cluster_")]

        data = []
        for row in reader:
            timestamp = int(row[timestamp_idx])
            x, y = float(row[x_idx]), float(row[y_idx])
            cluster_membership = np.array([int(row[i]) for i in cluster_indices])
            data.append((timestamp, x, y, cluster_membership))

    return data, len(cluster_indices)

def plot_animation(data, timestamps, num_clusters, interval):
    """Plots an animated scatter plot for a given range of timestamps."""

    cmap = matplotlib.colormaps.get_cmap("tab10")  # Updated for Matplotlib 3.7+
    cluster_colors = [cmap(i % 10) for i in range(num_clusters)]  # Dynamically assign colors

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.2, 0.2))
    ax.set_yticks(np.arange(-1, 1.2, 0.2))
    scatter = ax.scatter([], [], s=10)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

    def update(frame):
        """Updates scatter plot for each frame in the animation."""
        t = timestamps[frame]
        points = [(x, y, clusters) for ts, x, y, clusters in data if ts == t]

        if not points:
            scatter.set_offsets([])
            scatter.set_color([])
            time_text.set_text(f"Time: {t}")
            return scatter,

        x_vals, y_vals, cluster_membership = zip(*points)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        cluster_membership = np.array(cluster_membership)

        # Determine color based on cluster membership
        num_clusters_per_point = cluster_membership.sum(axis=1)
        colors = []
        for i, clusters in enumerate(cluster_membership):
            if num_clusters_per_point[i] == 1:
                cluster_index = np.argmax(clusters)
                colors.append(cluster_colors[cluster_index])  # Use assigned color
            elif num_clusters_per_point[i] == 2:
                colors.append("black")  # Two clusters → Black
            elif num_clusters_per_point[i] >= 3:
                colors.append("red")  # Three or more clusters → Red
            else:
                colors.append("gray")  # Default fallback

        scatter.set_offsets(np.column_stack((x_vals, y_vals)))
        scatter.set_color(colors)
        time_text.set_text(f"Time: {t}")

        return scatter,

    ani = FuncAnimation(fig, update, frames=len(timestamps), interval=interval, repeat=False)
    plt.show()

def main():
    """Command-line entry point for running the animation."""
    parser = argparse.ArgumentParser(description="Plot animated points from CSV data.")
    parser.add_argument("file", type=str, help="Path to the CSV file.")
    parser.add_argument("timestamp", type=str, help="Timestamp or range (e.g., '10' or '10:20').")
    parser.add_argument("--interval", type=float, default=50, help="Animation interval in milliseconds.")

    args = parser.parse_args()

    data, num_clusters = load_data(args.file)

    # Parse timestamp input
    if ":" in args.timestamp:
        start, end = map(int, args.timestamp.split(":"))
        timestamps = list(range(start, end + 1))
    else:
        timestamps = [int(args.timestamp)]

    plot_animation(data, timestamps, num_clusters, args.interval)

if __name__ == "__main__":
    main()

