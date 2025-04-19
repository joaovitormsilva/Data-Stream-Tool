import sys  # Import sys to exit the script
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.interpolate import interp1d
import time
import random
from itertools import combinations
from scipy.spatial import cKDTree  # Import KD-Tree for fast spatial queries
import csv  # Import CSV module for saving data


# Canvas settings
WIDTH, HEIGHT = 6, 6  # Inches for Matplotlib figure
GRID_SIZE = 0.2  # Defines smaller grid cells in lattice coordinates


# Animation speed hyperparameter (controls speed of the animation)
ANIMATION_SPEED = 0.000001  # Adjust this value to speed up or slow down the animation


# Store centroid trajectories with timestamps
centroid_trajectories = []  # Stores {"points": [...], "start_time": t1, "end_time": t2, "color": ...}
drawing = False
current_trajectory = []
current_color = None  # Store the current color while drawing
start_time = None
end_time = None


# Dictionary to store binary classifications of points
# This dictionary maps each generated point to a binary vector.
# Each vector represents the clusters (substreams) that the point belongs to.
point_classifications = {}  # {point_id, t: binary_vector}


# Create Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
plt.subplots_adjust(bottom=0.3)  # Leave space for buttons and inputs


# Create text boxes for timestamps
ax_start_time = plt.axes([0.2, 0.15, 0.2, 0.05])
start_time_box = TextBox(ax_start_time, "Start Time", initial="")


ax_end_time = plt.axes([0.7, 0.15, 0.2, 0.05])
end_time_box = TextBox(ax_end_time, "End Time", initial="")


# Create timer label
ax_timer = plt.axes([0.4, 0.9, 0.2, 0.05])
ax_timer.set_xticks([])
ax_timer.set_yticks([])
ax_timer.set_frame_on(False)
timer_text = ax_timer.text(0.5, 0.5, "Time: 0", transform=ax_timer.transAxes, ha="center", fontsize=12, fontweight='bold')




def draw_grid():
    """Draws the grid with labels using Matplotlib."""
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for i in np.arange(-1, 1.1, GRID_SIZE):
        ax.axvline(i, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(i, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="blue", linewidth=1.5)  # X-axis
    ax.axvline(0, color="red", linewidth=1.5)   # Y-axis
    
    # Commenting out the trajectory plotting
    #for trajectory in centroid_trajectories:
    #    points = trajectory["points"]
    #    color = trajectory["color"]
    #    if len(points) > 1:
    #        x_values, y_values = zip(*points)
    #        ax.plot(x_values, y_values, linestyle='-', color=color, linewidth=1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.draw()


# Function to stop execution
def stop_execution(event):
    print("Execution stopped.")
    plt.close(fig)  # Closes the Matplotlib figure and exits
    sys.exit(0)  # Fully terminate the script


# Handle mouse events
def on_mouse_down(event):
    global drawing, current_trajectory, current_color, start_time, end_time
    if event.xdata is None or event.ydata is None:
        return
    try:
        start_time = int(start_time_box.text)
        end_time = int(end_time_box.text)
        if start_time >= end_time:
            print("Start time must be less than end time.")
            return
    except ValueError:
        print("Please enter valid timestamps before drawing.")
        return
    drawing = True
    current_trajectory = [(event.xdata, event.ydata)]
    current_color = [random.random(), random.random(), random.random()]  # Assign a random color


def on_mouse_move(event):
    global drawing, current_trajectory, current_color
    if drawing and event.xdata is not None and event.ydata is not None:
        current_trajectory.append((event.xdata, event.ydata))
        if len(current_trajectory) > 1:
            x1, y1 = current_trajectory[-2]
            x2, y2 = current_trajectory[-1]
            ax.plot([x1, x2], [y1, y2], linestyle='-', color=current_color, linewidth=1)  # Use assigned color
        plt.draw()


def on_mouse_up(event):
    global drawing
    if drawing:
        drawing = False
        centroid_trajectories.append({
            "points": current_trajectory.copy(),
            "start_time": start_time,
            "end_time": end_time,
            "color": current_color
        })
        start_time_box.set_val("")
        end_time_box.set_val("")


def save_stream_to_csv(filename="stream_data.csv"):
    """Saves the interleaved stream data to a CSV file."""
    if not point_classifications:
        print("No data to save.")
        return


    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["global_timestamp", "timestamp", "x", "y"] + [f"cluster_{i+1}" for i in range(len(centroid_trajectories))]
        writer.writerow(header)


        # Group points by timestamp
        grouped_points = {}
        for (global_timestamp, timestamp), (x, y, binary_vector) in point_classifications.items():
            if timestamp not in grouped_points:
                grouped_points[timestamp] = []
            grouped_points[timestamp].append((global_timestamp, timestamp, x, y, *binary_vector))


        # Sort timestamps and shuffle points within each timestamp to interleave clusters
        for timestamp in sorted(grouped_points.keys()):
            random.shuffle(grouped_points[timestamp])  # Shuffle to interleave clusters
            for row in grouped_points[timestamp]:
                writer.writerow(row)


    print(f"Stream data saved to {filename}")






def generate_stream(event):


    if len(centroid_trajectories) == 0:
        print("No trajectories recorded. Click and drag on the plot first.")
        return


    global_start_time = min(t["start_time"] for t in centroid_trajectories)
    global_end_time = max(t["end_time"] for t in centroid_trajectories)
    print(f"Generating animated stream from t={global_start_time} to t={global_end_time}...")


    # Create a global index mapping for clusters
    global_cluster_indices = {id(t): idx for idx, t in enumerate(centroid_trajectories)}


    global_timestamp = 1 # Global timestamp


    # Display information about each substream
    for i, trajectory in enumerate(centroid_trajectories):
        num_captured_points = len(trajectory["points"])
        num_centroids = trajectory["end_time"] - trajectory["start_time"] + 1
        print(f"Substream {i+1}:")
        print(f"  - Start Timestamp: {trajectory['start_time']}")
        print(f"  - End Timestamp: {trajectory['end_time']}")
        print(f"  - Captured points: {num_captured_points}")
        print(f"  - Interpolated centroids: {num_centroids}\n")


    for t in range(global_start_time, global_end_time + 1):
        draw_grid()
        timer_text.set_text(f"Time: {t}")  # Update timer text
        fig.canvas.draw_idle()
        active_centroids = []


        for trajectory in centroid_trajectories:
            start_time = trajectory["start_time"]
            end_time = trajectory["end_time"]
            points = trajectory["points"]
            color = trajectory["color"]


            if start_time <= t <= end_time:
                x_values, y_values = zip(*points)
                if len(x_values) < 2:
                    continue
                f_x = interp1d(np.linspace(start_time, end_time, len(x_values)), x_values, kind="linear")
                f_y = interp1d(np.linspace(start_time, end_time, len(y_values)), y_values, kind="linear")
                cx, cy = f_x(t), f_y(t)
                gaussian_points = np.random.normal(loc=[cx, cy], scale=0.05, size=(100, 2))
                radius = np.std(gaussian_points[:, 0]) # Compute standard deviation (σ)
                adjusted_radius = 2 * radius  # Use 2σ as the new radius
                active_centroids.append((cx, cy, adjusted_radius, color))
                ax.scatter(gaussian_points[:, 0], gaussian_points[:, 1], color=color, s=5)


        # Find all intersection points across multiple clusters
        # Convert active centroids into NumPy arrays for fast computation
        centroid_positions = np.array([[cx, cy] for cx, cy, _, _ in active_centroids])
        centroid_radii = np.array([r for _, _, r, _ in active_centroids])


        # Store all Gaussian-distributed points
        all_gaussian_points = np.concatenate([
            np.random.normal(loc=[cx, cy], scale=0.05, size=(100, 2))
            for cx, cy, _, _ in active_centroids
        ])


        # Use KD-Tree to efficiently check which clusters each point belongs to
        tree = cKDTree(centroid_positions)  # Build a KD-Tree with centroids
        point_cluster_counts = np.zeros(len(all_gaussian_points), dtype=int)  # Store count of clusters per point


        # Create a point_labels matrix with global cluster indexing
        # Initialize as zeros, meaning no membership by default.
        point_labels = np.zeros((len(all_gaussian_points), len(centroid_trajectories)), dtype=int)


        for i, (px, py) in enumerate(all_gaussian_points):
            # Find clusters within radius using a range search
            # Used query_ball_point to find which clusters a point belongs to
            nearby_clusters = tree.query_ball_point([px, py], r=np.max(centroid_radii))
            
            # If the KD-Tree radius is too small, some points might not get assigned to any cluster
            if not nearby_clusters:  # If no clusters found, force assignment to the nearest one
                nearest_cluster = tree.query([px, py])[1]  # Find the single closest cluster
                nearby_clusters = [nearest_cluster]


            point_cluster_counts[i] = len(nearby_clusters)  # Store the count
            for local_idx in nearby_clusters:
                # Updated point_labels for each point by setting 1 in positions corresponding to clusters the point belongs to
                global_idx = global_cluster_indices[id(centroid_trajectories[local_idx])]
                point_labels[i, global_idx] = 1  # Assign 1 at correct cluster index


        # Converts the coordinates of each point into a dictionary key
        # Stores its corresponding binary vector, indicating its cluster memberships
        for i, point in enumerate(all_gaussian_points):
            x, y = point  # Extract x, y directly
            #point_classifications[(global_timestamp, t)] = point_labels[i].tolist()
            point_classifications[(global_timestamp, t)] = (x, y, point_labels[i].tolist())  # Store (x, y)
            global_timestamp += 1  # Ensure uniqueness
            #point_classifications[tuple(point)] = point_labels[i].tolist()


        # Highlight overlapping points efficiently
        #ax.scatter(all_gaussian_points[:, 0], all_gaussian_points[:, 1], color='gray', s=5)  # Default color


        # Apply colors based on number of cluster memberships
        ax.scatter(all_gaussian_points[point_cluster_counts == 2, 0],
                   all_gaussian_points[point_cluster_counts == 2, 1], color='black', s=10)  # 2 clusters → Black


        ax.scatter(all_gaussian_points[point_cluster_counts >= 3, 0],
                   all_gaussian_points[point_cluster_counts >= 3, 1], color='red', s=12)  # 3+ clusters → Red




        #plt.pause(ANIMATION_SPEED)  # Use animation speed variable
        #time.sleep(ANIMATION_SPEED)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    plt.draw()


    save_stream_to_csv()  # Save stream after generation


# Button to generate the stream
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, "Generate Stream")
button.on_clicked(generate_stream)




# Create "Stop" button
ax_stop_button = plt.axes([0.7, 0.05, 0.2, 0.075])  # Position of the button
stop_button = Button(ax_stop_button, "Stop Execution")


stop_button.on_clicked(stop_execution)  # Bind function to button


draw_grid()
fig.canvas.mpl_connect('button_press_event', on_mouse_down)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_up)
plt.show()
