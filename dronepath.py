import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import heapq
import time
from datetime import datetime

# Set paths to your processed data
base_dir = "/Users/shresthshrivastava/Downloads/LIDAR Side Project"  # Keep your original path
output_dir = os.path.join(base_dir, "processed")
terrain_path = os.path.join(output_dir, "merged_dtm.tif")
obstacles_path = os.path.join(output_dir, "obstacles.tif")


def plan_drone_path_downsampled(terrain_path, obstacles_path, start_point, end_point, downsample_factor=10,
                                max_runtime_seconds=200):
    """
    Plan an optimal drone path with downsampling to handle very large datasets.
    """
    start_time = time.time()

    # Load terrain and obstacles
    with rasterio.open(terrain_path) as src:
        full_terrain = src.read(1)
        transform = src.transform
        crs = src.crs

    with rasterio.open(obstacles_path) as src:
        full_obstacles = src.read(1)

    # Get original dimensions
    orig_rows, orig_cols = full_terrain.shape
    print(f"Original terrain dimensions: {orig_rows} rows x {orig_cols} columns")

    # Create a downsampled version - take every Nth pixel
    # This dramatically reduces the search space
    terrain = full_terrain[::downsample_factor, ::downsample_factor]
    obstacles = full_obstacles[::downsample_factor, ::downsample_factor]

    # Update the transform for the downsampled data
    new_transform = rasterio.transform.from_origin(
        transform.c, transform.f,
        transform.a * downsample_factor,
        abs(transform.e) * downsample_factor
    )

    rows, cols = terrain.shape
    print(f"Downsampled terrain dimensions: {rows} rows x {cols} columns (1/{downsample_factor} of original)")

    # Convert world coordinates to downsampled pixel coordinates
    def world_to_pixel(x, y, transform):
        col, row = ~transform * (x, y)
        return int(row), int(col)

    # Convert downsampled pixel coordinates to world coordinates
    def pixel_to_world(row, col, transform):
        x, y = transform * (col, row)
        return x, y

    # Convert start and end points to pixel coordinates in the downsampled grid
    start_row, start_col = world_to_pixel(start_point[0], start_point[1], new_transform)
    end_row, end_col = world_to_pixel(end_point[0], end_point[1], new_transform)

    # Make sure coordinates are within bounds
    start_row = max(0, min(start_row, rows - 1))
    start_col = max(0, min(start_col, cols - 1))
    end_row = max(0, min(end_row, rows - 1))
    end_col = max(0, min(end_col, cols - 1))

    print(f"Start pixel: ({start_row}, {start_col}), End pixel: ({end_row}, {end_col})")
    print(f"Distance in grid cells: {abs(end_row - start_row) + abs(end_col - start_col)}")

    # Parameters for energy calculation
    energy_factor_horizontal = 1.0
    energy_factor_vertical = 2.0

    # Modified A* algorithm with directional bias
    def heuristic(a, b):
        # Euclidean distance with slight direction bias
        direct = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        # Add a small bias in the direction of the goal
        # This helps guide the search more efficiently
        return direct * 1.1

    # Define neighbors (8-connected grid)
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Initialize data structures
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {(start_row, start_col): 0}
    f_score = {(start_row, start_col): heuristic((start_row, start_col), (end_row, end_col))}

    # Push start node to open set
    heapq.heappush(open_set, (f_score[(start_row, start_col)], (start_row, start_col)))

    # Counter for progress reporting
    iterations = 0

    # Focus on a narrower search corridor by prioritizing cells closer to the direct path
    direct_path_bias = 0.8  # Lower values make the search focus more on the direct path

    # Main loop with timeout
    while open_set:
        # Check if we've exceeded the time limit
        if time.time() - start_time > max_runtime_seconds:
            print(f"Timeout reached after {iterations} iterations")
            return None, np.inf

        iterations += 1
        if iterations % 1000 == 0:
            print(
                f"Iteration {iterations}, Open set size: {len(open_set)}, Time elapsed: {time.time() - start_time:.1f}s")

        _, current = heapq.heappop(open_set)

        if current == (end_row, end_col):
            print(f"Path found after {iterations} iterations")

            # Reconstruct path in the downsampled grid
            downsampled_path = []
            curr = current
            while curr in came_from:
                row, col = curr
                x, y = pixel_to_world(row, col, new_transform)
                # Get elevation - check bounds just in case
                if 0 <= row < rows and 0 <= col < cols:
                    z = terrain[row, col]
                else:
                    z = 0
                downsampled_path.append((x, y, z))
                curr = came_from[curr]

            # Add start point
            row, col = start_row, start_col
            x, y = pixel_to_world(row, col, new_transform)
            z = terrain[row, col]
            downsampled_path.append((x, y, z))

            # Reverse path to go from start to end
            downsampled_path.reverse()

            print(f"Path length: {len(downsampled_path)} points")
            return downsampled_path, g_score[(end_row, end_col)]

        closed_set.add(current)

        # Check all neighbors
        for dr, dc in neighbors:
            neighbor_row, neighbor_col = current[0] + dr, current[1] + dc

            # Check bounds
            if (neighbor_row < 0 or neighbor_row >= rows or
                    neighbor_col < 0 or neighbor_col >= cols):
                continue

            # Skip if in closed set (already processed)
            if (neighbor_row, neighbor_col) in closed_set:
                continue

            # Skip if obstacle
            if obstacles[neighbor_row, neighbor_col] > 0:
                continue

            # Calculate energy for this move
            current_height = terrain[current[0], current[1]]
            neighbor_height = terrain[neighbor_row, neighbor_col]
            height_diff = neighbor_height - current_height

            # Diagonal moves are longer
            if abs(dr) == 1 and abs(dc) == 1:
                distance = np.sqrt(2)
            else:
                distance = 1

            # Energy calculation based on distance and height change
            move_energy = (
                    distance * energy_factor_horizontal +
                    max(0, height_diff) * energy_factor_vertical
            )

            # Total energy to reach neighbor
            tentative_g_score = g_score[current] + move_energy

            if (neighbor_row, neighbor_col) not in g_score or tentative_g_score < g_score[(neighbor_row, neighbor_col)]:
                # This path is better
                came_from[(neighbor_row, neighbor_col)] = current
                g_score[(neighbor_row, neighbor_col)] = tentative_g_score

                # Update f_score with the corridor bias
                # Calculate how close this point is to the direct line from start to end
                # This helps focus the search along the most likely path
                dx = end_col - start_col
                dy = end_row - start_row
                if dx == 0 and dy == 0:
                    # Start and end are the same point
                    direct_line_dist = 0
                else:
                    # Distance from point to line
                    num = abs((neighbor_col - start_col) * dy - (neighbor_row - start_row) * dx)
                    den = np.sqrt(dx * dx + dy * dy)
                    direct_line_dist = num / den

                # Add bias based on how close point is to direct line
                corridor_bias = 1.0 + direct_line_dist * direct_path_bias
                f_score_value = tentative_g_score + heuristic((neighbor_row, neighbor_col),
                                                              (end_row, end_col)) * corridor_bias
                f_score[(neighbor_row, neighbor_col)] = f_score_value

                # Add to open set if not already there
                for i, (f, node) in enumerate(open_set):
                    if node == (neighbor_row, neighbor_col):
                        # If it is, we don't need to add it again, just update
                        open_set[i] = (f_score_value, node)
                        heapq.heapify(open_set)
                        break
                else:
                    # It's not in the open set, add it
                    heapq.heappush(open_set, (f_score_value, (neighbor_row, neighbor_col)))

    print(f"No path found after {iterations} iterations")
    return None, np.inf


def test_with_smaller_area_and_downsampling():
    """Test path planning with a much smaller area and downsampling."""
    with rasterio.open(terrain_path) as src:
        bounds = src.bounds
        print(f"Full dataset bounds: {bounds}")

        # Create a smaller test area - using just 5% of the full area
        area_width = bounds.right - bounds.left
        area_height = bounds.top - bounds.bottom

        # Use a much smaller area near the center of the dataset
        # to increase chances of finding a path quickly
        small_width = area_width * 0.05  # 5% of original width
        small_height = area_height * 0.05  # 5% of original height

        # Center it in the middle of the dataset
        center_x = bounds.left + area_width * 0.5
        center_y = bounds.bottom + area_height * 0.5

        small_left = center_x - small_width / 2
        small_right = center_x + small_width / 2
        small_bottom = center_y - small_height / 2
        small_top = center_y + small_height / 2

        print(f"Testing on a smaller area: ({small_left}, {small_bottom}) to ({small_right}, {small_top})")

        # Create test points within this smaller area
        # Move the points slightly inward from the edges for better results
        inset = 0.2  # 20% inset from edges
        start_point = (small_left + small_width * inset, small_bottom + small_height * inset)
        end_point = (small_right - small_width * inset, small_top - small_height * inset)

        print(f"Planning path from {start_point} to {end_point}")

        # Run the path planning with downsampling and a 3-minute timeout
        downsample_factor = 10  # Use every 10th point in each dimension
        path, energy = plan_drone_path_downsampled(
            terrain_path,
            obstacles_path,
            start_point,
            end_point,
            downsample_factor=downsample_factor,
            max_runtime_seconds=180
        )

        if path:
            print(f"Path found with {len(path)} points and energy cost: {energy:.2f}")

            # Load terrain and obstacles for visualization
            with rasterio.open(terrain_path) as terrain_src:
                terrain = terrain_src.read(1)
                transform = terrain_src.transform

            with rasterio.open(obstacles_path) as obs_src:
                obstacles = obs_src.read(1)

            # Convert world to pixel coordinates for plotting
            def world_to_pixel(x, y, transform):
                col, row = ~transform * (x, y)
                return int(row), int(col)

            # Get the pixel boundaries of our smaller test area
            area_top_left = world_to_pixel(small_left, small_top, transform)
            area_bottom_right = world_to_pixel(small_right, small_bottom, transform)

            min_row = max(0, min(area_top_left[0], area_bottom_right[0]))
            max_row = min(terrain.shape[0], max(area_top_left[0], area_bottom_right[0]))
            min_col = max(0, min(area_top_left[1], area_bottom_right[1]))
            max_col = min(terrain.shape[1], max(area_top_left[1], area_bottom_right[1]))

            # Extract the window for viewing
            terrain_window = terrain[min_row:max_row, min_col:max_col]
            obstacles_window = obstacles[min_row:max_row, min_col:max_col]

            # Convert path to pixel coordinates
            pixel_path = []
            for x, y, z in path:
                row, col = world_to_pixel(x, y, transform)
                # Adjust coordinates to our window
                row -= min_row
                col -= min_col
                pixel_path.append((row, col))

            # Create visualization
            plt.figure(figsize=(15, 10))

            # Create a custom colormap for terrain with transparent background
            terrain_cmap = plt.cm.terrain.copy()
            terrain_cmap.set_bad('white', alpha=0)  # Make NaN values transparent

            # Plot terrain
            plt.imshow(terrain_window, cmap=terrain_cmap, alpha=0.7)

            # Plot obstacles
            obstacles_vis = np.ma.masked_where(obstacles_window == 0, obstacles_window)
            plt.imshow(obstacles_vis, cmap='binary', alpha=0.5, vmin=0, vmax=1)

            # Plot path if we have valid points in our window
            if pixel_path:
                # Filter points to ensure they're in bounds of our window
                valid_points = [(r, c) for r, c in pixel_path
                                if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

                if valid_points:
                    path_rows, path_cols = zip(*valid_points)
                    plt.plot(path_cols, path_rows, 'r-', linewidth=3, label='Drone Path')

                    # Mark start and end
                    if len(valid_points) > 0:
                        plt.plot(valid_points[0][1], valid_points[0][0], 'go', markersize=10, label='Start')
                    if len(valid_points) > 1:
                        plt.plot(valid_points[-1][1], valid_points[-1][0], 'ro', markersize=10, label='End')

            plt.title('Drone Flight Path with Terrain and Obstacles')
            plt.legend()

            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(output_dir, f'drone_path_{timestamp}.png'))
            plt.show()

            # Save path to file
            csv_path = os.path.join(output_dir, f'drone_path_{timestamp}.csv')

            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y', 'Z', 'Energy'])

                # Calculate cumulative energy
                cumulative_energy = 0
                energy_factor_horizontal = 1.0
                energy_factor_vertical = 2.0

                for i, (x, y, z) in enumerate(path):
                    if i > 0:
                        prev_x, prev_y, prev_z = path[i - 1]

                        # Calculate segment energy
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        height_diff = max(0, z - prev_z)
                        segment_energy = distance * energy_factor_horizontal + height_diff * energy_factor_vertical
                        cumulative_energy += segment_energy

                    writer.writerow([x, y, z, cumulative_energy])

            print(f"Path saved to {csv_path}")
        else:
            print("No path found within the time limit!")


# Run the test function
if __name__ == "__main__":
    try:
        print("Running path planning with downsampling and smaller area...")
        test_with_smaller_area_and_downsampling()
    except Exception as e:
        import traceback

        print(f"Error in path planning: {e}")
        traceback.print_exc()