"""
Path planning algorithms for drone navigation.
"""
import os
import time
import numpy as np
import heapq
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime

import config
from src.utils import world_to_pixel, pixel_to_world, create_downsampled_transform


def plan_drone_path_downsampled(terrain_path, obstacles_path, start_point, end_point,
                                downsample_factor=10, max_runtime_seconds=200,
                                energy_factor_horizontal=1.0, energy_factor_vertical=2.0,
                                direct_path_bias=0.8):
    """
    Plan an optimal drone path with downsampling to handle very large datasets.

    Args:
        terrain_path: Path to terrain raster file (DTM)
        obstacles_path: Path to obstacles raster file
        start_point: (x, y) tuple for start point in world coordinates
        end_point: (x, y) tuple for end point in world coordinates
        downsample_factor: Factor by which to reduce resolution
        max_runtime_seconds: Maximum time to spend searching for a path
        energy_factor_horizontal: Energy cost multiplier for horizontal movement
        energy_factor_vertical: Energy cost multiplier for vertical movement (climbing)
        direct_path_bias: Higher values focus search more along direct line

    Returns:
        (path, energy) tuple with path as list of (x, y, z) points and total energy cost
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
    terrain = full_terrain[::downsample_factor, ::downsample_factor]
    obstacles = full_obstacles[::downsample_factor, ::downsample_factor]

    # Update the transform for the downsampled data
    new_transform = create_downsampled_transform(transform, downsample_factor)

    rows, cols = terrain.shape
    print(f"Downsampled terrain dimensions: {rows} rows x {cols} columns (1/{downsample_factor} of original)")

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

    # A* algorithm with directional bias
    def heuristic(a, b):
        # Euclidean distance with slight direction bias
        direct = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        # Add a small bias in the direction of the goal
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

    # Main loop with timeout
    while open_set:
        # Check if we've exceeded the time limit
        if time.time() - start_time > max_runtime_seconds:
            print(f"Timeout reached after {iterations} iterations")
            return None, np.inf

        iterations += 1
        if iterations % 1000 == 0:
            print(f"Iteration {iterations}, Open set size: {len(open_set)}, "
                  f"Time elapsed: {time.time() - start_time:.1f}s")

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

            if ((neighbor_row, neighbor_col) not in g_score or
                    tentative_g_score < g_score[(neighbor_row, neighbor_col)]):
                # This path is better
                came_from[(neighbor_row, neighbor_col)] = current
                g_score[(neighbor_row, neighbor_col)] = tentative_g_score

                # Add corridor bias based on how close point is to direct line
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
                f_score_value = tentative_g_score + heuristic(
                    (neighbor_row, neighbor_col), (end_row, end_col)
                ) * corridor_bias
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


def get_smaller_test_area(terrain_path, scale=0.05):
    """
    Create a smaller test area from a large terrain dataset.

    Args:
        terrain_path: Path to terrain raster file
        scale: Scale factor for area size (0.05 = 5% of original area)

    Returns:
        Dictionary with area bounds and test points
    """
    with rasterio.open(terrain_path) as src:
        bounds = src.bounds

        # Create a smaller test area
        area_width = bounds.right - bounds.left
        area_height = bounds.top - bounds.bottom

        small_width = area_width * scale
        small_height = area_height * scale

        center_x = bounds.left + area_width * 0.5
        center_y = bounds.bottom + area_height * 0.5

        small_left = center_x - small_width / 2
        small_right = center_x + small_width / 2
        small_bottom = center_y - small_height / 2
        small_top = center_y + small_height / 2

        # Create test points within this smaller area
        inset = 0.2  # 20% inset from edges
        start_point = (small_left + small_width * inset, small_bottom + small_height * inset)
        end_point = (small_right - small_width * inset, small_top - small_height * inset)

        return {
            "bounds": bounds,
            "small_bounds": (small_left, small_bottom, small_right, small_top),
            "start_point": start_point,
            "end_point": end_point
        }