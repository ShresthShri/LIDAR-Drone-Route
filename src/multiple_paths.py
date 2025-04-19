"""
Generate and visualize multiple drone flight paths.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import time
from datetime import datetime
import random

from src.path_planning import plan_drone_path_downsampled
from src.path_smoothing import smooth_drone_path
from src.utils import world_to_pixel, pixel_to_world, create_downsampled_transform, calculate_path_energy
import config


def generate_multiple_paths(terrain_path, obstacles_path, start_point, end_point, num_paths=5,
                            downsample_factor=10, max_runtime_seconds=60, smooth=True):
    """
    Generate multiple possible drone paths between two points with increased diversity.

    Args:
        terrain_path: Path to terrain raster file
        obstacles_path: Path to obstacles raster file
        start_point: (x, y) tuple for start point
        end_point: (x, y) tuple for end point
        num_paths: Number of paths to generate
        downsample_factor: Factor to downsample terrain
        max_runtime_seconds: Maximum runtime for each path
        smooth: Whether to apply path smoothing

    Returns:
        List of (path, energy) tuples
    """
    import numpy as np
    import random
    import rasterio

    paths = []

    # Generate the baseline path first
    print(f"Generating baseline path from {start_point} to {end_point}...")
    baseline_path, baseline_energy = plan_drone_path_downsampled(
        terrain_path,
        obstacles_path,
        start_point,
        end_point,
        downsample_factor=downsample_factor,
        max_runtime_seconds=max_runtime_seconds,
        energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
        energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL,
        direct_path_bias=config.DIRECT_PATH_BIAS
    )

    if baseline_path:
        paths.append((baseline_path, baseline_energy, "Baseline"))
    else:
        print("Failed to generate baseline path. Try increasing max_runtime.")
        return []

    # Load terrain and obstacles for smoothing and for introducing waypoints
    with rasterio.open(terrain_path) as terrain_src:
        terrain = terrain_src.read(1)
        if downsample_factor > 1:
            terrain = terrain[::downsample_factor, ::downsample_factor]
        transform = terrain_src.transform
        if downsample_factor > 1:
            transform = create_downsampled_transform(transform, downsample_factor)

    with rasterio.open(obstacles_path) as obs_src:
        obstacles = obs_src.read(1)
        if downsample_factor > 1:
            obstacles = obstacles[::downsample_factor, ::downsample_factor]

    # Apply smoothing to baseline path if requested
    if smooth and baseline_path:
        print("Smoothing baseline path...")
        smoothed_path = smooth_drone_path(
            baseline_path,
            terrain,
            obstacles,
            transform,
            max_slope_deg=config.MAX_SLOPE_DEG
        )

        smoothed_energy = calculate_path_energy(
            smoothed_path,
            energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
            energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
        )

        paths.append((smoothed_path, smoothed_energy, "Baseline (Smoothed)"))

    # Generate waypoints around the direct path to force diversity
    direct_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    path_length = np.sqrt(direct_vector[0] ** 2 + direct_vector[1] ** 2)

    # List to store waypoints for each alternative path
    path_waypoints = []

    # Create waypoints at different offsets from the direct path
    for i in range(1, num_paths):
        # Calculate perpendicular direction to the direct path
        perp_vector = (-direct_vector[1] / path_length, direct_vector[0] / path_length)

        # Alternate between positive and negative offsets
        if i % 2 == 0:
            # Create waypoint with positive perpendicular offset
            offset_distance = (i // 2) * path_length * 0.15  # Increasing offsets
        else:
            # Create waypoint with negative perpendicular offset
            offset_distance = -(i // 2 + 1) * path_length * 0.15  # Increasing offsets

        # Position waypoint along the path (from 30% to 70% of the way)
        path_position = 0.3 + (i % 3) * 0.2  # Varies between 0.3, 0.5, and 0.7

        waypoint_x = start_point[0] + direct_vector[0] * path_position + perp_vector[0] * offset_distance
        waypoint_y = start_point[1] + direct_vector[1] * path_position + perp_vector[1] * offset_distance

        # Store this waypoint
        path_waypoints.append((waypoint_x, waypoint_y))

    # Generate alternative paths with different parameters and waypoints
    for i in range(1, num_paths):
        print(f"Generating alternative path {i}...")

        # Get the waypoint for this path
        if i - 1 < len(path_waypoints):
            waypoint = path_waypoints[i - 1]

            # First try to find a path to the waypoint
            print(f"  Finding path to waypoint: {waypoint}")
            waypoint_path, waypoint_energy = plan_drone_path_downsampled(
                terrain_path,
                obstacles_path,
                start_point,
                waypoint,
                downsample_factor=downsample_factor,
                max_runtime_seconds=max_runtime_seconds // 2,  # Half time for first segment
                energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
                energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL * (0.8 + random.random() * 0.4),
                # Vary vertical cost
                direct_path_bias=config.DIRECT_PATH_BIAS * (0.7 + random.random() * 0.6)  # Vary bias
            )

            # Then from waypoint to destination
            if waypoint_path:
                print(f"  Finding path from waypoint to destination")
                dest_path, dest_energy = plan_drone_path_downsampled(
                    terrain_path,
                    obstacles_path,
                    waypoint,
                    end_point,
                    downsample_factor=downsample_factor,
                    max_runtime_seconds=max_runtime_seconds // 2,  # Half time for second segment
                    energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
                    energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL * (0.8 + random.random() * 0.4),
                    # Vary vertical cost
                    direct_path_bias=config.DIRECT_PATH_BIAS * (0.7 + random.random() * 0.6)  # Vary bias
                )

                # Combine the two path segments
                if dest_path:
                    # Remove duplicate waypoint at the join
                    combined_path = waypoint_path + dest_path[1:]
                    combined_energy = waypoint_energy + dest_energy

                    if smooth:
                        print(f"  Smoothing combined path")
                        smooth_combined_path = smooth_drone_path(
                            combined_path,
                            terrain,
                            obstacles,
                            transform,
                            max_slope_deg=config.MAX_SLOPE_DEG
                        )

                        smooth_combined_energy = calculate_path_energy(
                            smooth_combined_path,
                            energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
                            energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
                        )

                        paths.append((smooth_combined_path, smooth_combined_energy, f"Alternative {i}"))
                    else:
                        paths.append((combined_path, combined_energy, f"Alternative {i}"))

                    continue  # Skip the direct path attempt if we succeeded with the waypoint

        # Fallback to direct path with varied parameters if waypoint approach failed
        variation_factor = 0.5 + random.random() * 1.5  # 0.5 to 2.0
        energy_horizontal = config.ENERGY_FACTOR_HORIZONTAL * variation_factor
        energy_vertical = config.ENERGY_FACTOR_VERTICAL * (1.0 + random.random())
        direct_bias = config.DIRECT_PATH_BIAS * (0.2 + random.random() * 1.8)  # Much wider variation

        print(f"  Trying direct path with varied parameters")
        alt_path, alt_energy = plan_drone_path_downsampled(
            terrain_path,
            obstacles_path,
            start_point,
            end_point,
            downsample_factor=downsample_factor,
            max_runtime_seconds=max_runtime_seconds,
            energy_factor_horizontal=energy_horizontal,
            energy_factor_vertical=energy_vertical,
            direct_path_bias=direct_bias
        )

        if alt_path:
            if smooth:
                print(f"  Smoothing alternative direct path")
                smooth_alt_path = smooth_drone_path(
                    alt_path,
                    terrain,
                    obstacles,
                    transform,
                    max_slope_deg=config.MAX_SLOPE_DEG
                )

                smooth_alt_energy = calculate_path_energy(
                    smooth_alt_path,
                    energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
                    energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
                )

                paths.append((smooth_alt_path, smooth_alt_energy, f"Alternative {i}"))
            else:
                paths.append((alt_path, alt_energy, f"Alternative {i}"))

    return paths


def visualize_multiple_paths(paths, terrain, obstacles, transform, output_dir=None, title=None):
    """
    Visualize multiple paths on the same map.

    Args:
        paths: List of (path, energy, name) tuples
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations
        transform: Rasterio transform for coordinate conversion
        output_dir: Directory to save visualization
        title: Custom title for the visualization
    """
    if not paths:
        print("No paths to visualize")
        return

    # Find the most efficient (minimum energy) path
    min_energy = float('inf')
    min_energy_index = 0

    for i, (path, energy, name) in enumerate(paths):
        if energy < min_energy:
            min_energy = energy
            min_energy_index = i

    # Convert all paths to pixel coordinates
    pixel_paths = []

    # Get all coordinates to determine bounds
    all_coords = []

    for path, _, _ in paths:
        pixel_path = []
        for x, y, _ in path:
            row, col = world_to_pixel(x, y, transform)
            pixel_path.append((row, col))
            all_coords.append((row, col))
        pixel_paths.append(pixel_path)

    # Determine the window to show based on path bounds
    rows, cols = zip(*all_coords)
    padding = max(50, int(0.1 * max(len(terrain), len(terrain[0]))))

    min_row = max(0, min(rows) - padding)
    max_row = min(terrain.shape[0], max(rows) + padding)
    min_col = max(0, min(cols) - padding)
    max_col = min(terrain.shape[1], max(cols) + padding)

    # Extract the window for viewing
    terrain_window = terrain[min_row:max_row, min_col:max_col]
    obstacles_window = obstacles[min_row:max_row, min_col:max_col]

    # Adjust path coordinates to the window
    window_paths = []
    for pixel_path in pixel_paths:
        window_path = []
        for row, col in pixel_path:
            window_path.append((row - min_row, col - min_col))
        window_paths.append(window_path)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Create a custom colormap for terrain
    terrain_cmap = plt.cm.terrain.copy()
    terrain_cmap.set_bad('white', alpha=0)  # Make NaN values transparent

    # Plot terrain
    plt.imshow(terrain_window, cmap=terrain_cmap, alpha=0.7)

    # Plot obstacles
    obstacles_vis = np.ma.masked_where(obstacles_window == 0, obstacles_window)
    plt.imshow(obstacles_vis, cmap='binary', alpha=0.5, vmin=0, vmax=1)

    # Plot paths with different colors
    colors = plt.cm.tab10.colors

    for i, (window_path, (path, energy, name)) in enumerate(zip(window_paths, paths)):
        valid_points = [(r, c) for r, c in window_path
                        if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

        if valid_points:
            path_rows, path_cols = zip(*valid_points)

            # Highlight the most efficient path
            if i == min_energy_index:
                plt.plot(path_cols, path_rows, linewidth=4, color='red',
                         label=f"{name} - Energy: {energy:.1f} (BEST)")
            else:
                # Use different color for each path
                color_index = i % len(colors)
                plt.plot(path_cols, path_rows, linewidth=2, linestyle='--',
                         color=colors[color_index], alpha=0.7,
                         label=f"{name} - Energy: {energy:.1f}")

            # Mark start and end only for the first path
            if i == 0:
                plt.plot(valid_points[0][1], valid_points[0][0], 'go', markersize=10, label='Start')
                plt.plot(valid_points[-1][1], valid_points[-1][0], 'ro', markersize=10, label='End')

    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Multiple Drone Flight Paths')

    plt.colorbar(label='Elevation (m)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()

    # Save or show the visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'multiple_paths_{timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return plt.gcf()


def save_multiple_paths_to_csv(paths, output_dir):
    """
    Save multiple paths to CSV files.

    Args:
        paths: List of (path, energy, name) tuples
        output_dir: Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path, energy, name in paths:
        safe_name = name.replace(" ", "_").lower()
        csv_path = os.path.join(output_dir, f'path_{safe_name}_{timestamp}.csv')

        with open(csv_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z', 'Energy'])

            # Calculate cumulative energy
            cumulative_energy = 0

            for i, (x, y, z) in enumerate(path):
                if i > 0:
                    prev_x, prev_y, prev_z = path[i - 1]

                    # Calculate segment energy
                    distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    height_diff = max(0, z - prev_z)
                    segment_energy = (
                            distance * config.ENERGY_FACTOR_HORIZONTAL +
                            height_diff * config.ENERGY_FACTOR_VERTICAL
                    )
                    cumulative_energy += segment_energy

                writer.writerow([x, y, z, cumulative_energy])

        print(f"Path {name} saved to {csv_path}")