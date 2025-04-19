"""
Script to generate visibly diverse paths between Edinburgh landmarks.
"""
import os
import sys
import argparse
import rasterio
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.landmarks import get_landmark_coordinates, list_landmarks
from src.utils import create_downsampled_transform, world_to_pixel, calculate_path_energy
from src.path_planning import plan_drone_path_downsampled
from src.path_smoothing import smooth_drone_path


def generate_diverse_paths(terrain_path, obstacles_path, start_point, end_point, num_paths=5,
                           downsample_factor=3, max_runtime_seconds=60, smooth=True):
    """
    Generate multiple diverse drone paths between two points by using waypoints.

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
        List of (path, energy, name) tuples
    """
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
        if smooth:
            # Load terrain and obstacles for smoothing
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

            # Smooth the baseline path
            smoothed_baseline = smooth_drone_path(
                baseline_path,
                terrain,
                obstacles,
                transform,
                max_slope_deg=config.MAX_SLOPE_DEG
            )

            smoothed_energy = calculate_path_energy(
                smoothed_baseline,
                energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
                energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
            )

            paths.append((smoothed_baseline, smoothed_energy, "Baseline"))
        else:
            paths.append((baseline_path, baseline_energy, "Baseline"))
    else:
        print("Failed to generate baseline path. Try increasing max_runtime.")
        return []

    # Generate waypoints to create diverse paths
    direct_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    path_length = np.sqrt(direct_vector[0] ** 2 + direct_vector[1] ** 2)

    # Normalize the direct vector
    direct_unit = (direct_vector[0] / path_length, direct_vector[1] / path_length)

    # Calculate perpendicular vector (90 degrees to direct vector)
    perp_vector = (-direct_unit[1], direct_unit[0])

    # Generate a diverse set of alternative paths
    print(f"Generating {num_paths} alternative paths...")
    for i in range(1, num_paths + 1):
        # Create different offsets to force path diversity
        if i % 3 == 0:
            # Northern routes (positive perpendicular offset)
            offset_scale = 0.1 + (i // 3) * 0.1  # 0.1, 0.2, 0.3, etc.
            path_name = f"Northern Route {i // 3}"
        elif i % 3 == 1:
            # Southern routes (negative perpendicular offset)
            offset_scale = -(0.1 + (i // 3) * 0.1)  # -0.1, -0.2, -0.3, etc.
            path_name = f"Southern Route {i // 3 + 1}"
        else:
            # Zigzag routes (alternating waypoints)
            offset_scale = 0.15 * (1 if i % 4 == 0 else -1)
            path_name = f"Zigzag Route {i // 3 + 1}"

        # Create waypoints with offsets from the direct path
        waypoints = []

        # For zigzag routes, create multiple waypoints
        if i % 3 == 2:
            num_waypoints = 3
            for w in range(num_waypoints):
                # Alternate direction for each waypoint
                wp_offset = offset_scale * (-1 if w % 2 == 0 else 1)

                # Position along the path (spread waypoints evenly)
                path_pos = 0.25 + (w * 0.5 / (num_waypoints - 1))

                wp_x = start_point[0] + direct_vector[0] * path_pos + perp_vector[0] * path_length * wp_offset
                wp_y = start_point[1] + direct_vector[1] * path_pos + perp_vector[1] * path_length * wp_offset

                waypoints.append((wp_x, wp_y))
        else:
            # For simple northern/southern routes, use one waypoint
            path_pos = 0.5  # Middle of the path

            wp_x = start_point[0] + direct_vector[0] * path_pos + perp_vector[0] * path_length * offset_scale
            wp_y = start_point[1] + direct_vector[1] * path_pos + perp_vector[1] * path_length * offset_scale

            waypoints.append((wp_x, wp_y))

        # Generate path through waypoints
        alt_path_segments = []
        total_energy = 0

        # Start with path from start to first waypoint
        current_point = start_point

        # Also vary other path parameters to maximize diversity
        energy_h_factor = config.ENERGY_FACTOR_HORIZONTAL * (0.8 + random.random() * 0.4)  # 0.8-1.2
        energy_v_factor = config.ENERGY_FACTOR_VERTICAL * (0.7 + random.random() * 0.6)  # 0.7-1.3
        bias_factor = config.DIRECT_PATH_BIAS * (0.5 + random.random())  # 0.5-1.5

        # Add each waypoint segment
        for waypoint in waypoints:
            print(f"  Finding path segment to waypoint: {waypoint}")
            segment_path, segment_energy = plan_drone_path_downsampled(
                terrain_path,
                obstacles_path,
                current_point,
                waypoint,
                downsample_factor=downsample_factor,
                max_runtime_seconds=max_runtime_seconds // (len(waypoints) + 1),
                energy_factor_horizontal=energy_h_factor,
                energy_factor_vertical=energy_v_factor,
                direct_path_bias=bias_factor
            )

            if segment_path:
                alt_path_segments.append(segment_path)
                total_energy += segment_energy
                current_point = waypoint
            else:
                print(f"  Failed to find path segment to waypoint {waypoint}")
                break

        # Add final segment from last waypoint to destination
        if alt_path_segments and current_point != end_point:
            final_segment, final_energy = plan_drone_path_downsampled(
                terrain_path,
                obstacles_path,
                current_point,
                end_point,
                downsample_factor=downsample_factor,
                max_runtime_seconds=max_runtime_seconds // (len(waypoints) + 1),
                energy_factor_horizontal=energy_h_factor,
                energy_factor_vertical=energy_v_factor,
                direct_path_bias=bias_factor
            )

            if final_segment:
                alt_path_segments.append(final_segment)
                total_energy += final_energy

        # Combine all segments into a single path
        if alt_path_segments:
            combined_path = alt_path_segments[0]
            for segment in alt_path_segments[1:]:
                # Skip the first point of subsequent segments to avoid duplicates
                combined_path.extend(segment[1:])

            # Apply smoothing if requested
            if smooth:
                print(f"  Smoothing alternative path {i}")
                with rasterio.open(terrain_path) as terrain_src:
                    terrain = terrain_src.read(1)
                    if downsample_factor > 1:
                        terrain = terrain[::downsample_factor, ::downsample_factor]
                    transform = terrain_src.transform
                    if downsample_factor > 1:
                        transform = create