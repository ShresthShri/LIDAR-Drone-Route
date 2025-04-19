"""
Script to generate and visualize drone flight paths.
"""
import os
import argparse
import rasterio
import time
import numpy as np
from datetime import datetime

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.path_planning import plan_drone_path_downsampled, get_smaller_test_area
from src.path_smoothing import smooth_drone_path
from src.visualisation import visualize_path, save_path_to_csv, create_path_comparison_visualization
from src.utils import create_downsampled_transform


def main():
    """Main function to generate and visualize drone flight paths."""
    parser = argparse.ArgumentParser(description='Generate drone flight paths')
    parser.add_argument('--terrain', default=config.MERGED_DTM_PATH,
                        help='Path to terrain raster file')
    parser.add_argument('--obstacles', default=config.OBSTACLES_PATH,
                        help='Path to obstacles raster file')
    parser.add_argument('--output_dir', default=config.RUN_OUTPUT_DIR,
                        help='Directory for output files')
    parser.add_argument('--downsample', type=int, default=config.DOWNSAMPLE_FACTOR,
                        help='Downsampling factor')
    parser.add_argument('--max_runtime', type=int, default=config.MAX_RUNTIME_SECONDS,
                        help='Maximum runtime in seconds')
    parser.add_argument('--small_area', action='store_true',
                        help='Use a small test area')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply path smoothing')
    parser.add_argument('--max_slope', type=float, default=config.MAX_SLOPE_DEG,
                        help='Maximum slope in degrees for smoothing')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with all options enabled')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print(f"Terrain data: {args.terrain}")
    print(f"Obstacles data: {args.obstacles}")
    print(f"Output directory: {args.output_dir}")
    print(f"Downsample factor: {args.downsample}")
    print(f"Maximum runtime: {args.max_runtime} seconds")

    # Check if files exist
    if not os.path.exists(args.terrain):
        print(f"Error: Terrain file not found at {args.terrain}")
        return

    if not os.path.exists(args.obstacles):
        print(f"Error: Obstacles file not found at {args.obstacles}")
        return

    # Get start and end points for path planning
    if args.small_area or args.demo:
        print("Using a small test area")
        test_area = get_smaller_test_area(args.terrain, scale=0.05)
        start_point = test_area["start_point"]
        end_point = test_area["end_point"]
        bounds = test_area["small_bounds"]

        print(f"Planning path from {start_point} to {end_point}")
    else:
        # Use the full dataset and let the user specify start/end
        print("Using the full dataset")
        with rasterio.open(args.terrain) as src:
            bounds = src.bounds

            # Default to points near corners with inset
            width = bounds.right - bounds.left
            height = bounds.top - bounds.bottom
            inset = 0.1  # 10% inset from edges

            start_point = (bounds.left + width * inset, bounds.bottom + height * inset)
            end_point = (bounds.right - width * inset, bounds.top - height * inset)

        print(f"Planning path from {start_point} to {end_point}")

    # Start timing
    start_time = time.time()

    # Plan the path
    print(f"Planning drone path with downsampling factor {args.downsample}...")
    path, energy = plan_drone_path_downsampled(
        args.terrain,
        args.obstacles,
        start_point,
        end_point,
        downsample_factor=args.downsample,
        max_runtime_seconds=args.max_runtime,
        energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
        energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL,
        direct_path_bias=config.DIRECT_PATH_BIAS
    )

    # Check if a path was found
    if path is None:
        print("No path found within the time limit!")
        return

    path_time = time.time() - start_time
    print(f"Path planning completed in {path_time:.2f} seconds")
    print(f"Path found with {len(path)} points and energy cost: {energy:.2f}")

    # Load terrain and obstacles for visualization
    with rasterio.open(args.terrain) as terrain_src:
        terrain = terrain_src.read(1)
        if args.downsample > 1:
            # Downsample to match the path planning resolution
            terrain = terrain[::args.downsample, ::args.downsample]
        transform = terrain_src.transform
        if args.downsample > 1:
            # Update transform for downsampled data
            transform = create_downsampled_transform(transform, args.downsample)

    with rasterio.open(args.obstacles) as obs_src:
        obstacles = obs_src.read(1)
        if args.downsample > 1:
            # Downsample to match the path planning resolution
            obstacles = obstacles[::args.downsample, ::args.downsample]

    # Save the original path
    save_path_to_csv(
        path,
        args.output_dir,
        path_type="original",
        energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
        energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
    )

    # Apply path smoothing if requested
    if args.smooth or args.demo:
        print("Applying path smoothing...")
        smooth_start_time = time.time()

        smoothed_path = smooth_drone_path(
            path,
            terrain,
            obstacles,
            transform,
            max_slope_deg=args.max_slope
        )

        smooth_time = time.time() - smooth_start_time
        print(f"Path smoothing completed in {smooth_time:.2f} seconds")
        print(f"Smoothed path has {len(smoothed_path)} points")

        # Calculate energy for the smoothed path
        smooth_energy = 0
        for i in range(1, len(smoothed_path)):
            prev_x, prev_y, prev_z = smoothed_path[i - 1]
            x, y, z = smoothed_path[i]

            # Calculate horizontal distance
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            # Calculate vertical movement (only penalize climbing)
            height_diff = max(0, z - prev_z)

            # Calculate segment energy
            segment_energy = (
                    distance * config.ENERGY_FACTOR_HORIZONTAL +
                    height_diff * config.ENERGY_FACTOR_VERTICAL
            )

            smooth_energy += segment_energy

        print(f"Smoothed path energy cost: {smooth_energy:.2f}")
        energy_diff = ((energy - smooth_energy) / energy) * 100
        print(f"Energy difference: {energy_diff:.2f}%")

        # Save the smoothed path
        save_path_to_csv(
            smoothed_path,
            args.output_dir,
            path_type="smoothed",
            energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
            energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL
        )

        # Create comparison visualization
        create_path_comparison_visualization(
            path,
            smoothed_path,
            terrain,
            obstacles,
            transform,
            output_dir=args.output_dir
        )
    else:
        # Just visualize the original path
        visualize_path(
            path,
            terrain,
            obstacles,
            transform,
            bounds=bounds if args.small_area or args.demo else None,
            title='Drone Flight Path with Terrain and Obstacles',
            output_dir=args.output_dir
        )

    print(f"All operations completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in path generation: {e}")
        traceback.print_exc()