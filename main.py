"""
Main entry point for the drone path planning project.
"""
import os
import argparse
import time
import rasterio

import config
from src.data_processing import process_terrain_data, create_terrain_visualizations
from src.path_planning import plan_drone_path_downsampled, get_smaller_test_area
from src.path_smoothing import smooth_drone_path
from src.visualisation import visualize_path, save_path_to_csv, create_path_comparison_visualization
from src.utils import create_downsampled_transform


def process_data(args):
    """Process raw LiDAR data to create terrain and obstacle models."""
    # Process terrain data
    output_files = process_terrain_data(
        args.dsm_dir,
        args.dtm_dir,
        args.output_dir,
        args.obstacle_threshold
    )

    # Create visualizations
    if args.visualize:
        create_terrain_visualizations(args.output_dir)

    return output_files


def generate_path(args, terrain_path, obstacles_path):
    """Generate and visualize a drone flight path."""
    # Get start and end points
    if args.small_area:
        test_area = get_smaller_test_area(terrain_path, scale=0.05)
        start_point = test_area["start_point"]
        end_point = test_area["end_point"]
        bounds = test_area["small_bounds"]
    else:
        # Use full dataset with user-specified or default points
        with rasterio.open(terrain_path) as src:
            bounds = src.bounds

            if args.start_point and args.end_point:
                # Parse user-provided points
                start_x, start_y = map(float, args.start_point.split(','))
                end_x, end_y = map(float, args.end_point.split(','))
                start_point = (start_x, start_y)
                end_point = (end_x, end_y)
            else:
                # Default to points near corners with inset
                width = bounds.right - bounds.left
                height = bounds.top - bounds.bottom
                inset = 0.1  # 10% inset from edges

                start_point = (bounds.left + width * inset, bounds.bottom + height * inset)
                end_point = (bounds.right - width * inset, bounds.top - height * inset)

        bounds = None  # Use full extent for visualization

    print(f"Planning path from {start_point} to {end_point}")

    # Plan the path
    path, energy = plan_drone_path_downsampled(
        terrain_path,
        obstacles_path,
        start_point,
        end_point,
        downsample_factor=args.downsample,
        max_runtime_seconds=args.max_runtime,
        energy_factor_horizontal=config.ENERGY_FACTOR_HORIZONTAL,
        energy_factor_vertical=config.ENERGY_FACTOR_VERTICAL,
        direct_path_bias=config.DIRECT_PATH_BIAS
    )

    if path is None:
        print("No path found within the time limit!")
        return

    print(f"Path found with {len(path)} points and energy cost: {energy:.2f}")

    # Load terrain and obstacles for visualization
    with rasterio.open(terrain_path) as terrain_src:
        terrain = terrain_src.read(1)
        if args.downsample > 1:
            terrain = terrain[::args.downsample, ::args.downsample]
        transform = terrain_src.transform
        if args.downsample > 1:
            transform = create_downsampled_transform(transform, args.downsample)

    with rasterio.open(obstacles_path) as obs_src:
        obstacles = obs_src.read(1)
        if args.downsample > 1:
            obstacles = obstacles[::args.downsample, ::args.downsample]

    # Save the original path
    save_path_to_csv(path, args.output_dir, "original")

    # Apply path smoothing if requested
    if args.smooth:
        smoothed_path = smooth_drone_path(
            path,
            terrain,
            obstacles,
            transform,
            max_slope_deg=args.max_slope
        )

        print(f"Smoothed path has {len(smoothed_path)} points")

        # Save the smoothed path
        save_path_to_csv(smoothed_path, args.output_dir, "smoothed")

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
            bounds=bounds,
            title='Drone Flight Path with Terrain and Obstacles',
            output_dir=args.output_dir
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Drone Flight Path Optimization')

    # Mode selection
    parser.add_argument('--process_data', action='store_true',
                        help='Process LiDAR data to create terrain models')
    parser.add_argument('--generate_path', action='store_true',
                        help='Generate optimal drone flight path')

    # Data processing options
    parser.add_argument('--dsm_dir', default=config.DSM_DIR,
                        help='Directory containing DSM files')
    parser.add_argument('--dtm_dir', default=config.DTM_DIR,
                        help='Directory containing DTM files')
    parser.add_argument('--obstacle_threshold', type=float, default=config.OBSTACLE_THRESHOLD,
                        help='Height threshold for obstacles (meters)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of the processed data')

    # Path planning options
    parser.add_argument('--terrain', default=config.MERGED_DTM_PATH,
                        help='Path to terrain raster file (DTM)')
    parser.add_argument('--obstacles', default=config.OBSTACLES_PATH,
                        help='Path to obstacles raster file')
    parser.add_argument('--start_point',
                        help='Start point as "x,y" coordinates (overrides small_area)')
    parser.add_argument('--end_point',
                        help='End point as "x,y" coordinates (overrides small_area)')
    parser.add_argument('--small_area', action='store_true',
                        help='Use a small test area')
    parser.add_argument('--downsample', type=int, default=config.DOWNSAMPLE_FACTOR,
                        help='Downsampling factor')
    parser.add_argument('--max_runtime', type=int, default=config.MAX_RUNTIME_SECONDS,
                        help='Maximum runtime in seconds')

    # Path smoothing options
    parser.add_argument('--smooth', action='store_true',
                        help='Apply path smoothing')
    parser.add_argument('--max_slope', type=float, default=config.MAX_SLOPE_DEG,
                        help='Maximum slope in degrees for smoothing')

    # General options
    parser.add_argument('--output_dir', default=config.OUTPUT_DIR,
                        help='Directory for output files')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (process data, small area, smoothing)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Process data if requested or in demo mode
    if args.process_data or args.demo:
        print("Processing LiDAR data...")
        output_files = process_data(args)
        terrain_path = output_files["dtm"]
        obstacles_path = output_files["obstacles"]
    else:
        terrain_path = args.terrain
        obstacles_path = args.obstacles

    # Generate path if requested or in demo mode
    if args.generate_path or args.demo:
        print("Generating drone flight path...")

        # In demo mode, always use small area and smoothing
        if args.demo:
            args.small_area = True
            args.smooth = True

        generate_path(args, terrain_path, obstacles_path)

    print(f"All operations completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error: {e}")
        traceback.print_exc()