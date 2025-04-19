"""
Script to generate paths between Edinburgh landmarks.
"""
import os
import sys
import argparse
import rasterio
import time
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.landmarks import get_landmark_coordinates, list_landmarks
from src.multiple_paths import generate_multiple_paths, visualize_multiple_paths, save_multiple_paths_to_csv
from src.utils import create_downsampled_transform


def main():
    """Main function to generate paths between landmarks."""
    parser = argparse.ArgumentParser(description='Generate drone flight paths between Edinburgh landmarks')

    parser.add_argument('--from_landmark', required=True,
                        help='Starting landmark name')
    parser.add_argument('--to_landmark', required=True,
                        help='Destination landmark name')
    parser.add_argument('--terrain', default=config.MERGED_DTM_PATH,
                        help='Path to terrain raster file')
    parser.add_argument('--obstacles', default=config.OBSTACLES_PATH,
                        help='Path to obstacles raster file')
    parser.add_argument('--output_dir', default=config.RUN_OUTPUT_DIR,
                        help='Directory for output files')
    parser.add_argument('--num_paths', type=int, default=3,
                        help='Number of alternative paths to generate')
    parser.add_argument('--downsample', type=int, default=5,
                        help='Downsampling factor (lower is more detailed but slower)')
    parser.add_argument('--max_runtime', type=int, default=120,
                        help='Maximum runtime in seconds for each path')
    parser.add_argument('--list_landmarks', action='store_true',
                        help='List all available landmarks and exit')
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable path smoothing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # List landmarks if requested
    if args.list_landmarks:
        print("Available Edinburgh landmarks:")
        for landmark in list_landmarks():
            print(f"  - {landmark}")
        return

    # Check if the specified landmarks exist
    start_coords = get_landmark_coordinates(args.from_landmark)
    end_coords = get_landmark_coordinates(args.to_landmark)

    if not start_coords:
        print(f"Error: Landmark '{args.from_landmark}' not found. "
              f"Use --list_landmarks to see available options.")
        return

    if not end_coords:
        print(f"Error: Landmark '{args.to_landmark}' not found. "
              f"Use --list_landmarks to see available options.")
        return

    # Print the plan
    print(f"Planning drone flight from {args.from_landmark} to {args.to_landmark}")
    print(f"Start coordinates: {start_coords}")
    print(f"End coordinates: {end_coords}")
    print(f"Generating {args.num_paths} alternative paths")
    print(f"Downsampling factor: {args.downsample}")
    print(f"Maximum runtime per path: {args.max_runtime} seconds")
    print(f"Path smoothing: {'Disabled' if args.no_smooth else 'Enabled'}")

    # Check if files exist
    if not os.path.exists(args.terrain):
        print(f"Error: Terrain file not found at {args.terrain}")
        return

    if not os.path.exists(args.obstacles):
        print(f"Error: Obstacles file not found at {args.obstacles}")
        return

    # Start timing
    start_time = time.time()

    # Generate multiple paths
    paths = generate_multiple_paths(
        args.terrain,
        args.obstacles,
        start_coords,
        end_coords,
        num_paths=args.num_paths,
        downsample_factor=args.downsample,
        max_runtime_seconds=args.max_runtime,
        smooth=not args.no_smooth
    )

    if not paths:
        print("Failed to generate any paths.")
        return

    print(f"Generated {len(paths)} paths")

    # Load terrain and obstacles for visualization
    with rasterio.open(args.terrain) as terrain_src:
        terrain = terrain_src.read(1)
        if args.downsample > 1:
            terrain = terrain[::args.downsample, ::args.downsample]
        transform = terrain_src.transform
        if args.downsample > 1:
            transform = create_downsampled_transform(transform, args.downsample)

    with rasterio.open(args.obstacles) as obs_src:
        obstacles = obs_src.read(1)
        if args.downsample > 1:
            obstacles = obstacles[::args.downsample, ::args.downsample]

    # Create a title for the visualization
    title = f"Drone Flight Paths: {args.from_landmark} to {args.to_landmark}"

    # Visualize the paths
    visualize_multiple_paths(paths, terrain, obstacles, transform,
                             output_dir=os.path.join(args.output_dir, "visualizations"), title=title)

    # Save the paths to CSV
    save_multiple_paths_to_csv(paths, os.path.join(args.output_dir, "paths"))

    print(f"All operations completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error: {e}")
        traceback.print_exc()