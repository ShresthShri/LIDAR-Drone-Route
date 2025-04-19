"""
Script to process LiDAR data and generate terrain and obstacle models.
"""
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_processing import process_terrain_data, create_terrain_visualizations, explore_point_cloud


def main():
    """Main function to process LiDAR data."""
    parser = argparse.ArgumentParser(description='Process LiDAR data')
    parser.add_argument('--dsm_dir', default=config.DSM_DIR,
                        help='Directory containing DSM files')
    parser.add_argument('--dtm_dir', default=config.DTM_DIR,
                        help='Directory containing DTM files')
    parser.add_argument('--laz_dir', default=config.LAZ_DIR,
                        help='Directory containing LAZ files')
    parser.add_argument('--output_dir', default=config.OUTPUT_DIR,
                        help='Directory for output files')
    parser.add_argument('--obstacle_threshold', type=float, default=config.OBSTACLE_THRESHOLD,
                        help='Height threshold for obstacles (meters)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of the processed data')
    parser.add_argument('--explore_point_cloud', action='store_true',
                        help='Explore and visualize point cloud data')

    args = parser.parse_args()

    # Print configuration
    print(f"DSM directory: {args.dsm_dir}")
    print(f"DTM directory: {args.dtm_dir}")
    print(f"LAZ directory: {args.laz_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Obstacle threshold: {args.obstacle_threshold} meters")

    # Check if directories exist
    for dir_path, dir_name in [
        (args.dsm_dir, "DSM"),
        (args.dtm_dir, "DTM"),
        (args.laz_dir, "LAZ")
    ]:
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_name} directory not found at {dir_path}")
            if dir_name in ["DSM", "DTM"]:
                print(f"Error: {dir_name} directory is required for processing")
                return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process terrain data
    output_files = process_terrain_data(
        args.dsm_dir,
        args.dtm_dir,
        args.output_dir,
        args.obstacle_threshold
    )

    print("Terrain data processing complete!")
    print(f"Output files saved to {args.output_dir}:")
    for key, path in output_files.items():
        print(f"  - {key}: {os.path.basename(path)}")

    # Create visualizations if requested
    if args.visualize:
        create_terrain_visualizations(args.output_dir)

    # Explore point cloud if requested and available
    if args.explore_point_cloud and os.path.exists(args.laz_dir):
        try:
            explore_point_cloud(args.laz_dir, args.output_dir)
        except Exception as e:
            print(f"Error exploring point cloud: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in data processing: {e}")
        traceback.print_exc()