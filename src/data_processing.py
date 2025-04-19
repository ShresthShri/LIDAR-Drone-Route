"""
Functions for processing LiDAR and elevation data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
import config


def find_data_files(dir_path, extension='.asc'):
    """
    Find all files with the specified extension in a directory.

    Args:
        dir_path: Directory to search
        extension: File extension to filter by

    Returns:
        List of file paths
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.endswith(extension)]


def merge_raster_files(file_paths):
    """
    Merge multiple raster files into a single mosaic.

    Args:
        file_paths: List of file paths to raster files

    Returns:
        (mosaic, transform) tuple with the merged data and its transform
    """
    sources = [rasterio.open(f) for f in file_paths]

    try:
        mosaic, transform = merge(sources)
    finally:
        # Close all sources
        for src in sources:
            src.close()

    return mosaic, transform


def process_terrain_data(dsm_dir, dtm_dir, output_dir, obstacle_threshold=10):
    """
    Process DSM and DTM data to create merged terrain models and obstacle maps.

    Args:
        dsm_dir: Directory containing DSM files
        dtm_dir: Directory containing DTM files
        output_dir: Directory for output files
        obstacle_threshold: Height threshold for obstacles (meters)

    Returns:
        Dictionary of output file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all DSM and DTM files
    dsm_files = find_data_files(dsm_dir)
    dtm_files = find_data_files(dtm_dir)

    print(f"Found {len(dsm_files)} DSM files and {len(dtm_files)} DTM files")

    # Merge DSM files
    print("Merging DSM files...")
    dsm_mosaic, dsm_transform = merge_raster_files(dsm_files)

    # Merge DTM files
    print("Merging DTM files...")
    dtm_mosaic, dtm_transform = merge_raster_files(dtm_files)

    # Get metadata for output files
    with rasterio.open(dsm_files[0]) as src:
        dsm_meta = src.meta.copy()
        dsm_meta.update({
            "height": dsm_mosaic.shape[1],
            "width": dsm_mosaic.shape[2],
            "transform": dsm_transform,
            "driver": "GTiff"
        })

    with rasterio.open(dtm_files[0]) as src:
        dtm_meta = src.meta.copy()
        dtm_meta.update({
            "height": dtm_mosaic.shape[1],
            "width": dtm_mosaic.shape[2],
            "transform": dtm_transform,
            "driver": "GTiff"
        })

    # Write merged files
    dsm_merged_path = os.path.join(output_dir, "merged_dsm.tif")
    dtm_merged_path = os.path.join(output_dir, "merged_dtm.tif")

    with rasterio.open(dsm_merged_path, "w", **dsm_meta) as dest:
        dest.write(dsm_mosaic)

    with rasterio.open(dtm_merged_path, "w", **dtm_meta) as dest:
        dest.write(dtm_mosaic)

    # Calculate and save object heights
    dsm_data = dsm_mosaic[0]  # First band
    dtm_data = dtm_mosaic[0]  # First band

    print("Calculating object heights...")
    object_heights = dsm_data - dtm_data

    # Save object heights
    heights_path = os.path.join(output_dir, "object_heights.tif")
    with rasterio.open(dsm_merged_path) as src:
        height_meta = src.meta.copy()
        with rasterio.open(heights_path, "w", **height_meta) as dest:
            dest.write(object_heights, 1)

    # Create obstacle map (binary raster where height > threshold)
    obstacles = (object_heights > obstacle_threshold).astype(np.uint8)

    # Save obstacle map with correct metadata
    obstacle_path = os.path.join(output_dir, "obstacles.tif")
    with rasterio.open(dsm_merged_path) as src:
        obstacle_meta = {
            'driver': 'GTiff',
            'dtype': rasterio.uint8,
            'nodata': 0,
            'width': src.width,
            'height': src.height,
            'count': 1,
            'crs': src.crs,
            'transform': src.transform
        }

        with rasterio.open(obstacle_path, "w", **obstacle_meta) as dest:
            dest.write(obstacles, 1)

    return {
        "dsm": dsm_merged_path,
        "dtm": dtm_merged_path,
        "heights": heights_path,
        "obstacles": obstacle_path
    }


def create_terrain_visualizations(output_dir):
    """
    Create visualizations of the processed terrain data.

    Args:
        output_dir: Directory containing processed data and for output visualizations
    """
    # Load processed data
    dsm_path = os.path.join(output_dir, "merged_dsm.tif")
    dtm_path = os.path.join(output_dir, "merged_dtm.tif")
    heights_path = os.path.join(output_dir, "object_heights.tif")
    obstacles_path = os.path.join(output_dir, "obstacles.tif")

    with rasterio.open(dsm_path) as src:
        dsm_data = src.read(1)

    with rasterio.open(dtm_path) as src:
        dtm_data = src.read(1)

    with rasterio.open(heights_path) as src:
        heights_data = src.read(1)

    with rasterio.open(obstacles_path) as src:
        obstacles_data = src.read(1)

    # Create visualizations
    print("Creating visualizations...")

    plt.figure(figsize=(12, 10))
    plt.imshow(dsm_data, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Digital Surface Model (DSM)')
    plt.savefig(os.path.join(output_dir, 'dsm_visualization.png'))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(dtm_data, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Digital Terrain Model (DTM)')
    plt.savefig(os.path.join(output_dir, 'dtm_visualization.png'))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(heights_data, cmap='viridis')
    plt.colorbar(label='Height (m)')
    plt.title('Object Heights (Buildings, Trees, etc.)')
    plt.savefig(os.path.join(output_dir, 'object_heights_visualization.png'))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.imshow(obstacles_data, cmap='binary')
    plt.title('Obstacles (Objects > 10m tall)')
    plt.savefig(os.path.join(output_dir, 'obstacles_visualization.png'))
    plt.close()

    print("Visualizations saved to", output_dir)


def explore_point_cloud(laz_dir, output_dir):
    """
    Explore and visualize LiDAR point cloud data.

    Args:
        laz_dir: Directory containing LAZ files
        output_dir: Directory for output visualizations
    """
    try:
        import laspy

        # Find LAZ files
        laz_files = find_data_files(laz_dir, extension='.laz')

        if not laz_files:
            print("No LAZ files found.")
            return

        # Load the first LAZ file to explore
        sample_file = laz_files[0]
        print(f"Exploring LAZ file: {os.path.basename(sample_file)}")

        with laspy.open(sample_file) as fh:
            las = fh.read()

        # Get basic information
        print(f"Point count: {len(las.points)}")
        print(f"Point format: {las.point_format.id}")

        # Get coordinate bounds
        print(f"X range: {las.header.x_min} to {las.header.x_max}")
        print(f"Y range: {las.header.y_min} to {las.header.y_max}")
        print(f"Z range: {las.header.z_min} to {las.header.z_max}")

        # Sample some points
        sample_size = min(1000, len(las.points))
        indices = np.random.choice(len(las.points), sample_size, replace=False)

        # Extract coordinates
        x = las.x[indices]
        y = las.y[indices]
        z = las.z[indices]

        # Plot a 3D scatter of sample points
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Color points by height
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        plt.colorbar(sc, label='Elevation (m)')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Sample (1000 points)')

        plt.savefig(os.path.join(output_dir, 'point_cloud_sample.png'))
        plt.close()
        print("Point cloud exploration complete!")

    except ImportError:
        print("Please install laspy to work with LAZ files: pip install laspy")
    except Exception as e:
        print(f"Error exploring point cloud: {e}")


if __name__ == "__main__":
    # Example usage when run directly
    process_terrain_data(
        config.DSM_DIR,
        config.DTM_DIR,
        config.OUTPUT_DIR,
        config.OBSTACLE_THRESHOLD
    )

    create_terrain_visualizations(config.OUTPUT_DIR)

    try:
        explore_point_cloud(config.LAZ_DIR, config.OUTPUT_DIR)
    except Exception as e:
        print(f"Could not explore point cloud: {e}")