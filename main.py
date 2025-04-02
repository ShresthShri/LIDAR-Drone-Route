import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from pathlib import Path

# Set paths to your data directories
base_dir = '/Users/shresthshrivastava/Downloads/LIDAR Side Project'
dsm_dir = os.path.join(base_dir, "scotland-dsm-2m_5938282/nt")
dtm_dir = os.path.join(base_dir, "scotland-dtm-2m_5938283/nt")
output_dir = os.path.join(base_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

# Step 1: Find all DSM and DTM files (.asc format)
dsm_files = [os.path.join(dsm_dir, f) for f in os.listdir(dsm_dir)
             if f.endswith('.asc')]
dtm_files = [os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir)
             if f.endswith('.asc')]

print(f"Found {len(dsm_files)} DSM files and {len(dtm_files)} DTM files")

# Step 2: Merge DSM files
dsm_sources = [rasterio.open(f) for f in dsm_files]
dtm_sources = [rasterio.open(f) for f in dtm_files]

# Merge rasters
print("Merging DSM files...")
dsm_mosaic, dsm_transform = merge(dsm_sources)
print("Merging DTM files...")
dtm_mosaic, dtm_transform = merge(dtm_sources)

# Close sources
for src in dsm_sources + dtm_sources:
    src.close()

# Step 3: Save merged files
# Get metadata from first file for output
with rasterio.open(dsm_files[0]) as src:
    dsm_meta = src.meta.copy()
    dsm_meta.update({
        "height": dsm_mosaic.shape[1],
        "width": dsm_mosaic.shape[2],
        "transform": dsm_transform,
        "driver": "GTiff"  # We'll save as GeoTIFF for easier processing
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

# Step 4: Calculate and save object heights
dsm_data = dsm_mosaic[0]  # First band
dtm_data = dtm_mosaic[0]  # First band

# Create object height model (DSM - DTM)
print("Calculating object heights...")
object_heights = dsm_data - dtm_data

# Save object heights
heights_path = os.path.join(output_dir, "object_heights.tif")
with rasterio.open(dsm_merged_path) as src:
    height_meta = src.meta.copy()
    with rasterio.open(heights_path, "w", **height_meta) as dest:
        dest.write(object_heights, 1)

# Create obstacle map (binary raster where height > threshold)
obstacle_threshold = 10  # meters
obstacles = (object_heights > obstacle_threshold).astype(np.uint8)

# Save obstacle map with correct metadata
obstacle_path = os.path.join(output_dir, "obstacles.tif")
with rasterio.open(dsm_merged_path) as src:
    # Create fresh metadata for the obstacle map
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

# Step 5: Create visualizations
print("Creating visualizations...")
plt.figure(figsize=(12, 10))
plt.imshow(dsm_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Scotland Digital Surface Model (DSM)')
plt.savefig(os.path.join(output_dir, 'dsm_visualization.png'))

plt.figure(figsize=(12, 10))
plt.imshow(dtm_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Scotland Digital Terrain Model (DTM)')
plt.savefig(os.path.join(output_dir, 'dtm_visualization.png'))

plt.figure(figsize=(12, 10))
plt.imshow(object_heights, cmap='viridis')
plt.colorbar(label='Height (m)')
plt.title('Object Heights (Buildings, Trees, etc.)')
plt.savefig(os.path.join(output_dir, 'object_heights_visualization.png'))

plt.figure(figsize=(12, 10))
plt.imshow(obstacles, cmap='binary')
plt.title('Obstacles (Objects > 10m tall)')
plt.savefig(os.path.join(output_dir, 'obstacles_visualization.png'))


# Optional: Explore point cloud data if available
def explore_point_cloud():
    try:
        import laspy

        # Path to LAZ directory
        laz_dir = os.path.join(base_dir, "scotland-laz-1_5938284/nt")

        # Find LAZ files
        laz_files = [os.path.join(laz_dir, f) for f in os.listdir(laz_dir)
                     if f.endswith('.laz')]

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
        print("Point cloud exploration complete!")

    except ImportError:
        print("Please install laspy to work with LAZ files: pip install laspy")
    except Exception as e:
        print(f"Error exploring point cloud: {e}")


# Try to explore point cloud if data is available
try:
    explore_point_cloud()
except Exception as e:
    print(f"Could not explore point cloud: {e}")

print("Processing complete!")