"""
Visualization tools for drone flight paths.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
import csv

from src.utils import world_to_pixel, calculate_path_energy


def visualize_path(path, terrain, obstacles, transform, bounds=None, title='Drone Flight Path',
                   output_dir=None, compare_path=None):
    """
    Visualize a drone path overlaid on terrain and obstacles.

    Args:
        path: List of (x, y, z) coordinates
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform for coordinate conversion
        bounds: Tuple of (left, bottom, right, top) for visualization bounds
        title: Plot title
        output_dir: Directory to save visualization (if None, only shows)
        compare_path: Optional second path for comparison visualization
    """
    if not path:
        print("No path to visualize")
        return

    # Convert path to pixel coordinates
    pixel_path = []
    for x, y, _ in path:
        row, col = world_to_pixel(x, y, transform)
        pixel_path.append((row, col))

    # If we have a comparison path, convert it too
    pixel_compare = None
    if compare_path:
        pixel_compare = []
        for x, y, _ in compare_path:
            row, col = world_to_pixel(x, y, transform)
            pixel_compare.append((row, col))

    # Determine the window to show based on path bounds or specified bounds
    if bounds:
        # Convert world bounds to pixel coordinates
        left, bottom, right, top = bounds
        top_left = world_to_pixel(left, top, transform)
        bottom_right = world_to_pixel(right, bottom, transform)

        min_row = max(0, min(top_left[0], bottom_right[0]))
        max_row = min(terrain.shape[0], max(top_left[0], bottom_right[0]))
        min_col = max(0, min(top_left[1], bottom_right[1]))
        max_col = min(terrain.shape[1], max(top_left[1], bottom_right[1]))
    else:
        # Determine bounds from the path with some padding
        rows, cols = zip(*pixel_path)
        padding = max(50, int(0.1 * max(len(terrain), len(terrain[0]))))

        min_row = max(0, min(rows) - padding)
        max_row = min(terrain.shape[0], max(rows) + padding)
        min_col = max(0, min(cols) - padding)
        max_col = min(terrain.shape[1], max(cols) + padding)

    # Extract the window for viewing
    terrain_window = terrain[min_row:max_row, min_col:max_col]
    obstacles_window = obstacles[min_row:max_row, min_col:max_col]

    # Adjust path coordinates to the window
    window_path = []
    for row, col in pixel_path:
        window_path.append((row - min_row, col - min_col))

    # Adjust comparison path if present
    window_compare = None
    if pixel_compare:
        window_compare = []
        for row, col in pixel_compare:
            window_compare.append((row - min_row, col - min_col))

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

    # Plot path if we have valid points in our window
    if window_path:
        # Filter points to ensure they're in bounds
        valid_points = [(r, c) for r, c in window_path
                        if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

        if valid_points:
            path_rows, path_cols = zip(*valid_points)
            plt.plot(path_cols, path_rows, 'r-', linewidth=3, label='Drone Path')

            # Mark start and end
            if len(valid_points) > 0:
                plt.plot(valid_points[0][1], valid_points[0][0], 'go', markersize=10, label='Start')
            if len(valid_points) > 1:
                plt.plot(valid_points[-1][1], valid_points[-1][0], 'ro', markersize=10, label='End')

    # Plot comparison path if provided
    if window_compare:
        valid_compare = [(r, c) for r, c in window_compare
                         if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

        if valid_compare:
            compare_rows, compare_cols = zip(*valid_compare)
            plt.plot(compare_cols, compare_rows, 'b--', linewidth=2, label='Smoothed Path')

    plt.title(title)
    plt.colorbar(label='Elevation (m)')
    plt.legend()

    # Save or show the visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_type = "comparison" if compare_path else "original"
        plt.savefig(os.path.join(output_dir, f'drone_path_{path_type}_{timestamp}.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def save_path_to_csv(path, output_dir, path_type="original",
                     energy_factor_horizontal=1.0, energy_factor_vertical=2.0):
    """
    Save a path to a CSV file.

    Args:
        path: List of (x, y, z) coordinates
        output_dir: Directory to save the CSV file
        path_type: String identifier for the path type (original/smoothed)
        energy_factor_horizontal: Energy multiplier for horizontal movement
        energy_factor_vertical: Energy multiplier for vertical climbing
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'drone_path_{path_type}_{timestamp}.csv')

    with open(csv_path, 'w', newline='') as f:
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
                        distance * energy_factor_horizontal +
                        height_diff * energy_factor_vertical
                )
                cumulative_energy += segment_energy

            writer.writerow([x, y, z, cumulative_energy])

    print(f"Path saved to {csv_path}")
    return csv_path


def create_path_comparison_visualization(original_path, smoothed_path, terrain, obstacles,
                                         transform, output_dir=None):
    """
    Create a comparison visualization between original and smoothed paths.

    Args:
        original_path: List of (x, y, z) coordinates for original path
        smoothed_path: List of (x, y, z) coordinates for smoothed path
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform for coordinate conversion
        output_dir: Directory to save visualization
    """
    # Calculate energy for both paths
    original_energy = calculate_path_energy(original_path)
    smoothed_energy = calculate_path_energy(smoothed_path)

    energy_diff = ((original_energy - smoothed_energy) / original_energy) * 100
    energy_diff_str = f"+{energy_diff:.1f}%" if energy_diff < 0 else f"-{abs(energy_diff):.1f}%"

    title = (f'Drone Flight Path Comparison\n'
             f'Original: {len(original_path)} points, Energy: {original_energy:.1f}\n'
             f'Smoothed: {len(smoothed_path)} points, Energy: {smoothed_energy:.1f} ({energy_diff_str})')

    visualize_path(original_path, terrain, obstacles, transform,
                   title=title, output_dir=output_dir, compare_path=smoothed_path)

    # Also create a 3D comparison
    create_3d_path_comparison(original_path, smoothed_path, title, output_dir)


def create_3d_path_comparison(original_path, smoothed_path, title='3D Path Comparison',
                              output_dir=None):
    """
    Create a 3D visualization comparing original and smoothed paths.

    Args:
        original_path: List of (x, y, z) coordinates for original path
        smoothed_path: List of (x, y, z) coordinates for smoothed path
        title: Plot title
        output_dir: Directory to save visualization
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    orig_x = [p[0] for p in original_path]
    orig_y = [p[1] for p in original_path]
    orig_z = [p[2] for p in original_path]

    smooth_x = [p[0] for p in smoothed_path]
    smooth_y = [p[1] for p in smoothed_path]
    smooth_z = [p[2] for p in smoothed_path]

    # Plot paths
    ax.plot(orig_x, orig_y, orig_z, 'r-', linewidth=2, label='Original Path')
    ax.plot(smooth_x, smooth_y, smooth_z, 'b-', linewidth=2, label='Smoothed Path')

    # Mark start and end points
    ax.scatter(orig_x[0], orig_y[0], orig_z[0], c='g', s=100, label='Start')
    ax.scatter(orig_x[-1], orig_y[-1], orig_z[-1], c='r', s=100, label='End')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation (m)')
    ax.set_title(title)
    ax.legend()

    # Save or show the visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'drone_path_3d_comparison_{timestamp}.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_multiple_paths(paths, terrain, obstacles, transform, output_dir=None, title=None, zoom_factor=1.5):
    """
    Visualize multiple paths on the same map with improved zooming and labeling.

    Args:
        paths: List of (path, energy, name) tuples
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations
        transform: Rasterio transform for coordinate conversion
        output_dir: Directory to save visualization
        title: Custom title for the visualization
        zoom_factor: Factor to zoom in (higher values = more zoom)
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

    # Determine the window to show based on path bounds, with improved zooming
    rows, cols = zip(*all_coords)

    # Calculate center of all paths
    center_row = sum(rows) / len(rows)
    center_col = sum(cols) / len(cols)

    # Find the maximum distance from center to any point
    max_distance = max(max(abs(r - center_row) for r in rows),
                       max(abs(c - center_col) for c in cols))

    # Use zoom factor to reduce the view area (higher zoom = smaller view)
    max_distance = max_distance / zoom_factor

    # Calculate window boundaries with some padding
    padding = max(20, int(0.05 * max_distance))

    min_row = max(0, int(center_row - max_distance - padding))
    max_row = min(terrain.shape[0], int(center_row + max_distance + padding))
    min_col = max(0, int(center_col - max_distance - padding))
    max_col = min(terrain.shape[1], int(center_col + max_distance + padding))

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
    plt.figure(figsize=(16, 10))

    # Create a custom colormap for terrain
    terrain_cmap = plt.cm.terrain.copy()
    terrain_cmap.set_bad('white', alpha=0)  # Make NaN values transparent

    # Plot terrain
    plt.imshow(terrain_window, cmap=terrain_cmap, alpha=0.7)

    # Plot obstacles with improved visibility
    obstacles_vis = np.ma.masked_where(obstacles_window == 0, obstacles_window)
    plt.imshow(obstacles_vis, cmap='binary', alpha=0.7, vmin=0, vmax=1)

    # Plot paths with different colors and improved visibility
    colors = plt.cm.tab10.colors

    # Create a separate legend figure to avoid overlapping
    legend_elements = []

    for i, (window_path, (path, energy, name)) in enumerate(zip(window_paths, paths)):
        valid_points = [(r, c) for r, c in window_path
                        if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

        if valid_points:
            path_rows, path_cols = zip(*valid_points)

            # Highlight the most efficient path
            if i == min_energy_index:
                line, = plt.plot(path_cols, path_rows, linewidth=4, color='red')
                legend_elements.append((line, f"{name} - Energy: {energy:.1f} (BEST)"))
            else:
                # Use different color for each path with increased visibility
                color_index = i % len(colors)
                line, = plt.plot(path_cols, path_rows, linewidth=2.5, linestyle='--',
                                 color=colors[color_index], alpha=0.8)
                legend_elements.append((line, f"{name} - Energy: {energy:.1f}"))

            # Mark start and end only for the first path
            if i == 0:
                start_point = plt.plot(valid_points[0][1], valid_points[0][0], 'go', markersize=12, zorder=10)[0]
                end_point = plt.plot(valid_points[-1][1], valid_points[-1][0], 'ro', markersize=12, zorder=10)[0]
                legend_elements.append((start_point, 'Start'))
                legend_elements.append((end_point, 'End'))

    # Set title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Multiple Drone Flight Paths', fontsize=14)

    # Add colorbar for elevation
    cbar = plt.colorbar(label='Elevation (m)')
    cbar.ax.tick_params(labelsize=10)

    # Create a cleaner legend
    lines, labels = zip(*legend_elements)
    plt.legend(lines, labels, loc='upper left', fontsize=10,
               bbox_to_anchor=(1.01, 1), borderaxespad=0)

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


def visualize_zoomed_paths(paths, terrain, obstacles, transform, output_dir=None,
                           title=None, zoom_factor=2.0, highlight_buildings=True):
    """
    Visualize multiple paths with enhanced zoom and building highlighting.

    Args:
        paths: List of (path, energy, name) tuples
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations
        transform: Rasterio transform for coordinate conversion
        output_dir: Directory to save visualization
        title: Custom title for the visualization
        zoom_factor: Factor to zoom in (higher values = more zoom)
        highlight_buildings: Whether to highlight buildings more prominently
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from datetime import datetime
    import os

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

    # Determine the window to show based on path bounds, with improved zooming
    rows, cols = zip(*all_coords)

    # Calculate center of all paths
    center_row = sum(rows) / len(rows)
    center_col = sum(cols) / len(cols)

    # Find the maximum distance from center to any point
    max_distance = max(max(abs(r - center_row) for r in rows),
                       max(abs(c - center_col) for c in cols))

    # Use zoom factor to reduce the view area (higher zoom = smaller view)
    max_distance = max_distance / zoom_factor

    # Calculate window boundaries with some padding
    padding = max(20, int(0.05 * max_distance))

    min_row = max(0, int(center_row - max_distance - padding))
    max_row = min(terrain.shape[0], int(center_row + max_distance + padding))
    min_col = max(0, int(center_col - max_distance - padding))
    max_col = min(terrain.shape[1], int(center_col + max_distance + padding))

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
    plt.figure(figsize=(16, 12))

    # Create custom colormap for terrain
    terrain_cmap = plt.cm.terrain

    # Plot terrain
    plt.imshow(terrain_window, cmap=terrain_cmap, alpha=0.7)

    # Plot obstacles with enhanced visibility
    if highlight_buildings:
        # Create a custom colormap for buildings
        building_colors = [(0, 0, 0, 0), (1, 1, 1, 1)]  # From transparent to white
        building_cmap = LinearSegmentedColormap.from_list("buildings", building_colors)

        # Use high contrast for obstacles
        plt.imshow(obstacles_window, cmap=building_cmap, alpha=0.8, vmin=0, vmax=1)
    else:
        obstacles_vis = np.ma.masked_where(obstacles_window == 0, obstacles_window)
        plt.imshow(obstacles_vis, cmap='binary', alpha=0.7, vmin=0, vmax=1)

    # Define vibrant colors that stand out against the background
    path_colors = [
        'red',  # Best path
        'blue',  # Alt 1
        'lime',  # Alt 2
        'magenta',  # Alt 3
        'cyan',  # Alt 4
        'orange',  # Alt 5
        'yellow',  # Alt 6
        'purple',  # Alt 7
        'deeppink',  # Alt 8
        'greenyellow'  # Alt 9
    ]

    # Prepare for legend
    legend_elements = []

    # Plot the paths in reverse order so the best path is on top
    for i in range(len(window_paths) - 1, -1, -1):
        window_path = window_paths[i]
        path, energy, name = paths[i]

        valid_points = [(r, c) for r, c in window_path
                        if 0 <= r < terrain_window.shape[0] and 0 <= c < terrain_window.shape[1]]

        if valid_points:
            path_rows, path_cols = zip(*valid_points)

            # Highlight the most efficient path
            if i == min_energy_index:
                line, = plt.plot(path_cols, path_rows, linewidth=6, color=path_colors[0],
                                 solid_capstyle='round', zorder=20)
                legend_elements.append((line, f"{name} - Energy: {energy:.1f} (BEST)"))
            else:
                # Use different color for each path with increased visibility
                color_index = min(i, len(path_colors) - 1)
                line, = plt.plot(path_cols, path_rows, linewidth=4, linestyle='--',
                                 color=path_colors[color_index], alpha=0.9,
                                 solid_capstyle='round', zorder=10)
                legend_elements.append((line, f"{name} - Energy: {energy:.1f}"))

    # Mark start and end points - using the first path's endpoints
    if window_paths and window_paths[0]:
        first_path = window_paths[0]
        if len(first_path) >= 2:
            start_point = first_path[0]
            end_point = first_path[-1]

            # Make start and end markers very visible
            start_marker = plt.plot(start_point[1], start_point[0], 'o',
                                    color='lime', markersize=16, markeredgecolor='black',
                                    markeredgewidth=2.0, zorder=30)[0]
            end_marker = plt.plot(end_point[1], end_point[0], 'o',
                                  color='red', markersize=16, markeredgecolor='black',
                                  markeredgewidth=2.0, zorder=30)[0]

            legend_elements.append((start_marker, 'Start'))
            legend_elements.append((end_marker, 'End'))

    # Set title
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title('Zoomed Drone Flight Paths', fontsize=16, pad=20)

    # Add descriptive caption
    plt.figtext(0.5, 0.01,
                "White shapes represent buildings and obstacles. "
                "The red solid line shows the most energy-efficient path. "
                "Dashed lines show alternative paths.",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add north arrow
    arrow_pos = (0.97, 0.97)
    plt.annotate('N', xy=arrow_pos, xycoords='axes fraction',
                 fontsize=14, fontweight='bold', ha='center', va='center')
    plt.annotate('', xy=(arrow_pos[0], arrow_pos[1] - 0.07), xycoords='axes fraction',
                 xytext=arrow_pos, textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=10),
                 ha='center', va='center')

    # Create a cleaner legend - put it on the right side outside the plot
    lines, labels = zip(*legend_elements)
    plt.legend(lines, labels, loc='upper left', fontsize=12,
               bbox_to_anchor=(1.01, 1), borderaxespad=0.5,
               frameon=True, framealpha=0.9, fancybox=True, shadow=True)

    plt.tight_layout()

    # Add colorbar for elevation
    cbar = plt.colorbar(label='Elevation (m)', pad=0.01)
    cbar.ax.tick_params(labelsize=10)

    # Save or show the visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'zoomed_paths_{timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"Saved zoomed visualization to: {os.path.join(output_dir, f'zoomed_paths_{timestamp}.png')}")
        plt.close()
    else:
        plt.show()