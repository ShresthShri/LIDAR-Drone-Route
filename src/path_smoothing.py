"""
Path smoothing algorithms for drone flight paths.
"""
import numpy as np
from scipy.interpolate import CubicSpline
import rasterio

from src.utils import world_to_pixel, pixel_to_world, bresenham_line


def smooth_drone_path(path, terrain, obstacles, transform, max_slope_deg=30):
    """
    Smooth a drone path while maintaining obstacle avoidance and terrain constraints.

    Args:
        path: List of (x, y, z) coordinates from the A* algorithm
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform object for converting between coordinates
        max_slope_deg: Maximum allowed slope in degrees

    Returns:
        Smoothed path as a list of (x, y, z) coordinates
    """
    # First pass: Line-of-sight smoothing
    los_path = line_of_sight_smoothing(path, obstacles, transform)

    # Second pass: Spline smoothing with terrain constraints
    smooth_path = spline_smoothing_with_constraints(los_path, terrain, obstacles, transform, max_slope_deg)

    return smooth_path


def line_of_sight_smoothing(path, obstacles, transform):
    """
    Remove unnecessary waypoints where there's a clear line of sight.

    Args:
        path: List of (x, y, z) coordinates
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform for coordinate conversion

    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    current = 0

    while current < len(path) - 1:
        # Look ahead as far as possible
        for i in range(len(path) - 1, current, -1):
            # Check if there's a clear path
            p1 = (path[current][0], path[current][1])
            p2 = (path[i][0], path[i][1])

            if has_clear_path(p1, p2, obstacles, transform):
                smoothed.append(path[i])
                current = i
                break

    return smoothed


def has_clear_path(point1, point2, obstacles, transform):
    """
    Check if there's a clear path between two points.

    Args:
        point1: (x, y) tuple of world coordinates
        point2: (x, y) tuple of world coordinates
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform for coordinate conversion

    Returns:
        True if the path is clear, False if it intersects any obstacles
    """
    # Convert world coordinates to raster indices
    row1, col1 = world_to_pixel(point1[0], point1[1], transform)
    row2, col2 = world_to_pixel(point2[0], point2[1], transform)

    # Use Bresenham's line algorithm to check cells along the path
    cells = bresenham_line(row1, col1, row2, col2)

    for row, col in cells:
        # Check if the cell is within the bounds
        if (0 <= row < obstacles.shape[0] and
                0 <= col < obstacles.shape[1] and
                obstacles[row, col] == 1):
            return False

    return True


def spline_smoothing_with_constraints(path, terrain, obstacles, transform, max_slope_deg):
    """
    Apply spline smoothing with terrain and obstacle constraints.

    Args:
        path: List of (x, y, z) coordinates from the line-of-sight smoothing
        terrain: 2D numpy array of terrain elevation data
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        transform: Rasterio transform object for coordinate conversion
        max_slope_deg: Maximum allowed slope in degrees

    Returns:
        Smoothed path as a list of (x, y, z) coordinates
    """
    if len(path) <= 2:
        return path

    # Extract coordinates
    coords = np.array([(p[0], p[1]) for p in path])
    x = coords[:, 0]
    y = coords[:, 1]

    # Parameter along the path (cumulative distance)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    # Prevent zero distances which would cause issues with the spline
    dists = np.maximum(dists, 0.1)

    t = np.zeros(len(path))
    t[1:] = np.cumsum(dists)

    # Create splines
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)

    # Generate smooth path with more points
    num_points = max(len(path) * 3, 30)  # At least 30 points, or 3x the original
    t_fine = np.linspace(0, t[-1], num_points)
    smooth_x = spline_x(t_fine)
    smooth_y = spline_y(t_fine)

    # Initialize smoothed path
    smooth_path = []

    # Process points and validate them
    for i in range(len(smooth_x)):
        p = (smooth_x[i], smooth_y[i])

        # Skip if point is in an obstacle
        row, col = world_to_pixel(p[0], p[1], transform)
        if (0 <= row < obstacles.shape[0] and
                0 <= col < obstacles.shape[1] and
                obstacles[row, col] == 1):
            continue

        # Get elevation at this point
        elev = get_elevation_at_point(p, terrain, transform)

        # Check if we have a valid point
        if not np.isnan(elev) and add_with_slope_check(smooth_path, (p[0], p[1], elev), max_slope_deg):
            continue

    # Make sure we include the end point
    smooth_path.append(path[-1])

    # If we have too few points, fall back to the original path
    if len(smooth_path) < len(path) / 2:
        return path

    return smooth_path


def get_elevation_at_point(point, terrain, transform):
    """
    Get the elevation value at a specific world coordinate point.

    Args:
        point: (x, y) tuple of world coordinates
        terrain: 2D numpy array of terrain elevation data
        transform: Rasterio transform for coordinate conversion

    Returns:
        Elevation value or NaN if out of bounds
    """
    row, col = world_to_pixel(point[0], point[1], transform)

    if (0 <= row < terrain.shape[0] and 0 <= col < terrain.shape[1]):
        return terrain[row, col]
    else:
        return np.nan


def add_with_slope_check(path, new_point, max_slope_deg):
    """
    Add a point to the path if it doesn't exceed the maximum slope.

    Args:
        path: Current list of (x, y, z) points
        new_point: (x, y, z) tuple to potentially add
        max_slope_deg: Maximum allowed slope in degrees

    Returns:
        True if point was added, False if it was rejected
    """
    if not path:
        path.append(new_point)
        return True

    # Get the last point
    last_point = path[-1]

    # Calculate horizontal distance
    dx = new_point[0] - last_point[0]
    dy = new_point[1] - last_point[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Skip if points are too close
    if dist < 0.1:
        return False

    # Calculate elevation change
    dz = abs(new_point[2] - last_point[2])

    # Calculate slope
    slope_deg = np.degrees(np.arctan2(dz, dist))

    # Check if slope is acceptable
    if slope_deg <= max_slope_deg:
        path.append(new_point)
        return True

    return False


def bezier_smoothing(path, segments=10):
    """
    Apply Bézier curve smoothing to a path.

    Args:
        path: List of (x, y, z) coordinates
        segments: Number of segments to use for each Bézier curve

    Returns:
        Smoothed path as a list of (x, y, z) coordinates
    """
    if len(path) <= 2:
        return path

    # Define a function to calculate a point on a Bézier curve
    def bezier_point(points, t):
        if len(points) == 1:
            return points[0]

        new_points = []
        for i in range(len(points) - 1):
            x = points[i][0] * (1 - t) + points[i + 1][0] * t
            y = points[i][1] * (1 - t) + points[i + 1][1] * t
            z = points[i][2] * (1 - t) + points[i + 1][2] * t
            new_points.append((x, y, z))

        return bezier_point(new_points, t)

    smooth_path = []

    # Process the path in segments (e.g., 4 points at a time)
    i = 0
    while i < len(path) - 1:
        # Take a segment (up to 4 points)
        end_idx = min(i + 4, len(path))
        segment = path[i:end_idx]

        # Generate points along the Bézier curve
        for t in np.linspace(0, 1, segments):
            point = bezier_point(segment, t)
            smooth_path.append(point)

        i += 3  # Overlap segments slightly for continuity

    # Make sure the final point is included
    smooth_path.append(path[-1])

    return smooth_path


def potential_field_smoothing(path, obstacles, terrain, transform, iterations=10):
    """
    Smooth a path using potential field method.

    Args:
        path: List of (x, y, z) coordinates
        obstacles: 2D numpy array indicating obstacle locations (1 = obstacle)
        terrain: 2D numpy array of terrain elevation data
        transform: Rasterio transform for coordinate conversion
        iterations: Number of smoothing iterations

    Returns:
        Smoothed path as a list of (x, y, z) coordinates
    """
    if len(path) <= 2:
        return path

    smoothed_path = path.copy()

    # Parameters
    obstacle_weight = 1.0  # Repulsive force weight
    path_weight = 0.5  # Attractive force to original path
    smoothness_weight = 0.3  # Attractive force to neighbors
    obstacle_range = 10  # How far obstacles exert force

    for _ in range(iterations):
        new_path = [smoothed_path[0]]  # Keep start point fixed

        # For each interior point in the path
        for i in range(1, len(smoothed_path) - 1):
            current = smoothed_path[i]
            prev = smoothed_path[i - 1]
            next_point = smoothed_path[i + 1]

            # Calculate smoothness force (pulls toward midpoint of neighbors)
            smooth_x = (prev[0] + next_point[0]) / 2 - current[0]
            smooth_y = (prev[1] + next_point[1]) / 2 - current[1]

            # Calculate path force (pulls toward original path point)
            path_x = path[i][0] - current[0]
            path_y = path[i][1] - current[1]

            # Calculate obstacle forces (pushes away from obstacles)
            obstacle_x, obstacle_y = 0, 0

            row, col = world_to_pixel(current[0], current[1], transform)

            # Check surrounding area for obstacles
            for dr in range(-obstacle_range, obstacle_range + 1):
                for dc in range(-obstacle_range, obstacle_range + 1):
                    r, c = row + dr, col + dc

                    if (0 <= r < obstacles.shape[0] and 0 <= c < obstacles.shape[1] and obstacles[r, c] == 1):
                        # Calculate distance and direction
                        dist = max(1, np.sqrt(dr ** 2 + dc ** 2))
                        force = 1.0 / (dist ** 2)  # Force decreases with square of distance

                        # Add repulsive force
                        obstacle_x -= force * dr / dist
                        obstacle_y -= force * dc / dist

            # Update point position
            new_x = current[0] + (
                    smoothness_weight * smooth_x +
                    path_weight * path_x +
                    obstacle_weight * obstacle_x
            )
            new_y = current[1] + (
                    smoothness_weight * smooth_y +
                    path_weight * path_y +
                    obstacle_weight * obstacle_y
            )

            # Get elevation at new position
            new_z = get_elevation_at_point((new_x, new_y), terrain, transform)
            if np.isnan(new_z):
                new_z = current[2]  # Keep old elevation if new one is invalid

            new_path.append((new_x, new_y, new_z))

        new_path.append(smoothed_path[-1])  # Keep end point fixed
        smoothed_path = new_path

    return smoothed_path