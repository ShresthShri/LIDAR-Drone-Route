"""
Utility functions for the drone path planning project.
"""
import numpy as np
import rasterio


def world_to_pixel(x, y, transform):
    """
    Convert world coordinates to pixel/array coordinates using the transform.

    Args:
        x: X coordinate in world space
        y: Y coordinate in world space
        transform: Rasterio transform object

    Returns:
        (row, col) tuple of pixel coordinates
    """
    col, row = ~transform * (x, y)
    return int(row), int(col)


def pixel_to_world(row, col, transform):
    """
    Convert pixel coordinates to world coordinates using the transform.

    Args:
        row: Row in pixel space
        col: Column in pixel space
        transform: Rasterio transform object

    Returns:
        (x, y) tuple of world coordinates
    """
    x, y = transform * (col, row)
    return x, y


def create_downsampled_transform(original_transform, downsample_factor):
    """
    Create a transform for downsampled data.

    Args:
        original_transform: Original rasterio transform
        downsample_factor: Factor by which to downsample

    Returns:
        New transform for downsampled data
    """
    return rasterio.transform.from_origin(
        original_transform.c,
        original_transform.f,
        original_transform.a * downsample_factor,
        abs(original_transform.e) * downsample_factor
    )


def calculate_path_energy(path, energy_factor_horizontal=1.0, energy_factor_vertical=2.0):
    """
    Calculate the total energy required for a path.

    Args:
        path: List of (x, y, z) coordinates
        energy_factor_horizontal: Energy multiplier for horizontal movement
        energy_factor_vertical: Energy multiplier for vertical climbing

    Returns:
        Total energy consumption
    """
    if len(path) < 2:
        return 0

    total_energy = 0

    for i in range(1, len(path)):
        prev_x, prev_y, prev_z = path[i - 1]
        x, y, z = path[i]

        # Calculate horizontal distance
        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

        # Calculate vertical movement (only penalize climbing, not descending)
        height_diff = max(0, z - prev_z)

        # Calculate segment energy
        segment_energy = (
                distance * energy_factor_horizontal +
                height_diff * energy_factor_vertical
        )

        total_energy += segment_energy

    return total_energy


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm to get all cells along a line.

    Args:
        x0, y0: Start point coordinates
        x1, y1: End point coordinates

    Returns:
        List of (x, y) tuples representing cells along the line
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells