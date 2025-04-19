"""
Edinburgh landmarks with their coordinates.
"""

# Dictionary of Edinburgh landmarks with their coordinates
# These are approximate coordinates in the same projection as your terrain data
# Coordinates are in the form (easting, northing) for the British National Grid

EDINBURGH_LANDMARKS = {
    "National Galleries of Scotland": (325750, 673900),  # Approximate coordinates
    "Arthur's Seat": (327600, 673000),  # Approximate coordinates
    "Edinburgh Castle": (325150, 673500),  # Approximate coordinates
    "Holyrood Palace": (327000, 673900),  # Approximate coordinates
    "Calton Hill": (326400, 674200),  # Approximate coordinates
    "Royal Mile (St Giles Cathedral)": (325700, 673600),  # Approximate coordinates
    "Princes Street Gardens": (325400, 673800),  # Approximate coordinates
    "Waverley Station": (325800, 673800),  # Approximate coordinates
    "University of Edinburgh": (326200, 673200),  # Approximate coordinates
    "The Meadows": (325700, 672500),  # Approximate coordinates
}


def get_landmark_coordinates(landmark_name):
    """
    Get coordinates for a landmark by name.

    Args:
        landmark_name: Name of the landmark

    Returns:
        (x, y) tuple of coordinates or None if not found
    """
    return EDINBURGH_LANDMARKS.get(landmark_name)


def list_landmarks():
    """List all available landmarks."""
    return list(EDINBURGH_LANDMARKS.keys())