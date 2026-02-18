import numpy as np
from src.homography.transform_utils import compute_homography, apply_homography


class FieldMapper:
    """
    Handles homography between camera view and football field model.
    """

    def __init__(self):
        self.H = None  # Homography matrix

    def set_correspondences(self, image_points, field_points):
        """
        Compute homography matrix from image to field.

        Parameters:
        -----------
        image_points : list of (x, y)
            Points from video frame (pixel coordinates)

        field_points : list of (X, Y)
            Corresponding real-world field coordinates (meters)
        """

        if len(image_points) < 4 or len(field_points) < 4:
            raise ValueError("At least 4 point correspondences are required.")

        self.H = compute_homography(image_points, field_points)
        print("✅ Homography matrix computed successfully.")

    def map_bbox_to_field(self, bbox):
        """
        Map player bounding box to field coordinate.

        Parameters:
        -----------
        bbox : [x1, y1, x2, y2]

        Returns:
        --------
        (field_x, field_y) in meters
        """

        if self.H is None:
            raise ValueError("Homography matrix not initialized. Call set_correspondences() first.")

        x1, y1, x2, y2 = bbox

        # Use bottom-center (player feet)
        px = (x1 + x2) / 2.0
        py = y2

        field_x, field_y = apply_homography(self.H, (px, py))

        return float(field_x), float(field_y)

    def map_multiple_players(self, bboxes):
        """
        Map multiple bounding boxes to field coordinates.

        Parameters:
        -----------
        bboxes : list of [x1, y1, x2, y2]

        Returns:
        --------
        list of (field_x, field_y)
        """

        mapped_positions = []

        for bbox in bboxes:
            field_pos = self.map_bbox_to_field(bbox)
            mapped_positions.append(field_pos)

        return mapped_positions
