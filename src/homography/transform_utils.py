import numpy as np
import cv2


def compute_homography(src_points, dst_points):
    """
    Compute homography matrix from source to destination.

    Parameters:
    -----------
    src_points : list of (x, y)
        Points in the image (pixel coordinates)

    dst_points : list of (X, Y)
        Corresponding real-world field coordinates

    Returns:
    --------
    H : 3x3 homography matrix
    """

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    if src.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required.")

    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC)

    if H is None:
        raise RuntimeError("Homography computation failed.")

    return H


def apply_homography(H, point):
    """
    Apply homography matrix to a single 2D point.

    Parameters:
    -----------
    H : 3x3 homography matrix
    point : (x, y)

    Returns:
    --------
    (X, Y) transformed coordinates
    """

    x, y = point

    point_h = np.array([x, y, 1.0], dtype=np.float32)

    mapped = H @ point_h

    if mapped[2] == 0:
        raise ZeroDivisionError("Invalid homography transformation.")

    mapped /= mapped[2]

    return float(mapped[0]), float(mapped[1])


def apply_homography_multiple(H, points):
    """
    Apply homography to multiple points.

    Parameters:
    -----------
    H : 3x3 homography matrix
    points : list of (x, y)

    Returns:
    --------
    list of (X, Y)
    """

    transformed_points = []

    for pt in points:
        transformed_points.append(apply_homography(H, pt))

    return transformed_points
