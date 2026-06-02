import cv2
import numpy as np


def frame2bev(x, y, _frame, bev_matrix, vertical, horizon):
    """
    Convert a point in the original camera frame to real-world BEV coordinates.

    Parameters
    ----------
    x : float
        x pixel coordinate in original frame
    y : float
        y pixel coordinate in original frame
    frame : ndarray
        Original camera frame (unused in computation, kept for API consistency)
    bev_matrix : ndarray
        3x3 perspective transform matrix from camera view to BEV
    vertical : float
        y-scale: real-world units per BEV pixel (cm/pixel)
    horizon : float
        x-scale: real-world units per BEV pixel (cm/pixel)

    Returns
    -------
    real_x : float
        x coordinate in real-world units (cm)
    real_y : float
        y coordinate in real-world units (cm)
    """
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, bev_matrix)
    bev_x = float(transformed[0, 0, 0])
    bev_y = float(transformed[0, 0, 1])
    return bev_x * horizon, bev_y * vertical
