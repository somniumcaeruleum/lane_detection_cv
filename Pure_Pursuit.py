import numpy as np

# vehicle_x = frame_width/2*self.horizon
# vehicle_y = frame_height*self.vertice + 90
def pure_pursuit(central_fitx, ploty, vehicle_x, vehicle_y=None,
                 lookahead=170.0, wheelbase=55.0):
    """
    Pure pursuit lateral controller for the lane-following BEV pipeline.

    Coordinate convention (real-world units, after BEV pixel scaling):
      x: increases to the right
      y: increases toward the vehicle (bottom of BEV = largest y)
      vehicle heading: -y direction (up the image = forward)

    Parameters
    ----------
    central_fitx : array-like
        X-coordinates of the center path (cm). Index 0 = farthest, -1 = nearest.
    ploty : array-like
        Y-coordinates of the center path (cm). Same ordering as central_fitx.
    vehicle_x : float
        Lateral vehicle position (cm). Typically frame_width/2 * horizon.
    vehicle_y : float, optional
        Longitudinal vehicle position (cm). Defaults to max(ploty).
    lookahead : float
        Lookahead distance (cm).
    wheelbase : float
        Vehicle wheelbase (cm).

    Returns
    -------
    steering_angle : float
        Steering angle in radians. Negative = right turn, Positive = left turn.
    lookahead_pt : tuple or None
        (x, y) of the selected lookahead point, or None if path is empty.
    heading_error : float
        Heading error in radians (difference between vehicle heading and path tangent).
    """
    if len(central_fitx) == 0 or len(ploty) == 0 or vehicle_y is None:
        return 0.0, None, 0.0

    path_x = np.asarray(central_fitx, dtype=float)
    path_y = np.asarray(ploty, dtype=float)

    # Offsets in vehicle frame: lateral (right+), forward (ahead+)
    lateral = path_x - vehicle_x
    forward = vehicle_y - path_y
    dist = np.hypot(lateral, forward)

    # Select lookahead point: farthest point whose distance is still >= lookahead,
    # searching from the near end. When the entire path lies within lookahead,
    # fall back to the farthest available point.
    beyond = dist >= lookahead
    idx = int(np.where(beyond)[0][-1]) if np.any(beyond) else 0

    ld = dist[idx]
    alpha = np.arctan2(lateral[idx], forward[idx])

    # Pure pursuit: delta = arctan(2 * L_wb * sin(alpha) / L_d)
    steering_angle = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)

    # Compute path tangent at the lookahead point
    if idx > 0 and idx < len(path_x) - 1:
        # Tangent vector: derivative of path
        dx_path = path_x[idx+1] - path_x[idx-1]
        dy_path = path_y[idx+1] - path_y[idx-1]
        desired_heading = np.arctan2(dx_path, -dy_path)  # Note: -dy because forward is -y
    else:
        # Fallback: use direction to lookahead point
        desired_heading = alpha

    # Vehicle heading is nominally 0 in vehicle frame
    vehicle_heading = 0.0

    # Heading error: how far vehicle is from desired orientation
    heading_error = desired_heading - vehicle_heading
    steering_angle = -steering_angle

    return steering_angle, (float(path_x[idx]), float(path_y[idx])), heading_error