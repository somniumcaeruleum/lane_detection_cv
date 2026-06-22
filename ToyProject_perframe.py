from matplotlib import pyplot as plt
import numpy as np
import cv2

# thresholded -> sliding_window_lane_detect -> plot on reality -> path planning
class ToyProject:
    def __init__(self, pts_src=np.array([[190, 240],[435, 240],[540, 430],[10, 450]], dtype=np.float32), debug=False):
        # vertical distance, horizontal distance per pixel
        self.vertical = 0.376 # y-scale (cm/pixel)
        self.horizon = 0.159  # x-scale (cm/pixel)

        # horizontal distance between left, right lane (pixel)
        self.dist = (85/self.horizon)//2
        self.dist_real = 85//2

        # global vars
        self.debug = debug
        self.pts_src = pts_src

    def launch(self, frame):
        frame_height, frame_width = frame.shape[:2]
        # Define the Region of Interest (ROI) points
        pts_src = self.pts_src

        # Define destination points matching the clockwise order
        pts_dst = pts_dst=np.array([[0, 0],[frame_width, 0],[frame_width, frame_height],[0, frame_height]], dtype=np.float32)

        self.bev_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        pre_left_fitx, pre_left_ploty = [], []
        pre_right_fitx, pre_right_ploty = [], []
        pre_central_fitx, pre_central_ploty = [], []
        pre_steering_angle, pre_lookahead_pt, pre_heading_error = 0, (0, -170), 0
            
        ### BEV
        # perspective transform
        bev_img = raw_to_bev(frame, frame_width, frame_height, self.bev_matrix)
        gray_thresholded, color_thresholded = thresholding(bev_img)

        ### Sliding Window
        # coefficients of 2nd order polynomials / debug: Stage 2
        point_left, point_right, left_lane, right_lane, out_img = sliding_window_lane_detect(gray_thresholded)

        left_fitx, left_ploty = [], []
        right_fitx, right_ploty = [], []
        central_fitx, central_ploty = [], []

        # trust right lane
        if len(point_left) <= len(point_right):
            right_fitx, right_ploty, right_fit = fit_lane(frame.shape[0], point_right)
            if len(right_fitx):
                central_fitx, central_ploty = offset_curve(right_fitx, right_ploty, right_fit, self.dist, side='left')
                left_fitx, left_ploty = offset_curve(central_fitx, central_ploty, right_fit, self.dist, side='left')

        # trust left lane
        else:
            left_fitx, left_ploty, left_fit = fit_lane(frame.shape[0], point_left)
            if len(left_fitx):
                central_fitx, central_ploty = offset_curve(left_fitx, left_ploty, left_fit, self.dist, side='right')
                right_fitx, right_ploty = offset_curve(central_fitx, central_ploty, left_fit, self.dist, side='right')

                    # real world scaling
        if len(left_fitx):
            left_fitx = left_fitx*self.horizon
        if len(right_fitx):
            right_fitx = right_fitx*self.horizon
        if len(central_fitx):
            central_fitx = central_fitx*self.horizon

        if len(left_ploty) == 0:
            left_ploty = np.linspace(0, frame.shape[0], 50) #start, end, # of points
        if len(right_ploty) == 0:
            right_ploty = np.linspace(0, frame.shape[0], 50) #start, end, # of points
        if len(central_ploty) == 0:
            central_ploty = np.linspace(0, frame.shape[0], 50) #start, end, # of points

        left_ploty = left_ploty*self.vertical
        right_ploty = right_ploty*self.vertical
        central_ploty = central_ploty*self.vertical

        if len(point_left):
            for i in range(len(point_left)):
                point_left[i] = [point_left[i][0]*self.horizon, point_left[i][1]*self.vertical]
        if len(point_right):
            for i in range(len(point_right)):
                point_right[i] = [point_right[i][0]*self.horizon, point_right[i][1]*self.vertical]

        ### Control: Pure Pursuit
        vehicle_x, vehicle_y  = frame2bev(frame_width/2, frame_height, frame, bev_matrix=self.bev_matrix, vertical=self.vertical, horizon=self.horizon)
        vehicle_y += 90

        # front center of car -> 0, 0
        # vehicle_x, vehicle_y -> 0, 90
        if len(left_fitx):
            left_fitx, left_ploty = left_fitx - vehicle_x, left_ploty - vehicle_y+90
        if len(right_fitx):
            right_fitx, right_ploty = right_fitx - vehicle_x, right_ploty - vehicle_y+90
        if len(central_fitx):
            central_fitx, central_ploty = central_fitx - vehicle_x, central_ploty - vehicle_y+90

        if len(point_left):
            for i in range(len(point_left)):
                point_left[i] = [point_left[i][0]-vehicle_x, point_left[i][1] - vehicle_y+90]
        if len(point_right):
            for i in range(len(point_right)):
                point_right[i] = [point_right[i][0]-vehicle_x, point_right[i][1] - vehicle_y+90]

        vehicle_x, vehicle_y = 0, 90

        # y = -y
        if len(point_left):
            for i in range(len(point_left)):
                point_left[i][1] = -point_left[i][1]
        if len(point_right):
            for i in range(len(point_right)):
                point_right[i][1] = -point_right[i][1]

        left_ploty = -left_ploty
        central_ploty = -central_ploty
        right_ploty = -right_ploty

        vehicle_y = -vehicle_y
        '''
        Returns
        -------
        steering_angle : float
            Steering angle in radians. Positive = right turn, negative = left turn.
        lookahead_pt : tuple or None
            (x, y) of the selected lookahead point, or None if path is empty.
        heading_error : float
            Heading error in radians.
        '''
        steering_angle, lookahead_pt, heading_error = pure_pursuit(central_fitx, central_ploty, vehicle_x, vehicle_y=vehicle_y, lookahead=170.0, wheelbase=55.0)

        # handle undetected situation
        if lookahead_pt==None:
            left_fitx, left_ploty = pre_left_fitx, pre_left_ploty
            right_fitx, right_ploty = pre_right_fitx, pre_right_ploty
            central_fitx, central_ploty = pre_central_fitx, pre_central_ploty
            steering_angle, lookahead_pt, heading_error = pre_steering_angle, pre_lookahead_pt, pre_heading_error
            print('LP Point cannot be calculated. Load previous one.')
        else:
            pre_left_fitx, pre_left_ploty = left_fitx, left_ploty
            pre_right_fitx, pre_right_ploty = right_fitx, right_ploty
            pre_central_fitx, pre_central_ploty = central_fitx, central_ploty
            pre_steering_angle, pre_lookahead_pt, pre_heading_error = steering_angle, lookahead_pt, heading_error
            print('LP Point can be calculated.')
        print(f'Steering Angle={np.degrees(steering_angle):.1f} (deg)  Heading Error={np.degrees(heading_error):.1f} (deg)')
        print()

# Sliding Window
def sliding_window_lane_detect(binary_warped, mwindows=15, nwindows=15, margin=60, minpix=100):
    """
    Detect lane pixels using sliding windows.
    
    Args:
        binary_warped: bird's-eye-view binary image (thresholded)
        nwindows: number of sliding windows
        margin: width of windows +/- margin
        minpix: min pixels to recenter window
    
    Returns:
        left_fit, right_fit: 2nd-order polynomial coefficients
        out_img: visualization
    """
    # 1. Histogram of bottom half to find starting x positions
    histogram = np.sum(binary_warped[binary_warped.shape[0]-50:, :]//254, axis=0)
    midpoint = histogram.shape[0] // 2
    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = midpoint-np.argmax(histogram[midpoint:][::-1]) + midpoint

    leftx_peak = np.max(histogram[:midpoint])
    rightx_peak = np.max(histogram[midpoint:])

    peak_threshold = 21
    left_lane, right_lane = leftx_peak > peak_threshold, rightx_peak > peak_threshold

    # 2. Setup
    lwindow_height = binary_warped.shape[0] // mwindows
    rwindow_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    point_left, point_right = [], []

    left, right = True, True
    # 3. Slide windows from bottom to top
    # Left window
    for window in range(mwindows):
        win_y_low  = binary_warped.shape[0] - (window + 1) * lwindow_height
        win_y_high = binary_warped.shape[0] - window * lwindow_height
        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin

        # Identify nonzero pixels inside window
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        # Recenter next window if enough pixels found
        if len(good_left) > minpix and left:
            leftx_current = int(np.mean(nonzerox[good_left]))
            if left_lane:
                point_left.append([leftx_current, (win_y_low + win_y_high)//2])
                #cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
                #cv2.circle(out_img, (leftx_current, (win_y_low + win_y_high)//2), 10, (255, 0, 0), -1)
        else:
            left = False

    # Right window
    for window in range(nwindows):
        win_y_low  = binary_warped.shape[0] - (window + 1) * rwindow_height
        win_y_high = binary_warped.shape[0] - window * rwindow_height
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify nonzero pixels inside window
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Recenter next window if enough pixels found
        if len(good_right) > minpix and right:
            rightx_current = int(np.mean(nonzerox[good_right]))
            if right_lane:
                point_right.append([rightx_current, (win_y_low + win_y_high)//2])
                #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
                #cv2.circle(out_img, (rightx_current, (win_y_low + win_y_high)//2), 10, (0, 0, 255), -1)
        else:
            right = False

    left_fit, right_fit = [], []
    leftx, lefty = np.array([]), np.array([])
    rightx, righty = np.array([]), np.array([])

    return point_left, point_right, left_lane, right_lane, binary_warped

def fit_lane(height, points, dim=2):
    coef = []

    if len(points) == 2:
        dim = 1

    if len(points)>dim:
        coef = np.polyfit([y for x, y in points], [x for x, y in points], dim)

    fitx = []

    ploty = np.linspace(-height//3, height, 50) # start, end, # of points
    if len(coef):
        fitx = coef[0]*ploty**(dim-0)
        for i in range(1, dim+1):
            fitx += coef[i]*ploty**(dim-i)

    return fitx, ploty, coef

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

def raw_to_bev(frame, frame_width, frame_height, matrix):
    bev_frame = cv2.warpPerspective(frame, matrix, dsize=(frame_width, frame_height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return bev_frame

def gaussian_denoise(image, ksize=5, sigma=0):
    """
    Remove noise from an image using a Gaussian filter.
    
    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color).
    ksize : int or tuple
        Kernel size. Must be a positive odd integer (e.g., 3, 5, 7, 9).
        Larger = more smoothing but more blur. Default 5.
    sigma : float
        Standard deviation of the Gaussian. If 0, OpenCV computes it
        from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
    
    Returns
    -------
    denoised : ndarray
        Smoothed image.
    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma, sigmaY=sigma)

def thresholding(frame, threshold = 240):
    # frame: BGR img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = gaussian_denoise(gray, ksize=23, sigma=0)
    _, gray_thresholded = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY)
    color_thresholded = cv2.cvtColor(gray_thresholded, cv2.COLOR_GRAY2BGR)
    return gray_thresholded, color_thresholded

def offset_curve(x, y, coeffs, d, side='left'):
        """
        Parameters
        ----------
        coeffs : array-like *** decreasing order of power ***
            coefficients of y-x graph
        y, x : array-like
            Coordinates of the original curve (ordered points).
        d : float
            Offset distance. side='left' offsets to the left of travel direction.
        side : str
            'left' or 'right' relative to curve direction.

        Returns
        -------
        xo, yo : ndarray
            Coordinates of the offset curve, same length as the input.
        """

        coeffs_d = []
        for i in range(len(coeffs)-1):
            coeffs_d.append(coeffs[i]*(len(coeffs)-1-i))
        derivative_coeffs = np.array(coeffs_d)
        
        dxdy = np.polyval(derivative_coeffs, y)  # dx/dy
        norm = np.sqrt(1.0 + dxdy**2)

        # left unit normal: rotate tangent (dxdy, 1) by +90° → (-1, dxdy)
        nx = -1.0 / norm
        ny = dxdy / norm

        sign = 1 if side == 'left' else -1
        x0 = x + sign * d * nx
        y0 = y + sign * d * ny
        
        return x0, y0