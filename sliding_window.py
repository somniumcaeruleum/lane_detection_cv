import numpy as np
import cv2

def sliding_window_lane_detect(binary_warped, mwindows=6, nwindows=15, margin=60, minpix=100):
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
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

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
            left_lane_inds.append(good_left)
            if left_lane:
                point_left.append([leftx_current, (win_y_low + win_y_high)//2])
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
                cv2.circle(out_img, (leftx_current, (win_y_low + win_y_high)//2), 10, (255, 0, 0), -1)
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
            right_lane_inds.append(good_right)
            if right_lane:
                point_right.append([rightx_current, (win_y_low + win_y_high)//2])
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
                cv2.circle(out_img, (rightx_current, (win_y_low + win_y_high)//2), 10, (0, 0, 255), -1)
        else:
            right = False
        
    # attach histogram to out_img
    h_img = np.zeros_like(out_img)
    for i in range(len(histogram)):
        color = (255, 255, 255) if histogram[i] < peak_threshold else (0, 0, 255)
        cv2.circle(h_img, (i, binary_warped.shape[0]-histogram[i]), 1, color, -1)
    out_img_resized = cv2.resize(out_img, (400, 300))
    h_img_resized = cv2.resize(h_img, (400, 300))
    boundary_img = np.ones(shape=(300, 10, 3), dtype=np.uint8) * 255
    out_img = np.hstack((out_img_resized, boundary_img, h_img_resized))

    left_fit, right_fit = [], []
    leftx, lefty = np.array([]), np.array([])
    rightx, righty = np.array([]), np.array([])

    if left_lane_inds:
        left_lane_inds  = np.concatenate(left_lane_inds)
        leftx,  lefty  = nonzerox[left_lane_inds],  nonzeroy[left_lane_inds]
        if len(lefty)>3 and len(leftx)>3:
            left_fit  = np.polyfit(lefty[1:],  leftx[1:],  2)
    
    if right_lane_inds:
        right_lane_inds = np.concatenate(right_lane_inds)
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        if len(righty)>3 and len(rightx)>3:
            right_fit = np.polyfit(righty[1:], rightx[1:], 2)

    return point_left, point_right, left_lane, right_lane, out_img

def fit_lanes(frame, point_left, point_right, ldim = 2, rdim = 2):
    left_fit, right_fit = [], []

    if len(point_left) == 2:
        ldim = 1
    if len(point_right) == 2:
        rdim = 1

    if len(point_left)>ldim:
        left_fit = np.polyfit([y for x, y in point_left], [x for x, y in point_left], ldim)
    if len(point_right)>rdim:
        right_fit = np.polyfit([y for x, y in point_right], [x for x, y in point_right], rdim)

    left_fitx, right_fitx = [], []

    ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    if len(left_fit):
        left_fitx = left_fit[0]*ploty**(ldim-0)
        for i in range(1, ldim+1):
            left_fitx += left_fit[i]*ploty**(ldim-i)

    if len(right_fit):
        right_fitx = right_fit[0]*ploty**(rdim-0)
        for i in range(1, rdim+1):
            right_fitx += right_fit[i]*ploty**(rdim-i)

    return left_fitx, right_fitx, ploty, left_fit, right_fit