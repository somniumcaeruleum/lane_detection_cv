import cv2
import numpy as np

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