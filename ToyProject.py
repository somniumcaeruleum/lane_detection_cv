from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

from debug import show_curves

from bev import raw_to_bev
from bev import thresholding

from sliding_window import sliding_window_lane_detect
from sliding_window import fit_lanes

# thresholded -> sliding_window_lane_detect -> plot on reality -> path planning
class ToyProject:
    def __init__(self, input_path, output_path="", debug=False, video_save=False):
        # vertical distance, horizontal distance per pixel
        self.vertice = 0.208 # y-scale
        self.horizon = 0.190 # x-scale

        #horizontal distance between left, right lane (pixel)
        self.dist = (85/self.horizon)//2
        self.dist_real = 85//2

        # global vars
        self.input_path = input_path
        self.output_path = output_path
        self.debug = debug
        self.video_save = video_save

    def launch(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define the Region of Interest (ROI) points
        pts_src = np.array([
            [190, 240],
            [435, 240],
            [540, 430],
            [10, 450]
        ], dtype=np.float32)

        # Define destination points matching the clockwise order
        pts_dst = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, frame_height],
            [0, frame_height]
        ], dtype=np.float32)

        self.bev_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        fig_debug3, ax_debug3 = None, None
        if self.debug == 3:
            fig_debug3, ax_debug3 = plt.subplots(figsize=(10, 6))

        output = None
        if self.video_save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (frame_width, frame_height)
            if self.debug == 1:
                resolution = (800, 300)
            elif self.debug == 2:
                resolution = (810, 300)
            elif self.debug == 3:
                fig_debug3.canvas.draw()
                resolution = fig_debug3.canvas.get_width_height()
            output = cv2.VideoWriter(self.output_path, fourcc, 24.0, resolution)
        
        left_lane, right_lane = True, True
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video.")
                break # End of video
            
            ### BEV
            # perspective transform
            bev_img = raw_to_bev(frame, frame_width, frame_height, self.bev_matrix)
            gray_thresholded, color_thresholded = thresholding(bev_img)

            # debug: Stage 1
            if self.debug == 1:
                # Display the original and the BEV side-by-side (optional, for visualization)
                # Note: we resize them to fit on smaller screens
                display_TL = cv2.resize(bev_img, (400, 300))
                display_TR = cv2.resize(color_thresholded, (400, 300))
                combined_display = np.hstack((display_TL, display_TR))

                cv2.imshow("BEV vs Thresholded", combined_display)
                if self.video_save:
                    output.write(combined_display)

                # Wait 25ms and check if the 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

            ### Sliding Window
            # coefficients of 2nd order polynomials / debug: Stage 2
            point_left, point_right, left_lane, right_lane, out_img = sliding_window_lane_detect(gray_thresholded)

            # debug: Stage 2
            if self.debug == 2 or self.debug == 3:
                cv2.imshow('debug: Stage 2', out_img)
                if self.video_save and self.debug==2:
                    output.write(out_img)

                # Wait 25ms and check if the 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

            # draw lanes
            left_fitx, right_fitx, ploty, _, __= fit_lanes(color_thresholded, point_left, point_right)

            # central line & handle undetected lane
            central_fitx = []
            if left_lane == False and right_lane:
                # left_fitx = right_fitx-self.dist
                left_fitx, _ = self.offset_curve(right_fitx, ploty, 2 * self.dist, side='left')
            if right_lane == False and left_lane:
                # right_fitx = left_fitx+self.dist
                right_fitx, _ = self.offset_curve(left_fitx, ploty, 2 * self.dist, side='right')
            if len(right_fitx):
                central_fitx, _ = self.offset_curve(right_fitx, ploty, self.dist, side='left')

            # real world scaling
            if len(left_fitx):
                left_fitx = left_fitx*self.horizon
            if len(right_fitx):
                right_fitx = right_fitx*self.horizon
            if len(central_fitx):
                central_fitx = central_fitx*self.horizon
            ploty = ploty*self.vertice

            if len(point_left):
                for i in range(len(point_left)):
                    point_left[i] = point_left[i][0]*self.horizon, point_left[i][1]*self.vertice
            if len(point_right):
                for i in range(len(point_right)):
                    point_right[i] = point_right[i][0]*self.horizon, point_right[i][1]*self.vertice

            ### debug: Stage 3
            if self.debug == 3:
                ax_debug3.cla()
                ploty_3 = frame_height*self.vertice-ploty
                show_curves((left_fitx, ploty_3), (right_fitx, ploty_3), (central_fitx, ploty_3),
                            labels=['left', 'right', 'central'], colors=['blue', 'red', 'green'],
                            styles=['-', '-', '-'], title='debug: Stage 3', ax=ax_debug3)

                if len(point_left):
                    px = [x for x, _ in point_left]
                    py = [frame_height*self.vertice - y for _, y in point_left]
                    ax_debug3.scatter(px, py, c='blue', s=30, zorder=5)

                if len(point_right):
                    px = [x for x, _ in point_right]
                    py = [frame_height*self.vertice - y for _, y in point_right]
                    ax_debug3.scatter(px, py, c='red', s=30, zorder=5)

                ax_debug3.set_xlim(0, 120)
                ax_debug3.set_ylim(0, 110)

                fig_debug3.canvas.draw()
                img_bgr = cv2.cvtColor(np.asarray(fig_debug3.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

                cv2.imshow('debug: Stage 3', img_bgr)
                if self.video_save:
                    output.write(img_bgr)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

        if self.video_save:
            output.release()
        cap.release()
        cv2.destroyAllWindows()
        if fig_debug3 is not None:
            plt.close(fig_debug3)
    
    def offset_curve(self, x, y, d, side='left', smooth=True):
        """
        Compute an offset curve at perpendicular distance d from the input curve.
        
        Parameters
        ----------
        x, y : array-like
            Coordinates of the original curve (ordered points).
        d : float
            Offset distance. The minimum distance between curves equals |d|.
        side : str
            'left' or 'right' relative to curve direction.
        smooth : bool
            If True, fits a spline for smoother tangent estimation.
        
        Returns
        -------
        xo, yo : ndarray
            Coordinates of the offset curve.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 2:
            return np.array([]), np.array([])

        if smooth and len(x) >= 4:
            # Fit a parametric spline for smooth derivatives
            tck, u = splprep([x, y], s=0)
            u_fine = np.linspace(0, 1, max(len(x), 200))
            x, y = splev(u_fine, tck)
            dx, dy = splev(u_fine, tck, der=1)
        else:
            # Finite differences (central, with forward/backward at ends)
            dx = np.gradient(x)
            dy = np.gradient(y)
        
        # Unit normal vector (perpendicular to tangent)
        length = np.hypot(dx, dy)
        length[length == 0] = 1e-12  # avoid division by zero
        nx = -dy / length
        ny =  dx / length
        
        sign = 1 if side == 'left' else -1
        xo = x + sign * d * nx
        yo = y + sign * d * ny
        
        return xo, yo