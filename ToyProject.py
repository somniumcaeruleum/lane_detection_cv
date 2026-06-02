from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

from debug import show_curves

from bev import raw_to_bev
from bev import thresholding

from sliding_window import sliding_window_lane_detect
from sliding_window import fit_lanes

from frame2bev import frame2bev

from Pure_Pursuit import pure_pursuit

# thresholded -> sliding_window_lane_detect -> plot on reality -> path planning
class ToyProject:
    def __init__(self, input_path, output_path="", debug=False, video_save=False):
        # vertical distance, horizontal distance per pixel
        self.vertical = 0.376 # y-scale (cm/pixel)
        self.horizon = 0.159  # x-scale (cm/pixel)

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
        if self.debug:
            fig_debug3, ax_debug3 = plt.subplots(figsize=(10, 6))

        output = None
        if self.video_save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (frame_width, frame_height)
            #TODO: output resolution
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
            if self.debug:
                cv2.polylines(frame, [pts_src.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                display_TL = cv2.resize(frame, (400, 300))
                display_TR = cv2.resize(color_thresholded, (400, 300))
                combined_display = np.hstack((display_TL, display_TR))

                cv2.imshow("Original vs Thresholded", combined_display)

                # Wait 25ms and check if the 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

            ### Sliding Window
            # coefficients of 2nd order polynomials / debug: Stage 2
            point_left, point_right, left_lane, right_lane, out_img = sliding_window_lane_detect(gray_thresholded)

            # debug: Stage 2
            if self.debug:
                cv2.imshow('Sliding Window', out_img)

                # Wait 25ms and check if the 'q' key is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

            # draw lanes
            left_fitx, right_fitx, ploty, left_fit, right_fit= fit_lanes(color_thresholded, point_left, point_right)

            # central line & handle undetected lane
            central_fitx = []
            if left_lane == False and right_lane and len(right_fitx):
                # left = right-self.dist
                left_fitx, _ = self.offset_curve(right_fitx, ploty, right_fit, 2 * self.dist, side='left')
            if right_lane == False and left_lane and len(left_fitx):
                # right = left+self.dist
                right_fitx, _ = self.offset_curve(left_fitx, ploty, left_fit, 2 * self.dist, side='right')
            if len(right_fitx):
                central_fitx, _ = self.offset_curve(right_fitx, ploty, right_fit, self.dist, side='left')

            # real world scaling
            if len(left_fitx):
                left_fitx = left_fitx*self.horizon
            if len(right_fitx):
                right_fitx = right_fitx*self.horizon
            if len(central_fitx):
                central_fitx = central_fitx*self.horizon
            ploty = ploty*self.vertical

            if len(point_left):
                for i in range(len(point_left)):
                    point_left[i] = point_left[i][0]*self.horizon, point_left[i][1]*self.vertical
            if len(point_right):
                for i in range(len(point_right)):
                    point_right[i] = point_right[i][0]*self.horizon, point_right[i][1]*self.vertical

            ### debug: Stage 3
            if self.debug:
                ax_debug3.cla()
                ploty_3 = frame_height*self.vertical-ploty
                show_curves((left_fitx, ploty_3), (right_fitx, ploty_3), (central_fitx, ploty_3),
                            labels=['left', 'right', 'central'], colors=['blue', 'red', 'green'],
                            styles=['-', '-', '-'], title='plotting', ax=ax_debug3)

                if len(point_left):
                    px = [x for x, _ in point_left]
                    py = [frame_height*self.vertical - y for _, y in point_left]
                    ax_debug3.scatter(px, py, c='blue', s=30, zorder=5)

                if len(point_right):
                    px = [x for x, _ in point_right]
                    py = [frame_height*self.vertical - y for _, y in point_right]
                    ax_debug3.scatter(px, py, c='red', s=30, zorder=5)

                ax_debug3.set_xlim(0, 125)
                ax_debug3.set_ylim(0, 200)

                fig_debug3.canvas.draw()
                img_bgr = cv2.cvtColor(np.asarray(fig_debug3.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            
            ### Control: Pure Pursuit
            vehicle_x, vehicle_y  = frame2bev(frame_width/2, frame_height+90/self.vertical, frame, bev_matrix=self.bev_matrix, vertical=self.vertical, horizon=self.horizon)
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
            steering_angle, lookahead_pt, heading_error = pure_pursuit(central_fitx, ploty, vehicle_x, vehicle_y=vehicle_y, lookahead=170.0, wheelbase=55.0)

            if self.debug:

                if lookahead_pt == None:
                    print("Lookahead Point cannot be calculated")
                else:
                    print("Lookahead Point can be calculated")
                    print(f"heading error (rad): {heading_error}, steering angle (rad): {steering_angle}")

                    plot_vy = frame_height * self.vertical - vehicle_y
                    circle = Circle((vehicle_x, plot_vy), 170.0,
                                    fill=False, edgecolor='orange', linewidth=1.5)
                    ax_debug3.add_patch(circle)
                    circle.set_clip_path(ax_debug3.patch)

                    if lookahead_pt is not None:
                        lx, ly = lookahead_pt
                        ax_debug3.scatter([lx], [frame_height * self.vertical - ly],
                                        c='purple', s=60, zorder=6, marker='x')

                    fig_debug3.canvas.draw()
                    img_bgr = cv2.cvtColor(np.asarray(fig_debug3.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                    cv2.imshow('Plotting', img_bgr)

                    # Wait 25ms and check if the 'q' key is pressed
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        print("Process interrupted by user.")
                        break

                print()
        
        if self.video_save:
            output.release()
        cap.release()
        cv2.destroyAllWindows()
        if fig_debug3 is not None:
            plt.close(fig_debug3)
    
    def offset_curve(self, x, y, coeffs, d, side='left', smooth=True):
        """
        Compute an offset curve at perpendicular distance d from the input curve.

        Every input point is preserved one-to-one in the output: each point is
        translated along its own unit normal by the offset distance d. No
        resampling or change in point count occurs.

        Parameters
        ----------
        coeffs : array-like *** decreasing order of power ***
            coefficients of y-x graph
        y : array-like
            Coordinates of the original curve (ordered points).
        d : float
            Offset distance. side='left' offsets to the left of travel direction.
        side : str
            'left' or 'right' relative to curve direction.
        smooth : bool
            If True, fits a parametric spline to estimate smooth tangents/normals
            at the original points (derivatives only — the points are NOT moved
            onto the spline).

        Returns
        -------
        xo, yo : ndarray
            Coordinates of the offset curve, same length as the input.
        """
        # Calculate first derivative
        derivative_coeffs = np.polyder(coeffs)

        # derivative[y]=x'(y)
        derivative = np.polyval(derivative_coeffs, y)
        
        # unit normal vector (perpendicular to tangent (derivative, 1))
        norm = np.sqrt(derivative**2 + 1)
        nx, ny = -1.0 / norm, derivative / norm

        sign = -1 if side == 'right' else 1
        dx = sign * d * nx
        dy = sign * d * ny
        xo, yo = x + dx, y + dy
        return xo, yo