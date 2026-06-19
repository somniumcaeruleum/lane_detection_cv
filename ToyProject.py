from matplotlib import pyplot as plt
import numpy as np
import cv2

from debug import debug_plotting

from bev import raw_to_bev
from bev import thresholding

from sliding_window import sliding_window_lane_detect
from sliding_window import fit_lane

from Offset import offset_curve

from frame2bev import frame2bev

from Pure_Pursuit import pure_pursuit

# thresholded -> sliding_window_lane_detect -> plot on reality -> path planning
class ToyProject:
    def __init__(self, input_path, debug=False):
        # vertical distance, horizontal distance per pixel
        self.vertical = 0.376 # y-scale (cm/pixel)
        self.horizon = 0.159  # x-scale (cm/pixel)

        #horizontal distance between left, right lane (pixel)
        self.dist = (85/self.horizon)//2
        self.dist_real = 85//2

        # global vars
        self.input_path = input_path
        self.debug = debug

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
            plt.ion()
            fig_debug3, ax_debug3 = plt.subplots(figsize=(5, 5))
        
        pre_left_fitx, pre_left_ploty = [], []
        pre_right_fitx, pre_right_ploty = [], []
        pre_central_fitx, pre_central_ploty = [], []
        pre_steering_angle, pre_lookahead_pt, pre_heading_error = 0, (0, -170), 0
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

            ### Control: Pure Pursuit
            vehicle_x, vehicle_y  = frame2bev(frame_width/2, frame_height, frame, bev_matrix=self.bev_matrix, vertical=self.vertical, horizon=self.horizon)
            vehicle_y += 90

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

            if self.debug:
                debug_plotting(fig_debug3, ax_debug3, point_left, point_right, left_fitx, left_ploty, right_fitx, right_ploty, central_fitx, central_ploty, vehicle_x, vehicle_y, steering_angle, lookahead_pt, heading_error)

        cap.release()
        cv2.destroyAllWindows()
        if fig_debug3 is not None:
            plt.close(fig_debug3)