from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def debug_plotting(fig_debug3, ax_debug3, point_left, point_right, left_fitx, left_ploty, right_fitx, right_ploty, central_fitx, central_ploty, vehicle_x, vehicle_y, steering_angle, lookahead_pt, heading_error):
    ax_debug3.cla()
    if len(point_left):
        for i in range(len(point_left)):
            ax_debug3.plot(point_left[i][0], point_left[i][1], 'b.', markersize=10)
    if len(point_right):
        for i in range(len(point_right)):
            ax_debug3.plot(point_right[i][0], point_right[i][1], 'r.', markersize=10)

    if len(left_fitx):
        ax_debug3.plot(left_fitx, left_ploty, 'b-', linewidth=2, label='Left')
    if len(right_fitx):
        ax_debug3.plot(right_fitx, right_ploty, 'r-', linewidth=2, label='Right')
    if len(central_fitx):
        ax_debug3.plot(central_fitx, central_ploty, 'g-', linewidth=2, label='Center')

    ax_debug3.add_patch(Circle((vehicle_x, vehicle_y), radius=170, fill=False, color='orange', linewidth=1.5, label='Lookahead radius'))

    if lookahead_pt is not None:
        ax_debug3.plot(lookahead_pt[0], lookahead_pt[1], 'mo', markersize=8, label='Lookahead')

    ax_debug3.set_aspect('equal')
    ax_debug3.legend(loc='upper right', fontsize=7)
    ax_debug3.set_title(f'Steering Angle={np.degrees(steering_angle):.1f} (deg)  Heading Error={np.degrees(heading_error):.1f} (deg)')
    ax_debug3.set_xlim(-100, 100)
    ax_debug3.set_ylim(0, 200)

    fig_debug3.canvas.draw()
    fig_debug3.canvas.flush_events()