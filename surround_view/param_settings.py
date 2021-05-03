import os
import cv2
import numpy as np

camera_names = ["front_cam", "back_cam", "left_cam", "right_cam"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 106
shift_h = 125 # 155 - 30

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
inn_shift_w = 21.5
inn_shift_h = 62.5 # 32.5 + 30

# total width/height of the stitched image
total_w = 368 + 2 * shift_w # 580
total_h = 550 + 2 * shift_h # 800

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = int(shift_w + 87.5 + inn_shift_w) # 215
xr = int(total_w - xl) # 365
yt = int(shift_h + 87.5 + inn_shift_h) # 275
yb = int(total_h - yt) # 525
# --------------------------------------------------------------------
# chessboard_w = 87.5  chessboard_h = 87.5
# chesscarpter_w = 368 chesscarpet_h = 490


# --------------------------------------------------------------------
project_shapes = {
    "front_cam": (total_w, yt), # 580, 275
    "back_cam":  (total_w, yt), # 580, 275
    "left_cam":  (total_h, xl), # 800, 215
    "right_cam": (total_h, xl)  # 800, 215
}

# pixel locations of the four points to be choosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script
project_keypoints = {
    "front_cam": [(shift_w + 70, shift_h),
              (shift_w + 298, shift_h),
              (shift_w + 70, shift_h + 70),
              (shift_w + 298, shift_h + 70)],

    "back_cam":  [(shift_w + 70, shift_h),
              (shift_w + 298, shift_h),
              (shift_w + 70, shift_h + 70),
              (shift_w + 298, shift_h + 70)],

    "left_cam":  [(shift_h + 70, shift_w),
              (shift_h + 480, shift_w),
              (shift_h + 70, shift_w + 70),
              (shift_h + 480, shift_w + 70)],

    "right_cam": [(shift_h + 70, shift_w),
              (shift_h + 480, shift_w),
              (shift_h + 70, shift_w + 70),
              (shift_h + 480, shift_w + 70)]
}

car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
car_image = cv2.resize(car_image, (xr - xl, yb - yt)) if car_image is not None \
                else np.zeros((yb - yt, xr - xl, 3), dtype=np.uint8)
