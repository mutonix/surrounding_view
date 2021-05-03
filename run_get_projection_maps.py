"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import FisheyeCameraModel, PointSelector, display_image
import surround_view.param_settings as settings


def get_projection_map(camera_model, image):
    und_image = camera_model.undistort(image)
    name = camera_model.camera_name
    gui = PointSelector(und_image, title=name)
    dst_points = settings.project_keypoints[name]
    choice = gui.loop()
    if choice > 0:
        src = np.float32(gui.keypoints)
        dst = np.float32(dst_points)
        camera_model.project_matrix = cv2.getPerspectiveTransform(src, dst)
        proj_image = camera_model.project(und_image)

        ret = display_image("Bird's View", proj_image)
        if ret > 0:
            return True
        if ret < 0:
            cv2.destroyAllWindows()

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-camera", default="front_cam",
                        #required=True,
                        choices=["front_cam", "back_cam", "left_cam", "right_cam"],
                        help="The camera view to be projected")
    parser.add_argument("-scale", nargs="+", default=(0.6, 0.7),
                        help="scale the undistorted image")
    parser.add_argument("-shift", nargs="+", default=(0, 0),
                        help="shift the undistorted image")
    args = parser.parse_args()

    # front 0.7 0.8 0 -20
    # right 0.6 0.7 -50 20
    # back 0.7 0.7 0 0
    # left 0.6 0.7 -50 0
    if args.scale is not None:
        scale = [float(x) for x in args.scale]
    else:
        scale = (1.0, 1.0)

    if args.shift is not None:
        shift = [float(x) for x in args.shift]
    else:
        shift = (0, 0)

    camera_name = args.camera
    camera_file = os.path.join(os.getcwd(), "yaml", camera_name + ".yaml")
    image_file = os.path.join(os.getcwd(), "images", camera_name + ".png")
    image = cv2.imread(image_file)
    camera = FisheyeCameraModel(camera_file, camera_name)
    
    # img = cv2.imread(os.path.join(os.getcwd(), "..", "front.jpg"))
    # img = camera.project(img)
    # cv2.imshow('img', img)
    # if cv2.waitKey(0) & 0xff == ord('q'):
    #     cv2.destroyAllWindows()

    camera.set_scale_and_shift(scale, shift)
    success = get_projection_map(camera, image)
    if success:
        print("saving projection matrix to yaml")
        camera.save_data()
    else:
        print("failed to compute the projection map")


if __name__ == "__main__":
    main()
    
