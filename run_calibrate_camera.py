"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fisheye Camera calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    python calibrate_camera.py \
        -i 0 \
        -grid 8x6 \
        -o fisheye.yaml \
        -framestep 20 \
        -flip 0 \
        -resolution 640x480 \
        --fisheye
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import CaptureThread, MultiBufferManager
import surround_view.utils as utils

# we will save the camera param file to this directory
TARGET_DIR = os.path.join(os.getcwd(), "yaml")

# default param file
DEFAULT_PARAM_FILE = os.path.join(TARGET_DIR, "back_cam.yaml")


def main():
    parser = argparse.ArgumentParser()

    # input video stream
    parser.add_argument("-i", "--input", type=int, default=3,
                        help="input camera device")

    # chessboard pattern size
    # --表示可选参数
    # action 表示一旦有这个参数 则把它设为True eg. 只用输入-fisheye

    # 棋盘图中每行和每列角点(交叉点)的个数
    parser.add_argument("-grid", "--grid", default="8x6",
                        help="size of the calibrate grid pattern")

    parser.add_argument("-framestep", type=int, default=20,
                        help="use every nth frame in the video")

    parser.add_argument("-o", "--output", default=DEFAULT_PARAM_FILE,
                        help="path to output yaml file")

    parser.add_argument("-fisheye", "--fisheye", action="store_true", default=True,
                        help="set true if this is a fisheye camera")

    parser.add_argument("-flip", "--flip", default=0, type=int,
                        help="flip method of the camera")

    parser.add_argument("-r", "--resolution", default=(int(640), int(360)),
                        help="the resolution of camera")
    
    # args是类似字典的结构 利用.来访问
    args = parser.parse_args()
    resolution = tuple([int(_) for _ in args.resolution])
    W, H = resolution

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    text1 = "press c to calibrate"
    text2 = "press q to quit"
    text3 = "device: {}".format(args.input)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.6

    grid_size = tuple(int(x) for x in args.grid.split("x")) # 分离8x6 到 (8, 6)
    grid_points = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_points[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    device = args.input # 摄像头编号
    # 创建1个摄像头线程对象
    cap_thread = CaptureThread(device_id=device,
                               api_preference=cv2.CAP_ANY, resolution=resolution)
    buffer_manager = MultiBufferManager()
    # 将该线程绑定到缓存管理对象
    buffer_manager.bind_thread(cap_thread, buffer_size=8)
    if cap_thread.connect_camera():
        cap_thread.start()
    else:
        print("cannot open device")
        return

    quit = False
    do_calib = False
    i = -1
    while True:
        i += 1
        img = buffer_manager.get_device(device).get().image
        if i % args.framestep != 0:
            continue

        print("searching for chessboard corners in frame " + str(i) + "...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS
        )
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            print("OK")
            imgpoints.append(corners)
            objpoints.append(grid_points)
            cv2.drawChessboardCorners(img, grid_size, corners, found)

        cv2.putText(img, text1, (20, 70), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text2, (20, 110), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text3, (20, 30), font, fontscale, (255, 200, 0), 2)
        cv2.imshow("corners", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            do_calib = True
            break

        elif key == ord("q"):
            quit = True
            break

    if quit:
        cap_thread.stop()
        cap_thread.disconnect_camera()
        cv2.destroyAllWindows()

    if do_calib:
        print("\nPerforming calibration...\n")
        N_OK = len(objpoints)
        if N_OK < 12:
            print("Less than 12 corners detected, calibration failed")
            return

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             cv2.fisheye.CALIB_CHECK_COND +
                             cv2.fisheye.CALIB_FIX_SKEW)

        if args.fisheye:
            # 鱼眼摄像头标定
            # mtx: K dist: D rvecs: R tvecs: t
            ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (W, H),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        else:
            # 非鱼眼
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                (W, H),
                None,
                None)

        if ret:
            # 如果标定成果 写入相机参数文件
            fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
            fs.write("resolution", np.int32([W, H]))
            fs.write("camera_matrix", K)
            fs.write("dist_coeffs", D)
            fs.release()
            print("succesfully saved camera data")
            cv2.putText(img, "Success!", (220, 240), font, 2, (0, 0, 255), 2)

        else:
            cv2.putText(img, "Failed!", (220, 240), font, 2, (0, 0, 255), 2)

        cv2.imshow("corners", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
