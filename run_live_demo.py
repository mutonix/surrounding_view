import os
import cv2
import sys
from PyQt5.QtCore import QMutex

from surround_view import CaptureThread, CameraProcessingThread
from surround_view import FisheyeCameraModel, BirdView
from surround_view import MultiBufferManager, ProjectedImageBuffer
from surround_view import Buffer
import surround_view.param_settings as settings

from ps_detect import PSDetect
from ps_locate import PSLocate  

font = cv2.FONT_HERSHEY_SIMPLEX
img_buffer = Buffer(buffer_size=8)
locate_enabled = True

yamls_dir = os.path.join(os.getcwd(), "yaml")
camera_ids = [0, 1, 3, 4]
names = settings.camera_names
cameras_files = [os.path.join(yamls_dir, name + ".yaml") for name in names]
camera_models = [FisheyeCameraModel(camera_file, name) for camera_file, name in zip(cameras_files, names)]
resolution = None#(int(640), int(360))

detect_cfg = {
        'output_dir': './inference',
        'output_name': 'result0.jpg',
        'weights': os.path.join(os.getcwd(), './weights/PSregv3s_300last.pt'),
        'save_txt': False,
        'save_img': False,
        'img_size': 640,
        'conf_thres': 0.4
    }

ps_cfg = {
        'verPS_edge' : [110, 150], 
        'parPS_edge' : [450, 550], 
        'verPS_depth': 250,
        'parPS_depth': 250,
        'car_center' : (290, 400)
    }


def main():
    capture_tds = [CaptureThread(camera_id, name, resolution=resolution)
                   for camera_id, name in zip(camera_ids, names)]
    capture_buffer_manager = MultiBufferManager()
    for td in capture_tds:
        capture_buffer_manager.bind_thread(td, buffer_size=8)
        if (td.connect_camera()):
            td.start()

    proc_buffer_manager = ProjectedImageBuffer()
    process_tds = [CameraProcessingThread(capture_buffer_manager,
                                          camera_id,
                                          camera_model)
                   for camera_id, camera_model in zip(camera_ids, camera_models)]
    for td in process_tds:
        proc_buffer_manager.bind_thread(td)
        td.start()
    # 得到透视变换的图 结果在proc_buffer_manager的current_frames字典中

    birdview = BirdView(proc_buffer_manager)
    birdview.load_weights_and_masks("./weights.png", "./masks.png")

    psdetect = PSDetect(detect_cfg)
    psdetect.bind_buffer(img_buffer)

    pslocate = PSLocate(ps_cfg)

    ps_fstep = 0 # set framestep as 3
    result = None

    psdetect.start()
    birdview.start()
    
    while True:
        img = cv2.resize(birdview.get(), (580, 800))

        img_buffer.add(img, drop_if_full=True)

        if locate_enabled:
            if ps_fstep == 1:
                result = psdetect.get()
                ps_fstep = 0
            ps_fstep += 1

        img = pslocate.locate(img, result, box=True, ifpar=False)
        cv2.putText(img, "FPS: {}".format(birdview.stat_data.average_fps), (10,30), font, 1, (255,255,255), 2)
        cv2.imshow("PS_LOCATE", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # for td in capture_tds:
        #     print("camera {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")
        # for td in process_tds:
        #     print("process {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")
        # print("birdview fps: {}".format(birdview.stat_data.average_fps))
    

    for td in process_tds:
        td.stop()

    for td in capture_tds:
        td.stop()
        td.disconnect_camera()

if __name__ == "__main__":
    main()
