import cv2
import numpy as np
from PyQt5.QtCore import qDebug

from .base_thread import BaseThread

class CameraProcessingThread(BaseThread):

    """
    Thread for processing individual camera images, i.e. undistort, project and flip.
    """

    def __init__(self,
                 capture_buffer_manager,
                 device_id,
                 camera_model,
                 drop_if_full=True,
                 parent=None):
        """
        capture_buffer_manager: an instance of the `MultiBufferManager` object.
        device_id: device number of the camera to be processed.
        camera_model: an instance of the 'FisheyeCameraModel` object.
        drop_if_full: drop if the buffer is full.
        """
        super(CameraProcessingThread, self).__init__(parent)
        self.capture_buffer_manager = capture_buffer_manager
        self.device_id = device_id
        self.camera_model = camera_model
        self.drop_if_full = drop_if_full
        # an instance of the `ProjectedImageBuffer` object
        self.proc_buffer_manager = None

    def run(self):
        if self.proc_buffer_manager is None:
            raise ValueError("This thread has not been binded to any processing thread yet")

        while True:

            if self.stopped:
                self.stopped = False
                break

            self.processing_time = self.clock.elapsed()
            self.clock.start()


            raw_frame = self.capture_buffer_manager.get_device(self.device_id).get().image
            und_frame = self.camera_model.undistort(raw_frame)
            pro_frame = self.camera_model.project(und_frame)
            flip_frame = self.camera_model.flip(pro_frame)            

            self.proc_buffer_manager.set_frame_for_device(self.device_id, flip_frame)
            self.proc_buffer_manager.sync(self.device_id)
            
            # update statistics
            self.update_fps(self.processing_time)
            self.stat_data.frames_processed_count += 1
            # inform GUI of updated statistics
            self.update_statistics_gui.emit(self.stat_data)
