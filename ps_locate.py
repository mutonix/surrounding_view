import cv2
import os
import sys
import numpy as np

from PyQt5.QtCore import QMutex
from queue import Queue

from ps_detect import PSDetect
from ps_locate_utils import *

class PSLocate():

    def __init__(self, ps_cfg):

        self.ps_num = None
        self.init_cfg(ps_cfg)

    def init_cfg(self, ps_cfg):
        self.verPS_edge = ps_cfg['verPS_edge']
        self.parPS_edge = ps_cfg['parPS_edge']
        self.verPS_depth = ps_cfg['verPS_depth']
        self.parPS_depth = ps_cfg['parPS_depth']
        self.car_center = ps_cfg['car_center']

    def locate(self, 
               img, 
               result_xywh, 
               box=False, 
               ifpar=False, 
               var_thres=0.05, 
               occupy_thres=0.5):
        
        self.img = img

        if result_xywh is not None:
            result = result_xywh[:, :2]
            pts_pair, pdist = find_neighbour_point(result, car_center=self.car_center, var_thres=0.05)
            self.ps_num, ps_detected = ps_retrieval(pts_pair, pdist, self.verPS_edge, self.parPS_edge, ifpar)

            if ps_detected is not None:
                L_ps, R_ps = get_ps_poly(ps_detected, self.verPS_depth, self.parPS_depth)
                L_center, L_psc= get_center(L_ps)
                R_center, R_psc= get_center(R_ps)
                F_center = center_LR2F(L_center, R_center, car_center=self.car_center)
                final_ps = get_finalps(F_center, L_psc, R_psc)
                self.img = plot_ps(self.img, final_ps)

            if box and (ps_detected is not None):

                ps_detected = ps_detected.reshape(-1, 2)
                boxes = get_box(ps_detected)
                result_coord = get_ps_coord(ps_detected, self.img, self.car_center)
                
                for i, b in enumerate(boxes): 
                    label = '({}, {})'.format(*result_coord[i, :])
                    plot_one_box(b, self.img, label=label, color=(0, 0, 0), line_thickness=2)
        else:
            pass

        return self.img

    def show(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.img, "FPS: {}".format(50), (10, 30), font, 1,(255,255,255),2)
        cv2.imshow('img', self.img)
        
        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyAllWindows()

    def get_ps_num(self):
        return self.ps_num



if __name__ == "__main__":

    img = cv2.imread('./testimg/6.jpg')    

    model_cfg = {
            'output_dir': './inference',
            'output_name': 'result.jpg',
            'weights': './weights/PSregv3s_300last.pt',
            'save_txt': False,
            'save_img': False,
            'img_size': 640,
            'conf_thres': 0.4
        }

    ps_cfg = {
            'verPS_edge' : [100 ,200], 
            'parPS_edge' : [250, 500], 
            'verPS_depth': 370,
            'parPS_depth': 145,
            'car_center' : (300, 285)
        }

    # psdetect = PSDetect(model_cfg)
    # psdetect.load_model()
    # psdetect.run_inference(img)
    # result = psdetect.get_result()

    class Test_buffer():
        
        def __init__(self, img):
            self.buffer = Queue()
            self.buffer.put(img)


    gpu_mutex = QMutex()

    testb = Test_buffer(img)
    print(id(testb))
    psdetect = PSDetect(model_cfg)
    psdetect.bind_buffer(testb)
    psdetect.bind_lock(gpu_mutex)
    psdetect.start()

    result = psdetect.get()

    pslocate = PSLocate(ps_cfg)
    pslocate.locate(img, result, box=True, ifpar=False)
    pslocate.show()


        