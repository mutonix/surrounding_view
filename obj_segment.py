
import sys
import os
import torch
import numpy as np
import cv2
from PyQt5.QtCore import QMutex
from queue import Queue

from lib.models import model_factory
from configs import cfg_factory
from utils.torch_utils import select_device

from surround_view import BaseThread, Buffer

class ObjSeg(BaseThread):

    def __init__(self, 
                 config, 
                 device_id=0,
                 drop_if_full=True,
                 buffer_size=8,
                 parent=None):
        super(ObjSeg, self).__init__(parent)
        self.model_init(config)

        self.device_id = device_id
        self.proc_buffer = None
        self.gpu_mutex = None
        self.buffer = Buffer(buffer_size)
        self.drop_if_full = drop_if_full
        self.psdetect_buffer = None

    def model_init(self, config):
        self.weight = config['weight']
        self.output = config['output_path']
        self.road_alpha = config['road_alpha']
        self.road = config['road']
        self.car = config['car']
        self.person = config['person']

        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        self.palette[[13, 0, 11]] = np.array([
                                [0, 0, 255],
                                [255, 0, 0],
                                [0, 255, 0]
                            ], dtype=np.uint8)
        self.mean = torch.tensor([0.3257, 0.3690, 0.3223], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor([0.2112, 0.2148, 0.2115], dtype=torch.float32).view(-1, 1, 1)
        self.model = None
        self.result = None

    def bind_proc_buffer(self, proc_buffer):
        self.proc_buffer = proc_buffer.buffer

    def bind_psdetect_buffer(self, psdetect):
        self.psdetect_buffer = psdetect.buffer

    def bind_lock(self, lock):
        self.gpu_mutex = lock

    def connect_model(self, model):
        self.model = model
        print('model connected!')

    def get(self):
        return self.buffer.get()

    def load_model(self):
        torch.set_grad_enabled(False)

        cfg = cfg_factory['bisenetv2']
        self.model = model_factory[cfg.model_type](n_classes=19) # class num
        self.model.load_state_dict(torch.load(self.weight, map_location='cpu'))
        self.model.eval()
        self.model.cuda()
        return self.model

    def run_inference(self, img):

        assert self.model is not None, 'You have to load the model first.'

        torch.set_grad_enabled(False)

        self.img = img
        
        obj_img = self.img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        obj_img = torch.from_numpy(obj_img).div_(255).sub_(self.mean).div_(self.std).unsqueeze(0).cuda()
        self.result = self.model(obj_img)[0].argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint8)

    def plot_road(self, img, result, ifwrite=False):
        self.img = img
        result_pal = self.palette[result]

        if ifwrite:
            palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
            result_pal = palette[result]
            cv2.imwrite(self.output, result_pal)

        if self.road:
            road_mask = np.where(result==0)
            if np.any(road_mask):
                self.img[road_mask] = result_pal[road_mask] * self.road_alpha + self.img[road_mask] * (1 - self.road_alpha)

        return self.img

    def get_result(self):
        car_mask = np.where(self.result==13)
        road_mask = np.where(self.result==11)
        if np.any(car_mask):
            self.result[car_mask] = 100
        if np.any(road_mask):
            self.result[road_mask] = 200
        return self.result

    def show(self, img=None):

        if img is None:
            if self.road:
                result_pal = self.img
            else:
                result_pal = self.palette[self.result]
        else:
            result_pal = img

        cv2.imshow('img', result_pal)
        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyAllWindows()

    def run(self):

        assert self.model is not None, 'You have to connect a model first'
        assert self.gpu_mutex is not None, 'Need a gpu lock'
        assert self.proc_buffer is not None, 'Need a proc_buffer_manager to run'
        
        while True:
            if self.stopped:
                self.stopped = False
                break
            
            if self.psdetect_buffer.size() > 1:
                self.gpu_mutex.lock()
                self.run_inference(self.proc_buffer.get()[self.device_id])
                torch.cuda.synchronize()
                self.gpu_mutex.unlock()
                self.buffer.add(self.get_result(), self.drop_if_full)
        

# road_mask = np.where(out==0)
# car_mask = np.where(out==13)

# out_pal = palette[out]
# blank[car_mask] = out_pal[car_mask]
# img = cv2.imread(args.img_path)
# alpha = 0.5
# img[road_mask] = out_pal[road_mask] * alpha + img[road_mask] * (1 - alpha)
# #img[car_mask] = out_pal[car_mask] * alpha + img[car_mask] * (1 - alpha)


if __name__ == "__main__":

    model_cfg = {
        'weight' : 'weights/seg_cityscapes.pth',
        'output_path': './project.jpg',
        'road_alpha' : 0.5,
        'road' : True,
        'car' : True,
        'person' : True
    }
    

    img = cv2.imread('./testimg/test_01.png')
    
    dict = {0 : img}
    class Proc_Buffer():
    
        def __init__(self, dict):
            self.buffer = Queue()
            self.buffer.put(dict)

    proc_buffer = Proc_Buffer(dict)
    gpu_mutex = QMutex()

    objseg = ObjSeg(model_cfg)
    public_seg_model = objseg.load_model()

    objseg1 = ObjSeg(model_cfg)
    objseg1.connect_model(public_seg_model)
    objseg1.bind_proc_buffer(proc_buffer)
    objseg1.bind_lock(gpu_mutex)

    objseg1.start()
    result = objseg1.get()
    img = objseg1.plot_road(img, result)
    objseg1.show(img)


    # objseg = ObjSeg(model_cfg)
    # objseg.load_model()
    # objseg.run_inference(img)
    # result = objseg.get_result()
    # objseg.plot_result(img, result)
    # objseg.show()

