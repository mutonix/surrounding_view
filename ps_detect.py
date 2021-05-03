import os
import shutil
import time
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression,scale_coords, xyxy2xywh, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized

from surround_view import BaseThread, Buffer

class PSDetect(BaseThread):

    def __init__(self, 
                 config, 
                 set_log=False, 
                 drop_if_full=True,
                 buffer_size=8,
                 parent=None):
        super(PSDetect, self).__init__(parent)
        self.model_init(config, set_log)

        self.bv_buffer = None
        self.drop_if_full = drop_if_full
        self.buffer = Buffer(buffer_size)
    
    def get(self):
        return self.buffer.get()

    def bind_buffer(self, img_buffer):
        self.bv_buffer = img_buffer

    def model_init(self, config, set_log):

        self.im0 = None
        self.out_dir = config['output_dir']
        self.out_name = config['output_name']
        self.weights = config['weights']
        self.save_txt = config['save_txt']
        self.save_img = config['save_img']
        self.imgsz = config['img_size']
        self.conf_thres = config['conf_thres']
        self.set_log = set_log
        self.device = ''
        self.half = False
        self.model = None
        self.names = None
        self.pred_xyxy = None
        self.pred_xywh = None

    def load_model(self):
        
        torch.set_grad_enabled(False)

        if self.conf_thres:
            set_logging()
        self.device = select_device(self.device) # 选定计算设备 cuda:0
        if os.path.exists(self.out_dir):   
            shutil.rmtree(self.out_dir)  # 如果输出文件夹存在，则删除，并重新新建输出文件夹
        os.makedirs(self.out_dir)  
        self.half = self.device.type != 'cpu'  # 半精度标志位，非cpu则为True

        # 装载模型
        self.model = attempt_load(self.weights, map_location=self.device)  #  装载单个或多个模型
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # 判断图片大小是否能被模型步长长整除
        if self.half:
            self.model.half()  # to FP16  CUDA enabled -> 半精度模型

        # 获取分类
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names # 不同分类的名字

        # 预运行一次模型
        img_test = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device) 
        _ = self.model(img_test.half() if self.half else img_test) if self.device.type != 'cpu' else None  


    def run_inference(self, img0):

        torch.set_grad_enabled(False)

        assert self.model is not None, 'you have to load the model first'

        # h, w = img0.shape[:2]
        # img = cv2.resize(img0, (int(w / 3), int(h / 3)))
        # img = cv2.blur(img, (5, 5))
        # img = util.random_noise(img,mode='gaussian',var=0.001)
        # img0 = cv2.resize(img, (w, h))

        # 装载图片
        img = letterbox(img0, new_shape=self.imgsz)[0] # 调整图片使得最长边为640, 另一边不一定为640，但一定为步长64的倍数（等比例缩放）
        #  -> (H, W, 3'BGR') 
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB -> (3'RGB', H, W)
        img = np.ascontiguousarray(img) # 转化为连续存储

        # 图片预处理
        img = torch.from_numpy(img).to(self.device) # -> (3'RGB', H, W)
        img = img.half() if self.half else img.float()  #  转换为半精度数据
        img /= 255.0  # 0 - 255 to 0.0 - 1.0  RGB值归一化
        if img.ndimension() == 3:   # -> (1, 3'RGB', H, W)
            img = img.unsqueeze(0) 

        # 正式开始推断 单张图片
        t1 = time_synchronized() # 等待cuda完成计算，cpu才获取当前时间 （预运行一次模型的时间: t1 - t0）
        pred = self.model(img)[0] # 进行单张图片的推断 

        # 进行非极大值抑制
        pred = non_max_suppression(pred, self.conf_thres)
        t2 = time_synchronized() # 进行单张图片推断所用时间: t2 - t1

        # 处理推断结果
        # 推断结果det -> (boxs_num, 6)  6 -> [xmin, ymin, xmax, ymax, 推断分数（0~1）, class_id] 这里的xyxy是非归一化坐标
        s, im0 = '', img0 #  一次只处理一张图片
        pred = pred[0]

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))] # 不同分类有不同标记的颜色
        save_path = str(Path(self.out_dir) / Path(self.out_name).name) # 推断后的图片的保存路径
        txt_path = str(Path(self.out_dir) / Path(self.out_name).stem)  # 结果保存在txt文件
        # 结果写入txt -> 单张图片写入 或者某视频的一帧帧地写入
        s += '%gx%g ' % img.shape[2:]  # print string 图片（帧）的大小写入 HxW
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化xywh坐标用 -> [im0_w, im0_h, im0_w, im0_h]
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            # 调整输出框的大小对应原始图片的大小 (xyxy)

            # 打印结果
            for c in pred[:, -1].unique(): # 将分类的编号取出 去除重复值 并从小到大排序 -> 目的为下一步的掩码
                n = (pred[:, -1] == c).sum()  # detections per class 逻辑矩阵调用sum 加的是0或者1 -> 某类的推断框的个数
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string 每张图片的每一类的名字

            # 写入结果
            for *xyxy, conf, cls in reversed(pred):  # 将det的迭代顺序翻转 
                # *xyxy 类似函数中用法（接受剩余的） 前提是后面或前有其他暂存变量 -> [:,[xyxy, conf , cls]]
                if self.save_txt:  # 写入txt
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # 将xyxy表示的框转化为 归一化xywh表示 的框
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    # 将推断的框以label文件格式写入保存

                # 用opencv画框和标签
                if self.save_img: 
                    label = '%s %.2f' % (self.names[int(cls)], conf) # 展示分类的名字 及推断分数
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # 保存图片
            if self.save_img:
                cv2.imwrite(save_path, im0)

        if self.save_txt or self.save_img:
            print('Results saved to %s' % Path(self.out_dir))

        print('Done. (%.3fs)' % (time.time() - t1))

        self.pred_xyxy = pred.cpu().numpy() if pred is not None else None
        self.pred_xywh = self.pred_xyxy.copy() if pred is not None else None
        if self.pred_xywh is not None:
            self.pred_xywh[:, :4] = xyxy2xywh(self.pred_xywh[:, :4]) 

        
    def get_result(self, cls_id=0):
        if self.pred_xywh is not None:
            result_flag = self.pred_xywh[:, -1] == cls_id
            result = self.pred_xywh[result_flag]
        else:
            result = None

        return result

    def run(self):

        assert self.bv_buffer is not None, "This thread requires a buffer of birdview to run"
        self.load_model()

        while True:
            if self.stopped:
                self.stopped = False
                break

            self.run_inference(self.bv_buffer.get())            
            torch.cuda.synchronize()
            self.buffer.add(self.get_result(), self.drop_if_full)



if __name__ == "__main__":

    img = cv2.imread(r'C:\Users\mutonix\Desktop\test.jpg')

    model_cfg = {
            'output_dir': './inference',
            'output_name': 'result0.jpg',
            'weights': './weights/PSregv3s_300last.pt',
            'save_txt': False,
            'save_img': True,
            'img_size': 640,
            'conf_thres': 0.4
        }

    psdetect = PSDetect(model_cfg)
    psdetect.load_model()
    psdetect.run_inference(img)
    result = psdetect.get_result()
    
