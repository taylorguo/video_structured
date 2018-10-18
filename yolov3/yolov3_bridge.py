"""
yolov3_bridge to call yolov3 detect().
Read all scenes for detection by yolov3.

Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.10.16
"""

#!/usr/bin/python
# coding: UTF-8

# from .detect import detect
from yolov3.detect import detect

class optparams:
        # image_folder ='data/samples'
        # output_folder ='output'
        plot_flag =True
        txt_out =False

        cfg ='yolov3/cfg/yolov3.cfg'
        class_path ='yolov3/data/coco.names'
        conf_thres =0.50
        nms_thres =0.45
        batch_size =1
        img_size =32 * 13


        def __init__(self, image_folder, output_folder):
            self.image_folder = image_folder #'data/samples'
            self.output_folder = output_folder #'output'

        def displayFolders(self):
            print("****** image folder : ", self.image_folder,
                  ", output folder : ", self.output_folder)


def yolov3_detect(params):
    detect(params)


if __name__ == '__main__':

    params = optparams('goldeneye', 'output')
    params.displayFolders()
    yolov3_detect(params)