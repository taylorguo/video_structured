"""
Body Analysis in video frame from Baidu API in Python
The main Body Analysis implemenetation.

Copyright (c) 2018 innotechx Shanghai
Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.10.10
"""
#!/usr/bin/env python
# coding: UTF-8

################ Define by Baidu Document #################
# Baidu_Aip Python SDK Installation： pip install baidu-aip
# import Baidu face detection Python SDK client
from aip import AipBodyAnalysis

# before create AipOCR client below, create app in cloud.baidu.com
""" video_body_analysis: APPID AK SK """
APP_ID = '14409622'
API_KEY = 'nyNgj3XE8L25YVTEFaW5c2Lq'
SECRET_KEY = 'xYuppHC7huURghiKofdiiSxXV92uka0g'

client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

""" read image from a path(string) """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

### below 5 functions return json format, detail definition:
# https://ai.baidu.com/docs#/BodyAnalysis-Python-SDK/25af04ec

""" pose estimation """
def pose_estimation(image):
    # Argument- image: String-BASE64
    image = get_file_content(image)

    """ 调用人体关键点识别 """
    result = client.bodyAnalysis(image)

    # return json format
    return result

""" body attribute """
def body_attribute(image, *options):
    # Argument- image: String-BASE64
    image = get_file_content(image)

    """ 如果有可选参数 """
    options = {}
    options["type"] = "gender"

    """ 带参数调用人体属性识别 """
    result = client.bodyAttr(image, options)

    # return json format
    return result

""" body numbering """
def body_numbering(image, *options):
    # Argument- image: String-BASE64
    image = get_file_content(image)

    """ 如果有可选参数 """
    options = {}
    options["area"] = "0,0,100,100,200,200"
    options["show"] = "false"

    """ 带参数调用人流量统计 """
    result = client.bodyNum(image, options)

    # return json format
    return result

""" gesture recognition """
def gesture_recognition(image):
    # Argument- image: String-BASE64
    image = get_file_content(image)

    """ 调用手势识别 """
    result = client.gesture(image)

    # return json format
    return result

""" body segmentation """
def body_segmentation(image):
    # Argument- image: String-BASE64
    image = get_file_content(image)

    """ 调用人像分割 """
    result = client.bodySeg(image)

    # return json format: result[labelmap] is binary map with base64encode
    # will process next update
    return result

def base64_2binaryimage(result):
    import cv2
    import numpy as np
    import base64
    import datetime
    width = 300
    height = 500
    body_seg_file_name = "body_seg_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    labelmap = base64.b64decode(result['labelmap'])    # res为通过接口获取的返回json
    nparray = np.fromstring(labelmap, np.uint8)
    labelimg = cv2.imdecode(nparray, 1)
    # width, height为图片原始宽、高
    labelimg = cv2.resize(labelimg, (width, height), interpolation=cv2.INTER_NEAREST)
    im_new = np.where(labelimg==1, 255, labelimg)
    cv2.imwrite(body_seg_file_name, im_new)
    return None

if __name__ =='__main__':

    image = "wz_01.jpg"

    pose_result = pose_estimation(image)
    print("人体关键点识别:", pose_result)

    attr_result = body_attribute(image)
    print("\n人体属性识别:", attr_result)

    num_result = body_numbering(image)
    print("\n人流量统计:", num_result)

    ges_result = gesture_recognition(image)
    print("\n手势识别:", ges_result)

    seg_result = body_segmentation(image)
    base64_2binaryimage(seg_result)
    print("\n人像分割:", seg_result, "\n分割图片已保存!")
    # import base64
    # image_seg = base64.b64decode(seg_result['labelmap'])
    # # print(image_seg)
    # fh = open("body_segment_result.png", "wb")
    # fh.write(image_seg)
    # fh.close()