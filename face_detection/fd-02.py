# coding=UTF-8
# 利用baidu-aip库进行人脸识别
import cv2
#import matplotlib.pyplot as plt
from aip import AipFace
 
 
def detection(APP_ID, API_KEY, SECRET_KEY, filename, maxnum):
    '''
    :param APP_ID: https://console.bce.baidu.com/ai/创建人脸检测应用对应的APP_ID
    :param API_KEY: https://console.bce.baidu.com/ai/创建人脸检测应用对应的API_KEY
    :param SECRET_ID: https://console.bce.baidu.com/ai/创建人脸检测应用对应的SECRET_ID
    :param filename: 图片路径
    :param maxnum: 最大检测数
    :return:
    '''
    # 初始化AirFace对象
    aipface = AipFace(APP_ID, API_KEY, SECRET_KEY)
 
    # 设置
    options = {
        'max_face_num': 10,  # 检测人脸的最大数量
        'face_fields': "age,beauty,expression,faceshape",
    }
 
    # 读取文件内容
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()
 
    result = aipface.detect(get_file_content(filename), options)
    return result
 
 
def result_show(filename, result):
    '''
    :param filename: 原始图像
    :param result: 检测结果
    :return:
    '''
    img = cv2.imread(filename)
    face_num = len(result['result'])
    for i in range(face_num):
        location = result['result'][i]['location']
        left_top = (location['left'], location['top'])
        right_bottom = (left_top[0] + location['width'], left_top[1] + location['height'])
        cv2.rectangle(img, left_top, right_bottom, (200, 100, 0), 2)
 
    cv2.imshow('img', img)
    cv2.waitKey(0)
 
 
if __name__ =='__main__':
 
    # 定义APP_ID、API_KEY、SECRET_KEY
    APP_ID = '14325294'
    API_KEY = 'AsvKVw4Kb1Hk5aYiZifxihHh'
    SECRET_KEY = 'opAuwOddyUIfHgpkE4tdn3qf94PsaYAT'
 
    filename = 'girl.jpg'
    result = detection(APP_ID, API_KEY, SECRET_KEY, filename, 10)
    #result_show(filename, result)
 
 
