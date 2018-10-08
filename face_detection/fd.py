"""
Face Detection from Baidu API for Python
The main Face Detection implemenetation.

Copyright (c) 2018 QTT@Shanghai
Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.09.30
"""
#!/usr/bin/python
# -*- coding: UTF-8 -*-


#################### Define by Baidu Document ####################
# import Baidu face detection Python SDK client
from aip import AipFace

# create Aipface client 
""" 你的 APPID AK SK """
APP_ID = '14325294'
API_KEY = 'AsvKVw4Kb1Hk5aYiZifxihHh'
SECRET_KEY = 'opAuwOddyUIfHgpkE4tdn3qf94PsaYAT'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

########## detect face in picture and mark position ##########
#image = "取决于image_type参数，传入BASE64字符串//或URL字符串//或FACE_TOKEN字符串"
image = "http://pic5.newssc.org/upload/ori/20160413/1460515143090.jpg"
imageType = "URL"

""" 调用人脸检测 """
#result = client.detect(image, imageType)

""" 如果有可选参数 """
options = {}
options["face_field"] = "age"
options["max_face_num"] = 2
options["face_type"] = "LIVE"

""" 带参数调用人脸检测 """
result = client.detect(image, imageType, options)

if __name__ =='__main__':
    print(result)

'''
# 读取图片
filePath = "girl.jpg"
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
'''
