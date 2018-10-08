"""
Face Detection from Baidu API in Python
The main Face Detection implemenetation.

Copyright (c) 2018 QTT@Shanghai
Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.09.30
"""
#!/usr/bin/python
# coding=UTF-8

def face_detection(image, imageType, *options):
    #################### Define by Baidu Document ####################
    # pip3 install baidu_aip - After installing Baidu-AI python package,
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
    # Example: 
    #### image = "http://pic5.newssc.org/upload/ori/20160413/1460515143090.jpg"
    #### imageType = "URL"
    image = image
    imageType = imageType

    """ 调用人脸检测 """
    #result = client.detect(image, imageType)

    """ 如果有可选参数 """
    options = {}
    options["face_field"] = "age"
    options["max_face_num"] = 10
    options["face_type"] = "LIVE"

    """ 带参数调用人脸检测 """
    result = client.detect(image, imageType, options)
    # return json format
    return result

if __name__ =='__main__':

    ############ imageType = BASE64 ############
    #####     convert image to BASE64      #####
    # https://www.css-js.com/tools/base64.html #
    #import base64
    #f = open("/Users/taylorguo/Documents/Innotech/iqa-001.jpg",'rb') #二进制方式打开图文件
    #image = str(base64.b64encode(f.read()),"utf-8") #读取文件内容，转换为base64编码
    #f.close()

    #imageType = "BASE64"
    # result = face_detection(image, imageType)

    ########## imageType = URL ########## 
    image = "http://pic5.newssc.org/upload/ori/20160413/1460515143090.jpg"
    imageType = "URL"
    result = face_detection(image, imageType)

    print(result)