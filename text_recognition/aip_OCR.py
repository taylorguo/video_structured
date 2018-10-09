"""
Text Recognition in video frame from Baidu API in Python
The main Text Recognition implemenetation.

Copyright (c) 2018 innotechx Shanghai
Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.10.09
"""
#!/usr/bin/python
# coding=UTF-8

################ Define by Baidu Document #################
# Baidu_Aip Python SDK Installation： pip install baidu-aip

""" read image from a path(string) """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

""" convert image to BASE64: OCR is not needed """
def converter(filePath):
    # import base64
    # f = open(filePath,'rb') #二进制方式打开图文件
    # image = str(base64.b64encode(f.read()),"utf-8") #读取文件内容，转换为base64编码
    # f.close()
    # image = str(base64.b64encode(get_file_content(image)), "utf-8")
    image = get_file_content(filePath)
    return image

""" check if image address is URL or local picture """
def is_http_url(s):
    # Returns true if s is valid http url, else false
    # very simple reguarlization, need to improve !!!!!
    import re
    # if re.match('https?://(?:www)?(?:[\w-]{2,255}(?:\.\w{2,6}){1,2})(?:/[\w&%?#-]{1,300})?',s, re.I):
    if re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s, re.I):
        return True
    else:
        return False

""" aip_OCR main function """
def aip_OCR(image, *options):
    # import Baidu face detection Python SDK client
    from aip import AipOcr

    # create AipOCR client, create app in cloud.baidu.com
    """ 上海寅诺管理咨询有限公司: video_ocr: APPID AK SK """
    APP_ID = '14377678'
    API_KEY = 'rR41uq8xv5VpRTpOgLFKuuMZ'
    SECRET_KEY = 'UckxPOZ1GVHZMdt1N6EFsGlfE3AeyuCq'

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    ########## detect face in picture and mark position ##########
    #image = "BASE64字符串//或URL字符串"
    ###### image==local-image ######
    # Example:
    #### image = "http://pic5.newssc.org/upload/ori/20160413/1460515143090.jpg"
    """ 如果有可选参数 """
    options = {}
    options["detect_direction"] = "true"
    options["detect_language"] = "true"

    if is_http_url(image):
        ### image==URL ###
        # url = "https//www.x.com/sample.jpg"
        # """ 调用网络图片文字识别, 图片参数为远程url图片 """
        # result = client.webImageUrl(url);

        """ 带参数调用网络图片文字识别, 图片参数为远程url图片 """
        result = client.webImageUrl(image, options)
        ##################
    else:
        image = converter(image)
        # """ 调用网络图片文字识别, 图片参数为本地图片 """
        # result = client.webImage(image);

        """ 带参数调用网络图片文字识别, 图片参数为本地图片 """
        result = client.webImage(image, options)

    # return json format
    return result

if __name__ =='__main__':

    ########## imageType = URL ##########
    # 电影海报图片
    # image = "http://pic-bucket.nosdn.127.net/photo/0003/2018-10-09/DTM4GPTI3LF60003NOS.jpg"
    # image = "http://n.sinaimg.cn/ent/4_img/upload/79e6b5ad/402/w1024h1778/20181009/KT0t-hkvrhpt2726879.jpg"
    # 人民网新闻图片
    # image = "http://gx.people.com.cn/NMediaFile/2018/1009/LOCAL201810090941000026851057238.jpg"
    # 英文网页截图
    # image = "https://cdn0.tnwcdn.com/wp-content/blogs.dir/1/files/2017/09/Untitled-design-1-796x459.png"
    # 微博截屏 - 无法处理
    # 微博评论截图
    # image = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1539089519434&di=5297ca3260914bfa4a234b28b2c2f662&imgtype=0&src=http%3A%2F%2Fs9.rr.itc.cn%2Fr%2FwapChange%2F20165_18_19%2Fa2pi4c9787436409352.jpg"
    image = "weibo_comment.jpg" # 保存在本地可以识别
    # powerpoint 截屏 - 可以处理
    # image = "https://camo.githubusercontent.com/cb956f6329a2c3c31a0ecd2b209a75e97d140cfb/68747470733a2f2f7777772e64776865656c65722e636f6d2f6573736179732f666c6f73732d6c6963656e73652d736c6964652d696d6167652e706e67"
    ##### imageType = local picture #####
    # image = "KT0t-hkvrhpt2726879.jpg"
    # image = "image-0001.jpg"

    result = aip_OCR(image)
    print(result)