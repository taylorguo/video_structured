"""
Read all detected scenes from video file in Python
Read all scenes for next detail analysis.

Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.09.30
"""

#!/usr/bin/python
# coding: UTF-8

import os

def get_files_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        return files


def download_video(url_address):

    # 下载mp4视频文件
    ADDRESS_ITEM = url_address.split("/")
    VIDEO_PREFIX = ADDRESS_ITEM[-2]
    EXT_NAME = "mp4"
    saved_file_name = VIDEO_PREFIX + "." + EXT_NAME

    # for Python3
    from urllib import request
    f = request.urlopen(url_address)
    video_data = f.read()
    with open(saved_file_name, "wb") as code:
        code.write(video_data)

    return saved_file_name


if __name__ =='__main__':

    # # get_files_name('/Users/taylorguo/Documents/Innotech/yolo_video/video_structured/scene_detect/goldeneye/')
    # current_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'goldeneye'+'/')
    # print(current_path)
    # files = get_files_name(current_path)
    # print('file_num:', len(files), '\nfiles:', files)


    URL_ADDRESS = "http://v4.qutoutiao.net/Act-ss-mp4-sd/4ca4c515e04b4f4e82b7bd254c69c2e8/sd.mp4"
    print(download_video(URL_ADDRESS))

