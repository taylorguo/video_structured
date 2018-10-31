"""
Scene Detection for video file in Python
The main Scene Detection implemenetation.

Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.09.30
"""

#!/usr/bin/python
# coding: UTF-8

# Standard Library Imports
from __future__ import print_function
import os
import time
import math

## https://github.com/Breakthrough/PySceneDetect
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

# STATS_FILE_PATH = 'testvideo.stats.csv'

def scene_detect(video):
    # Create a video_manager point to video file testvideo.mp4. Note that multiple
    # videos can be appended by simply specifying more file paths in the list
    # passed to the VideoManager constructor. Note that appending multiple videos
    # requires that they all have the same frame size, and optionally, framerate.
    video_manager = VideoManager([video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    print("****** 正在分析视频场景 ... create managers!")

    try:
        ### If stats file exists, load it.
        # if os.path.exists(STATS_FILE_PATH):
        #     # Read stats from CSV file opened in read mode:
        #     with open(STATS_FILE_PATH, 'r') as stats_file:
        #         stats_manager.load_from_csv(stats_file, base_timecode)

        # start_time = base_timecode + 0     # 00:00:00.667 -> 20
        # end_time = base_timecode + 300.0     # 00:00:20.000 -> 200
        # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
        # video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager. | , end_time=end_time
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.

        print('List of scenes obtained:')
        if scene_list:
            for i, scene in enumerate(scene_list):
                print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                    i+1,
                    scene[0].get_timecode(), scene[0].get_frames(),
                    scene[1].get_timecode(), scene[1].get_frames(),))
        else:
            print('****** 视频是单一场景 ...  only 1 Scene in the video ')

        # ## We only write to the stats file if a save is required:
        # if stats_manager.is_save_required():
        #     with open(STATS_FILE_PATH, 'w') as stats_file:
        #         stats_manager.save_to_csv(stats_file, base_timecode)

        generate_images(video_manager, scene_list, video)

    finally:
        video_manager.release()




def get_output_file_path(file_path, output_dir=None):
    # type: (str, Optional[str]) -> str
    """ Get Output File Path: Gets full path to output file passed as argument, in
    the specified global output directory (scenedetect -o/--output) if set, creating
    any required directories along the way.

    Arguments:
        file_path (str): File name to get path for.  If file_path is an absolute
            path (e.g. starts at a drive/root), no modification of the path
            is performed, only ensuring that all output directories are created.
        output_dir (Optional[str]): An optional output directory to override the
            global output directory option, if set.

    Returns:
        (str) Full path to output file suitable for writing.

    """
    if file_path is None:
        return None

    project_path = os.path.dirname(os.path.realpath(__file__))

    output_dir = project_path if output_dir is None else output_dir
    # If an output directory is defined and the file path is a relative path, open
    # the file handle in the output directory instead of the working directory.
    if output_dir is not None and not os.path.isabs(file_path):
        file_path = os.path.join(output_dir, file_path)
    # Now that file_path is an absolute path, let's make sure all the directories
    # exist for us to start writing files there.
    try:
        os.makedirs(os.path.split(os.path.abspath(file_path))[0])
    except OSError:
        pass
    return file_path


# ### Create folder to generate images
def create_folder(video_name):
    img_folder = video_name[0:-4]
    isExists = os.path.exists(img_folder)
    if not isExists:
        os.makedirs(img_folder)
        print('场景图片目录 ' + img_folder + ' 创建成功， 生成图片 ......')
        return img_folder
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(img_folder + ' 目录已存在')
        return img_folder
    return img_folder


def generate_images(video_manager, scene_list, video_name,
                     image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
                     output_dir=None):
    import logging
    import cv2
    from string import Template
    from scenedetect.platform import tqdm
    logging.getLogger().setLevel(logging.INFO)

    num_images = 2
    image_extension = "jpg"

    # imwrite_param = []
    # if image_param is not None:
    #     imwrite_param = [imwrite_params[image_extension], image_param]

    # Reset video manager and downscale factor.
    video_manager = video_manager
    video_manager.release()
    video_manager.reset()
    video_manager.set_downscale_factor(1)
    video_manager.start()

    print("Detect Scenes:", len(scene_list))

    # Setup flags and init progress bar if available.
    completed = True
    logging.info('Generating output images (%d per scene)...', num_images)
    progress_bar = None
    if tqdm:
        progress_bar = tqdm(
            total=len(scene_list) * num_images, unit='images')

    filename_template = Template(image_name_template)

    if scene_list:
        scene_num_format = '%0'
        scene_num_format += str(max(3, math.floor(math.log(len(scene_list), 10)) + 1)) + 'd'
        image_num_format = '%0'
        image_num_format += str(math.floor(math.log(num_images, 10)) + 2) + 'd'

    timecode_list = dict()

    for i in range(len(scene_list)):
        timecode_list[i] = []

    if num_images == 1:
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time - start_time
            timecode_list[i].append(start_time + int(duration.get_frames() / 2))

    else:
        middle_images = num_images - 2
        for i, (start_time, end_time) in enumerate(scene_list):
            timecode_list[i].append(start_time)

            if middle_images > 0:
                duration = (end_time.get_frames() - 1) - start_time.get_frames()
                duration_increment = None
                duration_increment = int(duration / (middle_images + 1))
                for j in range(middle_images):
                    timecode_list[i].append(start_time + ((j+1) * duration_increment))

            # End FrameTimecode is always the same frame as the next scene's start_time
            # (one frame past the end), so we need to subtract 1 here.
            timecode_list[i].append(end_time - 1)


    img_folder = create_folder(video_name)

    for i in timecode_list:
        for j, image_timecode in enumerate(timecode_list[i]):
            video_manager.seek(image_timecode)
            video_manager.grab()
            ret_val, frame_im = video_manager.retrieve()
            if ret_val:
                cv2.imwrite(
                    get_output_file_path(
                        '%s.%s' % (filename_template.safe_substitute(
                            VIDEO_NAME=video_name,
                            SCENE_NUMBER=scene_num_format % (i+1),
                            IMAGE_NUMBER=image_num_format % (j+1)
                        ), image_extension),
                        output_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), img_folder)), frame_im)
                # return output_dir
            else:
                completed = False
                break
            if progress_bar:
                progress_bar.update(1)


    # type: (List[Tuple[FrameTimecode, FrameTimecode]) -> None

    if not scene_list:

        img_folder = create_folder(video_name)
        video_manager.release()
        video_manager.reset()
        video_manager.set_downscale_factor(1)
        video_manager.start()
        video_manager.seek("00:00:02.000")
        video_manager.grab()
        ret_val, frame_im = video_manager.retrieve()
        if ret_val:
            cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     img_folder,
                                     video_name[0:-4]+"."+image_extension),
                        frame_im)
    #         return os.path.join(os.path.dirname(os.path.realpath(__file__)), img_folder, video_name[0:-4])

    if not completed:
        logging.error('Could not generate all output images.')



# 计算某一目录下所有图片的iqamos,返回mos最大值
# get image and return image iqamos
import os, subprocess
# import tethys_utils, conf
def calc_iqamos(imagelist, imagefolder):
    # second get images's mos
    moslist = []
    for image in imagelist:
        full_image_name = os.path.join(imagefolder, image)
        cmdstr = "iqa %s" % (full_image_name)
        # print(cmdstr)
        result = subprocess.Popen(cmdstr, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = result.stdout.readlines()
        # print(out[0].decode("utf-8"))
        result = out[0].decode("utf-8")
        if (result.find("err") >= 0):
            # print(" cmdstr : ", cmdstr)
            # print("out : ", result)
            cmdstr = "rm -f %s" % (full_image_name)
            os.system(cmdstr)
            break
        moslist.append(float(result))
        cmdstr = "rm -f %s" % (full_image_name)
        os.system(cmdstr)
    moslist.sort(reverse=True)
    if (len(moslist) >= 1):
        iqamos = moslist[0]
    else:
        iqamos = -3
    return iqamos

# 获取指定目录下所有目录
def get_dirs_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return dirs

# 获取指定目录下所有文件
def get_files_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

# 获取由video文件名(把.mp4之前的文件名作为目录)生成的目录
def get_video_folder_name(video):
    if video[-3:] == "mp4":
        folder_name = video[:-4]
    return folder_name


def get_iqamos(VIDEO_PATH):
    # 获取项目目录下的视频文件分解好的 图片目录名 和 目录下的所有图片
    video_foler_name = get_video_folder_name(VIDEO_PATH)
    all_images = get_files_name(video_foler_name)
    # 获取完整的图片目录路径, 传递给iqa命令行 (默认已安装到系统bash路径中)
    current_path = os.path.dirname(os.path.realpath(__file__))
    video_images_path = os.path.join(current_path, video_foler_name)
    result = calc_iqamos(all_images, video_images_path)
    os.system("rm -rf %s" %(video_foler_name))
    if not os.path.exists("url_videos"):
        os.makedirs("url_videos")
    os.system("mv %s %s" % (video_name, "url_videos"))
    return result



if __name__ == "__main__":

    ###############  Set video file name   ###############

    # VIDEO_PATH = "goldeneye.mp4"
    # VIDEO_PATH = "2girls.mp4"


    # from .txt or .csv file download video
    from scene_detect.scene_util import download_video

    with open("url_20181029", "r") as txtfile:
        url_items =  txtfile.readlines()

    for url_address in url_items:
        video_name = download_video(url_address)
        scene_detect(video_name)
        print(get_iqamos(video_name), "\n")
        print(video_name)



    # ###############  Video Base Information  ###############
    # # import video_baseinfo
    # from scene_detect.video_baseinfo import getBaseInfo
    # print("\n****** 视频文件信息分析： \n")
    # print("%s -> video base information:"%VIDEO_PATH)
    # print("***************************")
    # info = getBaseInfo(VIDEO_PATH)
    # print(info)
    # print("***************************\n")




    #################  Scene Detect & Split #################
    # scene_detect(VIDEO_PATH)
    # print(get_iqamos(VIDEO_PATH))





    #
    # #########################################################
    # #################  import Place Detect  #################
    # # from scene_util import get_files_name
    # from scene_detect.scene_util import get_files_name
    # print("\n****** 开始分析视频的场景： \n")
    # # get scene images
    # current_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), VIDEO_PATH[0:-4]+'/')
    # # print(current_path)
    # files = get_files_name(current_path)
    # print('****** 场景图片的数量:', len(files))
    #
    # # call placeCNN_basic/places_predict() to predict place
    # from place365.run_placesCNN_basic import places_predict
    # from face_detection.face_detection import face_detection
    # from body_analysis.body_analysis import body_attribute, body_numbering, body_segmentation, pose_estimation, gesture_recognition, base64_2binaryimage
    # from text_recognition.aip_OCR import aip_OCR
    # import base64
    #
    # # th architecture to use
    # arch = 'resnet18'
    # i=0
    # for file in files:
    #     i = i + 1
    #     file = os.path.join('./', VIDEO_PATH[0:-4], file)
    #
    #     #####################  Place Detect #####################
    #     print('\n****** Scene %d prediction result:' % i)
    #     places_predict(file, arch)




        # #################  Baidu AiP 5 functions #################
        # with open(file, 'rb') as f:
        #     face_detection(str(base64.b64encode(f.read()),"utf-8"), "BASE64")
        #
        # pose_result = pose_estimation(file)
        # print("人体关键点识别:", pose_result)
        #
        # attr_result = body_attribute(file)
        # print("\n人体属性识别:", attr_result)
        #
        # num_result = body_numbering(file)
        # print("\n人流量统计:", num_result)
        #
        # ges_result = gesture_recognition(file)
        # print("\n手势识别:", ges_result)
        #
        # seg_result = body_segmentation(file)
        # base64_2binaryimage(VIDEO_PATH[0:-4], seg_result)
        # print("\n人像分割:", seg_result, "\n分割图片已保存!")
        #
        # ocr_result = aip_OCR(file)
        # print("\n识别出来的字幕是：", ocr_result)

    # ####################  Yolov3 Detect  ####################
    # print("\n****** 开始分析视频中的物体： \n")
    # from yolov3.yolov3_bridge import yolov3_detect, optparams
    # params = optparams(VIDEO_PATH[0:-4], 'output')
    # params.displayFolders()
    # yolov3_detect(params)
    #
    # #########################################################