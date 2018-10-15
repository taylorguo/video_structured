"""
FFMpeg applied in video frame in Python
The main video  implemenetation.

Copyright (c) 2018 innotechx Shanghai
Licensed under the MIT License (see LICENSE for details)
Written by Guo Yufeng @ 2018.10.10
"""

# fork from project tethys - baseinfo.py

#!/usr/bin/env python
# coding: UTF-8


# Get video information as json, see below json format.

import platform, subprocess, json, os

def getBaseInfo(localvideo):
    baseinfo = { }
    if (os.path.isfile(localvideo)!=True):
        return None
    filesize = os.path.getsize(localvideo)
    if (filesize<=0):
        return None

    if (platform.system()=='Windows'):
        command = ["ffprobe","-loglevel","quiet","-print_format","json","-show_format","-show_streams","-i",localvideo]
    else:
        command = "ffprobe -loglevel quiet -print_format json -show_format -show_streams -i " +  localvideo

    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = result.stdout.read()

    # print(str(out))
    # print(json.loads(out.decode('utf-8')))

    streams = json.loads(out.decode("utf-8"))["streams"]
    for stream in streams:
        if (stream['codec_type']=="video"):
            videoinfo = { }
            videoinfo['width'] = stream['width']
            videoinfo['height'] = stream['height']
            videoinfo['codec_name'] = stream['codec_name']
            videoinfo['duration'] = stream['duration']
            videoinfo['bit_rate'] = stream['bit_rate']
            videoinfo['frame_number'] = stream['nb_frames']
            fr = stream['r_frame_rate']
            fr_list = str(fr).split('/')
            if (len(fr_list) == 2):
                videoinfo['frame_rate'] = float(fr_list[0]) / float(fr_list[1])
            else:
                videoinfo['frame_rate'] = float(fr_list[0])
            baseinfo['video'] = videoinfo
        if (stream['codec_type']=="audio"):
            audioinfo = { }
            audioinfo['duration'] = stream['duration']
            audioinfo['codec_name'] = stream['codec_name']
            audioinfo['bit_rate'] = stream['bit_rate']
            audioinfo['sample_rate'] = stream['sample_rate']
            audioinfo['channels'] = stream['channels']
            baseinfo['audio'] = audioinfo

    return baseinfo

# if __name__ == "__main__":
#     print(getBaseInfo("my_video.mp4"))



'''
# Below code will detect any available video split tool in the system.
#
def split_video_command(ctx, output, filename, high_quality, override_args, quiet, copy,
                        rate_factor, preset):
    """Split input video(s) using ffmpeg or mkvmerge."""
    if ctx.obj.split_video:
        logging.warning('split-video command is specified twice.')
    ctx.obj.check_input_open()
    ctx.obj.split_video = True
    ctx.obj.split_quiet = True if quiet else False
    ctx.obj.split_directory = output
    ctx.obj.split_name_format = filename
    if copy:
        ctx.obj.split_mkvmerge = True
        if high_quality:
            logging.warning('-hq/--high-quality flag ignored due to -c/--copy.')
        if override_args:
            logging.warning('-f/--ffmpeg-args option ignored due to -c/--copy.')
    if not override_args:
        if rate_factor is None:
            rate_factor = 22 if not high_quality else 17
        if preset is None:
            preset = 'veryfast' if not high_quality else 'slow'
        override_args = ('-c:v libx264 -preset {PRESET} -crf {RATE_FACTOR} -c:a copy'.format(
            PRESET=preset, RATE_FACTOR=rate_factor))
    if not copy:
        logging.info('FFmpeg codec args set: %s', override_args)
    if filename:
        logging.info('Video output file name format: %s', filename)
    if ctx.obj.split_directory is not None:
        logging.info('Video output path set:  \n%s', ctx.obj.split_directory)
    ctx.obj.split_args = override_args

    mkvmerge_available = is_mkvmerge_available()
    ffmpeg_available = is_ffmpeg_available()
    if not (mkvmerge_available or ffmpeg_available) or (
            (not mkvmerge_available and copy) or (not ffmpeg_available and not copy)):
        split_tool = 'ffmpeg/mkvmerge'
        if (not mkvmerge_available and copy):
            split_tool = 'mkvmerge'
        elif (not ffmpeg_available and not copy):
            split_tool = 'ffmpeg'
        error_strs = [
            "{EXTERN_TOOL} is required for split-video{EXTRA_ARGS}.".format(
                EXTERN_TOOL=split_tool, EXTRA_ARGS=' -c/--copy' if copy else ''),
            "Install the above tool%s to enable video splitting support." % (
                's' if split_tool.find('/') > 0 else '')]
        if mkvmerge_available:
            error_strs += [
                'You can also specify `split-video -c/--copy` to use mkvmerge for splitting.']
        error_str = '\n'.join(error_strs)
        logging.debug(error_str)
        ctx.obj.options_processed = False
        raise click.BadParameter(error_str, param_hint='split-video')
'''