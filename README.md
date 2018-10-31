<img src="https://www.mobotix.com/sites/default/files/styles/media_image/public/2017-11/Mx_495_Products_MxAnalytics_930x520.jpg" width="200">  

# 功能介绍

项目实现如下功能：

 scene_detect 视频分析主程序

> * 获取视频文件的基本信息
>
> * 通过内容或阈值分析视频场景，进行切割；
> 
> * 对切割后的场景，预测图片场景的场所；
>
> * 调用百度接口，实现的7个功能，如下所示；
>
> * YoloV3 识别多种物体种类和数量。
>
> >  face_detection- 人脸识别
> >
> >  text_recognition- 网络图片中的文字识别，也叫OCR
> >
> >  body_analysis- 人体关键点识别 | 人体属性识别  | 人流量统计  | 手势识别 | 人像分割
                  



# 依赖项

Python 3.6 或以上版本，其他依赖项及其版本在 requirements.txt 有描述。

自行搜索安装 Python 3.6 环境 / 虚拟环境；

安装requirements里面所列依赖项：

 `pip3 install -U -r requirements.txt`

> - `numpy`
> - `torch`
> - `opencv-python`
> - `baidu-aip==2.2.8.0` 
> - 2018.10.10 百度AIP更新到2.2.8， 下载地址：http://ai.baidu.com/sdk#body


# 运行并打印信息

命令行下，进入 yolov3/checkpoints 目录下，运行如下命令，下载预训练的权重参数文件：

`bash download_yolov3_weights.sh`


命令行下或其他开发工具类似环境中，进入项目根目录， 运行如下命令：

`python scene_detect.py`

**信息打印顺序：**

视频文件信息  ->  视频场景切割 -> 视频中场景识别 -> aip接口7功能 -> 视频中的物体识别


## 单一功能测试：

详细键值信息，参考： https://ai.baidu.com/docs#/Face-Detect-V3/top

运行并返回json

>  python face_detection.py
>
>  python aip_OCR.py
>
>  python body_analysis.py


# 联系方式

如有任何问题，请联络： Taylor Guo at taylorguo@126.com
