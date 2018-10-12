# **Video_Structured**
Detect a video file to output structure. 

------------------------------------------
## **使用百度AI接口实现如下功能**

face_detection- 人脸识别

text_recognition- 网络图片中的文字识别，也叫OCR

body_analysis- 人体关键点识别 | 人体属性识别  | 人流量统计  | 手势识别 | 人像分割
                  
scene_detect- 通过内容或阈值分析视频场景，进行分割

------------------------------------------
## **安装 Python 3.6 环境 /虚拟环境**

项目目录下执行： pip install -r requirements.txt 


## **依赖项**
### 2018.10.10 百度AIP更新到2.2.8
下载地址：http://ai.baidu.com/sdk#body

baidu-aip==2.2.8.0

------------------------------------------
## **运行并返回json**
详细键值信息，参考： https://ai.baidu.com/docs#/Face-Detect-V3/top

python face_detection.py

python aip_OCR.py

python body_analysis.py

python scene_detect.py