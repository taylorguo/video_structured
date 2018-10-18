#!/usr/bin/env bash

# Start
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 train.py -img_size 416 -epochs 160

# Resume
python3 train.py -img_size 416 -resume 1

# Detect
gsutil cp gs://ultralytics/fresh9_5_e201.pt yolov3/checkpoints
python3 detect.py

# Test
python3 test.py -img_size 416 -weights_path checkpoints/latest.pt -conf_thresh 0.5


# Download and Test
sudo rm -rf yolov3 && git clone https://github.com/ultralytics/yolov3
cd yolov3
cd checkpoints
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
python3 test.py -img_size 416 -weights_path checkpoints/backup5.pt -nms_thres 0.45
