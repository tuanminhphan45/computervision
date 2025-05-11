# YOLOv7 Pose with Webcam

This project uses YOLOv7 to perform real-time human pose estimation through webcam.

## System Requirements

- Python 3.8 or higher
- PyTorch
- CUDA (optional, but recommended for better performance)
- Webcam

## Installation

1. Clone YOLOv7 repository:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

2. Set up environment:
```bash
conda create -n yolov7-pose python=3.8 -y
conda activate yolov7-pose
pip install torch==1.13.1 torchvision==0.14.1 opencv-python matplotlib numpy==1.22.0 tqdm Pillow PyYAML tensorboard seaborn pandas scipy
```

3. Download YOLOv7 Pose model:
```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
```

4. Create `pose_webcam.py` file with the provided content.

## Usage

Run the pose detection program with webcam:

```bash
conda activate yolov7-pose
cd yolov7
python pose_webcam.py
```

### Keyboard Shortcuts

- Press `q` to exit the program

## Features

- Real-time human pose detection
- Display skeleton and keypoints
- FPS (Frames Per Second) monitoring
- Save result video in `runs/detect` directory
- Display confidence score for each detection

## Customization

You can adjust parameters in `pose_webcam.py`:

- `weights`: Path to YOLOv7 Pose model file
- `source`: Webcam ID (usually 0 for default webcam)
- `img_size`: Input image size
- `conf_thres`: Confidence threshold for detection
- `iou_thres`: IoU threshold for NMS

## Troubleshooting

If you encounter keypoints errors, make sure you're using the correct `non_max_suppression_kpt` function instead of the regular `non_max_suppression`. YOLOv7 Pose has a special output format for handling keypoints.

## References

- [YOLOv7 Official Repository](https://github.com/WongKinYiu/yolov7)
- [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) 