# YOLOv7 Pose Detection with Squat Counter

This project uses YOLOv7 to perform real-time human pose estimation and squat counting through webcam.

## Project Structure

```
pose_detection/
├── __init__.py
├── config.py              # Global configuration variables
├── main.py               # Main program entry point
├── utils/
│   ├── __init__.py
│   ├── angle_utils.py    # Angle calculation and squat detection
│   ├── drawing_utils.py  # Skeleton and bounding box drawing
│   └── debug_utils.py    # Debug information display
└── models/
    ├── __init__.py
    └── pose_detector.py  # YOLOv7 pose detection model wrapper
```

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

4. Copy the model file to the project directory:
```bash
cp yolov7-w6-pose.pt pose_detection/
```

## Usage

Run the pose detection program with webcam:

```bash
conda activate yolov7-pose
cd pose_detection
python main.py
```

### Command Line Arguments

- `--weights`: Path to YOLOv7 Pose model file (default: 'yolov7-w6-pose.pt')
- `--source`: Webcam ID or video file path (default: '0')
- `--img-size`: Input image size (default: 384)
- `--conf-thres`: Confidence threshold for detection (default: 0.25)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)
- `--device`: CUDA device (default: '')
- `--knee-angle`: Knee angle threshold for squat detection (default: 140)
- `--min-squat-frames`: Minimum frames to confirm a squat (default: 5)
- `--conf-kpt`: Keypoint confidence threshold (default: 0.3)
- `--debug`: Enable debug mode
- `--cpu-only`: Force CPU processing

### Keyboard Shortcuts

- Press `q` to exit the program

## Features

- Real-time human pose detection
- Squat counting with configurable thresholds
- Display skeleton and keypoints
- FPS (Frames Per Second) monitoring
- Save result video in `runs/detect` directory
- Display confidence score for each detection
- Debug mode with detailed information display

## Customization

You can adjust various parameters in `config.py`:

- `KNEE_ANGLE_THRESHOLD`: Threshold for squat detection (default: 140)
- `MIN_SQUAT_FRAMES`: Minimum frames to confirm a squat (default: 5)
- `CONFIDENCE_THRESHOLD`: Keypoint confidence threshold (default: 0.3)
- `DEFAULT_IMG_SIZE`: Default input image size (default: 384)
- `DEFAULT_CONF_THRES`: Default confidence threshold (default: 0.25)
- `DEFAULT_IOU_THRES`: Default IoU threshold (default: 0.45)

## Troubleshooting

1. If you encounter keypoints errors:
   - Make sure you're using the correct `non_max_suppression_kpt` function
   - Check if the model file is correctly downloaded and placed
   - Verify CUDA installation if using GPU

2. If the program runs slowly:
   - Try reducing the input image size
   - Use CPU mode if GPU memory is insufficient
   - Close other GPU-intensive applications

3. If squat detection is inaccurate:
   - Adjust the knee angle threshold
   - Increase the minimum squat frames
   - Enable debug mode to see detailed information

## References

- [YOLOv7 Official Repository](https://github.com/WongKinYiu/yolov7)
- [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) 