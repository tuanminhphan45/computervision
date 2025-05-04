# YOLOv7 Pose với Webcam

Dự án này sử dụng YOLOv7 để thực hiện phát hiện tư thế con người (human pose estimation) thông qua webcam theo thời gian thực.

## Yêu Cầu Hệ Thống

- Python 3.8 hoặc cao hơn
- PyTorch
- CUDA (tùy chọn, nhưng khuyến nghị để tăng hiệu suất)
- Webcam

## Cài Đặt

1. Clone repository YOLOv7:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

2. Cài đặt môi trường:
```bash
conda create -n yolov7-pose python=3.8 -y
conda activate yolov7-pose
pip install torch==1.13.1 torchvision==0.14.1 opencv-python matplotlib numpy==1.22.0 tqdm Pillow PyYAML tensorboard seaborn pandas scipy
```

3. Tải mô hình YOLOv7 Pose:
```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
```

4. Tạo file `pose_webcam.py` với nội dung đã cung cấp.

## Sử Dụng

Chạy chương trình phát hiện tư thế với webcam:

```bash
conda activate yolov7-pose
cd yolov7
python pose_webcam.py
```

### Các Phím Tắt

- Nhấn `q` để thoát khỏi chương trình

## Tính Năng

- Phát hiện tư thế con người theo thời gian thực
- Hiển thị khung xương (skeleton) và các điểm nút (keypoints)
- Theo dõi FPS (Frames Per Second)
- Lưu video kết quả vào thư mục `runs/detect`
- Hiển thị độ tin cậy (confidence) của mỗi phát hiện

## Tùy Chỉnh

Bạn có thể điều chỉnh các tham số trong file `pose_webcam.py`:

- `weights`: Đường dẫn đến file mô hình YOLOv7 Pose
- `source`: ID của webcam (thường là 0 cho webcam mặc định)
- `img_size`: Kích thước ảnh đầu vào
- `conf_thres`: Ngưỡng tin cậy cho phát hiện
- `iou_thres`: Ngưỡng IoU cho NMS

## Xử Lý Sự Cố

Nếu gặp lỗi về keypoints, hãy đảm bảo bạn đang sử dụng đúng hàm `non_max_suppression_kpt` thay vì `non_max_suppression` thông thường. YOLOv7 Pose có định dạng đầu ra đặc biệt để xử lý keypoints.

## Tham Khảo

- [YOLOv7 Official Repository](https://github.com/WongKinYiu/yolov7)
- [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) 