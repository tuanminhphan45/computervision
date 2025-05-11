import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np

from config import (
    DEBUG, KNEE_ANGLE_THRESHOLD, HIP_RATIO_THRESHOLD,
    MIN_SQUAT_FRAMES, CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHTS, DEFAULT_IMG_SIZE, DEFAULT_CONF_THRES, DEFAULT_IOU_THRES
)
from models.pose_detector import PoseDetector
from utils.angle_utils import detect_squat
from utils.drawing_utils import plot_skeleton_kpts, plot_one_box
from utils.debug_utils import draw_debug_info
from utils.datasets import LoadStreams
from utils.general import increment_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source (video file or webcam index)')
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=DEFAULT_CONF_THRES, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=DEFAULT_IOU_THRES, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--knee-angle', type=int, default=KNEE_ANGLE_THRESHOLD, help='knee angle threshold for squat detection')
    parser.add_argument('--hip-ratio', type=float, default=HIP_RATIO_THRESHOLD, help='hip ratio threshold for squat detection')
    parser.add_argument('--min-squat-frames', type=int, default=MIN_SQUAT_FRAMES, help='minimum frames to confirm a squat')
    parser.add_argument('--conf-kpt', type=float, default=CONFIDENCE_THRESHOLD, help='keypoint confidence threshold')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--cpu-only', action='store_true', help='force CPU processing')
    return parser.parse_args()

def run():
    # Parse arguments
    opt = parse_args()
    
    # Update global variables
    global DEBUG, KNEE_ANGLE_THRESHOLD, HIP_RATIO_THRESHOLD, MIN_SQUAT_FRAMES, CONFIDENCE_THRESHOLD
    DEBUG = opt.debug
    KNEE_ANGLE_THRESHOLD = opt.knee_angle
    HIP_RATIO_THRESHOLD = opt.hip_ratio
    MIN_SQUAT_FRAMES = opt.min_squat_frames
    CONFIDENCE_THRESHOLD = opt.conf_kpt
    
    if DEBUG:
        print(f"DEBUG MODE: Enabled")
        print(f"Knee angle threshold: {KNEE_ANGLE_THRESHOLD}")
        print(f"Hip ratio threshold: {HIP_RATIO_THRESHOLD}")
        print(f"Minimum frames for squat: {MIN_SQUAT_FRAMES}")
        print(f"Keypoint confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Create directory to save results
    save_dir = Path(increment_path(Path('runs/detect'), exist_ok=False))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pose detector
    detector = PoseDetector(
        weights=opt.weights,
        img_size=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        cpu_only=opt.cpu_only
    )
    
    # Setup webcam
    dataset = LoadStreams(opt.source, img_size=detector.img_size, stride=detector.stride)
    
    # Variables for FPS tracking
    frame_count = 0
    fps = 0
    prev_time = time.time()
    
    # Variables for squat counting and state
    squat_count = 0
    is_squatting_prev = False
    squat_frames = 0
    
    # Variables for error tracking
    error_count = 0
    max_errors = 5
    
    print(f"YOLOv7 Pose Detection ready! Press 'q' to exit.")
    
    try:
        for path, img, im0s, vid_cap in dataset:
            frame_count += 1
            
            try:
                # Detect poses
                output, img = detector.detect(img)
                
                # Calculate FPS
                curr_time = time.time()
                if curr_time - prev_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    prev_time = curr_time
                    
                # Process detection results
                for i, det in enumerate(output):
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        
                        # Rescale keypoints
                        if det.shape[1] > 6:
                            num_kpts = (det.shape[1] - 6) // 3
                            for k in range(num_kpts):
                                det[:, 6+k*3] = det[:, 6+k*3] * im0.shape[1] / img.shape[3]
                                det[:, 6+k*3+1] = det[:, 6+k*3+1] * im0.shape[0] / img.shape[2]
                        
                        # Process first detected person
                        if len(det) > 0:
                            *xyxy, conf, cls = det[0, :6]
                            kpts = det[0, 6:].cpu().numpy() if det.shape[1] > 6 else None
                            
                            if kpts is not None:
                                # Draw skeleton
                                plot_skeleton_kpts(im0, kpts, 3)
                                
                                # Detect squat
                                is_squatting, knee_angle, debug_info = detect_squat(
                                    kpts, 3,
                                    knee_angle_threshold=KNEE_ANGLE_THRESHOLD,
                                    conf_threshold=CONFIDENCE_THRESHOLD,
                                    debug=DEBUG
                                )
                                
                                # Track squat state
                                if is_squatting:
                                    squat_frames += 1
                                else:
                                    if is_squatting_prev and squat_frames >= MIN_SQUAT_FRAMES:
                                        squat_count += 1
                                    squat_frames = 0
                                    
                                is_squatting_prev = is_squatting
                                
                                # Display status
                                status_text = "Squatting" if is_squatting else "Standing"
                                status_color = (0, 0, 255) if is_squatting else (0, 255, 0)
                                
                                cv2.putText(im0, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                                cv2.putText(im0, f"Knee angle: {int(knee_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                if squat_frames > 0:
                                    cv2.putText(im0, f"Squat frames: {squat_frames}/{MIN_SQUAT_FRAMES}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                                
                                # Draw debug info
                                if DEBUG:
                                    im0 = draw_debug_info(im0, debug_info, KNEE_ANGLE_THRESHOLD)
                            
                            # Draw bounding box
                            label = f'person {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=2)
                    
                    # Display FPS and squat count
                    cv2.putText(im0, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(im0, f'Squat count: {squat_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Add instructions
                    cv2.putText(im0, f'Press q to exit', (10, im0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display image
                    if opt.view_img:
                        cv2.imshow(str(p), im0)
                        if cv2.waitKey(1) == ord('q'):
                            cv2.destroyAllWindows()
                            return
                            
                error_count = 0
                        
            except KeyboardInterrupt:
                print("Received exit command")
                break
                
            except Exception as e:
                error_count += 1
                if DEBUG:
                    print(f"Error processing frame: {e}")
                if error_count > max_errors:
                    print(f"Too many errors, exiting")
                    break
        
    except KeyboardInterrupt:
        print("Received exit command")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        cv2.destroyAllWindows()
        print(f'Done. ({time.time() - t0:.3f}s)')
        print(f'Total squats: {squat_count}')

if __name__ == '__main__':
    run() 