import argparse
import time
from pathlib import Path
import random
import math
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression_kpt, scale_coords, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# Global variables setup
DEBUG = False
KNEE_ANGLE_THRESHOLD = 140
HIP_RATIO_THRESHOLD = 1.5
MIN_SQUAT_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.3

# Function to draw skeleton and keypoints with error handling
def plot_skeleton_kpts(im, kpts, steps=3):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        # Fixed comparison to avoid errors
        if steps == 3:
            conf = kpts[steps * kid + 2]
            if conf < 0.5:
                continue
        
        # Check if x_coord and y_coord are tensor elements
        if isinstance(x_coord, np.ndarray):
            if x_coord.size > 0 and y_coord.size > 0:
                x_int, y_int = int(x_coord.item()), int(y_coord.item())
                cv2.circle(im, (x_int, y_int), radius, (int(r), int(g), int(b)), -1)
        else:
            x_int, y_int = int(x_coord), int(y_coord)
            if x_int != 0 and y_int != 0:  # Instead of modulo comparison
                cv2.circle(im, (x_int, y_int), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        
        # Get positions of 2 keypoints for the connection
        pos1_idx, pos2_idx = (sk[0]-1)*steps, (sk[1]-1)*steps
        
        # Check for valid indices
        if pos1_idx >= len(kpts) or pos2_idx >= len(kpts) or pos1_idx+1 >= len(kpts) or pos2_idx+1 >= len(kpts):
            continue
            
        # Get x, y coordinates of the 2 keypoints
        x1, y1 = kpts[pos1_idx], kpts[pos1_idx+1]
        x2, y2 = kpts[pos2_idx], kpts[pos2_idx+1]
        
        # Get confidence for keypoints
        if steps == 3 and pos1_idx+2 < len(kpts) and pos2_idx+2 < len(kpts):
            conf1 = kpts[pos1_idx+2]
            conf2 = kpts[pos2_idx+2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
                
        # Convert coordinates to int
        x1_int, y1_int = int(x1), int(y1)
        x2_int, y2_int = int(x2), int(y2)
        
        # Check for valid coordinates
        if x1_int == 0 or y1_int == 0 or x1_int < 0 or y1_int < 0:
            continue
        if x2_int == 0 or y2_int == 0 or x2_int < 0 or y2_int < 0:
            continue
            
        # Draw connection line between 2 keypoints
        cv2.line(im, (x1_int, y1_int), (x2_int, y2_int), (int(r), int(g), int(b)), thickness=2)
        
    return im


# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    """
    Calculate angle between three points with b as the vertex
    a, b, c are points in the form [x, y]
    Returns angle in range 0-180 degrees
    """
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Check input data
        if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
            return 180  # Return default angle if there are NaN values
        
        # Check for duplicated points
        if np.array_equal(a, b) or np.array_equal(b, c) or np.array_equal(a, c):
            return 180
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Check for zero vectors
        if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
            return 180
        
        # Calculate cosine angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Ensure cosine value is in valid range [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert from radians to degrees
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
        
        return angle
    except Exception as e:
        if DEBUG:
            print(f"Error calculating angle: {e}")
        return 180  # Return default angle if there's an error


# Improved squat detection function
def detect_squat(kpts, steps=3, knee_angle_threshold=140, conf_threshold=0.3, debug=False):
    """
    Check if person is squatting based on angle between hip-knee-ankle
    
    Returns:
    - is_squatting: True if squatting, False if not
    - knee_angle: Knee angle
    - debug_info: Debug information (dict)
    """
    # Define keypoints needed to calculate angle
    # COCO keypoint indices: 
    # 11: left hip, 12: right hip, 
    # 13: left knee, 14: right knee, 
    # 15: left ankle, 16: right ankle

    # Initialize debug info
    debug_info = {
        "left_detected": False,
        "right_detected": False,
        "left_angle": 180,
        "right_angle": 180,
        "left_conf": 0,
        "right_conf": 0
    }

    # Check if there are enough keypoints
    if len(kpts) < 17*steps:
        return False, 0, debug_info

    try:
        # Get left side keypoint coordinates
        left_hip = [kpts[11*steps], kpts[11*steps+1]]
        left_knee = [kpts[13*steps], kpts[13*steps+1]]
        left_ankle = [kpts[15*steps], kpts[15*steps+1]]
        
        # Get right side keypoint coordinates
        right_hip = [kpts[12*steps], kpts[12*steps+1]]
        right_knee = [kpts[14*steps], kpts[14*steps+1]]
        right_ankle = [kpts[16*steps], kpts[16*steps+1]]
        
        # Get confidence values
        left_hip_conf = kpts[11*steps+2] if steps == 3 else 1.0
        right_hip_conf = kpts[12*steps+2] if steps == 3 else 1.0
        left_knee_conf = kpts[13*steps+2] if steps == 3 else 1.0
        right_knee_conf = kpts[14*steps+2] if steps == 3 else 1.0
        left_ankle_conf = kpts[15*steps+2] if steps == 3 else 1.0
        right_ankle_conf = kpts[16*steps+2] if steps == 3 else 1.0
        
        # Update debug info
        debug_info["left_conf"] = min(left_hip_conf, left_knee_conf, left_ankle_conf)
        debug_info["right_conf"] = min(right_hip_conf, right_knee_conf, right_ankle_conf)
        
        # Calculate angle between hip-knee-ankle for both sides
        left_angle = 180
        right_angle = 180
        
        # Check left side detection
        if left_hip_conf > conf_threshold and left_knee_conf > conf_threshold and left_ankle_conf > conf_threshold:
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            debug_info["left_detected"] = True
            debug_info["left_angle"] = left_angle
            
        # Check right side detection
        if right_hip_conf > conf_threshold and right_knee_conf > conf_threshold and right_ankle_conf > conf_threshold:
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            debug_info["right_detected"] = True
            debug_info["right_angle"] = right_angle
        
        # Check if at least one side is detected
        if not debug_info["left_detected"] and not debug_info["right_detected"]:
            return False, 0, debug_info
        
        # Take smaller angle if both sides are detected
        # Or take the detected side's angle
        if debug_info["left_detected"] and debug_info["right_detected"]:
            knee_angle = min(left_angle, right_angle)
        elif debug_info["left_detected"]:
            knee_angle = left_angle
        else:
            knee_angle = right_angle
        
        # Determine squat state based on knee angle
        # If angle is less than threshold, consider it a squat
        is_squatting = knee_angle < knee_angle_threshold
        
        if debug:
            print(f"Left angle: {left_angle:.1f}, Right angle: {right_angle:.1f}, Threshold: {knee_angle_threshold}")
        
        return is_squatting, knee_angle, debug_info
        
    except Exception as e:
        if debug:
            print(f"Error detecting squat: {e}")
        return False, 0, debug_info

# Add function to display debug info
def draw_debug_info(img, debug_info, knee_angle_threshold):
    """
    Draw debug information on the image
    """
    h, w = img.shape[:2]
    # Draw debug panel in bottom right corner
    cv2.rectangle(img, (w-300, h-200), (w-10, h-10), (0, 0, 0), -1)
    cv2.rectangle(img, (w-300, h-200), (w-10, h-10), (255, 255, 255), 2)
    
    # Title
    cv2.putText(img, "DEBUG INFO", (w-290, h-175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Left leg info
    left_color = (0, 255, 0) if debug_info["left_detected"] else (0, 0, 255)
    cv2.putText(img, f"Left leg: {debug_info['left_detected']}", (w-290, h-150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 1)
    cv2.putText(img, f"Left angle: {debug_info['left_angle']:.1f}", (w-290, h-130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 1)
    cv2.putText(img, f"Left conf: {debug_info['left_conf']:.2f}", (w-290, h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 1)
    
    # Right leg info
    right_color = (0, 255, 0) if debug_info["right_detected"] else (0, 0, 255)
    cv2.putText(img, f"Right leg: {debug_info['right_detected']}", (w-290, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 1)
    cv2.putText(img, f"Right angle: {debug_info['right_angle']:.1f}", (w-290, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 1)
    cv2.putText(img, f"Right conf: {debug_info['right_conf']:.2f}", (w-290, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 1)
    
    # Threshold info
    cv2.putText(img, f"Threshold: {knee_angle_threshold}", (w-290, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img

def run():
    global DEBUG, KNEE_ANGLE_THRESHOLD, HIP_RATIO_THRESHOLD, MIN_SQUAT_FRAMES, CONFIDENCE_THRESHOLD
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7-w6-pose.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source (video file or webcam index)')
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--knee-angle', type=int, default=140, help='knee angle threshold for squat detection')
    parser.add_argument('--hip-ratio', type=float, default=1.5, help='hip ratio threshold for squat detection')
    parser.add_argument('--min-squat-frames', type=int, default=5, help='minimum frames to confirm a squat')
    parser.add_argument('--conf-kpt', type=float, default=0.3, help='keypoint confidence threshold')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--cpu-only', action='store_true', help='force CPU processing')
    opt = parser.parse_args()
    
    # Setup input parameters
    weights = opt.weights
    source = opt.source  # webcam, video file
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    view_img = True
    save_dir = Path('runs/detect/exp')
    
    # Setup squat detection parameters
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
    
    # Initialize device
    if opt.cpu_only:
        device = select_device('cpu')
    else:
        device = select_device(opt.device)
    
    # Check and free GPU memory if needed
    if device.type != 'cpu':
        try:
            # Free GPU memory
            torch.cuda.empty_cache()
            # Calculate safe memory limit
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if DEBUG:
                print(f"GPU memory: {gpu_memory/1024**3:.2f} GB")
            
            # Automatically adjust image size based on GPU memory
            if gpu_memory < 4 * 1024**3:  # If less than 4GB
                img_size = min(img_size, 320)
                if DEBUG:
                    print(f"GPU memory < 4GB, reducing img_size to {img_size}")
            elif gpu_memory < 6 * 1024**3:  # If less than 6GB
                img_size = min(img_size, 384)
                if DEBUG:
                    print(f"GPU memory < 6GB, reducing img_size to {img_size}")
            elif gpu_memory < 8 * 1024**3:  # If less than 8GB
                img_size = min(img_size, 512)
                if DEBUG:
                    print(f"GPU memory < 8GB, reducing img_size to {img_size}")
        except Exception as e:
            if DEBUG:
                print(f"Error checking GPU memory: {e}")
            device = select_device('cpu')
            print("Switched to CPU mode due to GPU error")
    
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    try:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        img_size = check_img_size(img_size, s=stride)  # check img_size
        
        if half:
            model.half()  # to FP16
            
        if DEBUG:
            print(f"Model loaded successfully, stride={stride}, img_size={img_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the file {weights} exists and is in the correct format")
        return
        
    # Setup webcam
    cudnn.benchmark = True  # set True to speed up processing
    dataset = LoadStreams(source, img_size=img_size, stride=stride)
    
    # Perform inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()
    
    # Variables for FPS tracking
    frame_count = 0
    fps = 0
    prev_time = time.time()
    
    # Create random color
    color = (0, 255, 0)  # Green color
    
    # Variables for squat counting and state
    squat_count = 0
    is_squatting_prev = False
    squat_start_time = None
    squat_duration = 0
    squat_frames = 0
    
    # Variables for error tracking
    error_count = 0
    max_errors = 5
    
    print(f"YOLOv7 Pose Detection ready! Press 'q' to exit.")
    
    try:
        for path, img, im0s, vid_cap in dataset:
            frame_count += 1
            
            try:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                t1 = time_synchronized()
                with torch.no_grad():
                    output, _ = model(img)
                    
                # Apply special NMS for pose
                output = non_max_suppression_kpt(output, conf_thres, iou_thres, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                t2 = time_synchronized()
                
                # Calculate FPS
                curr_time = time.time()
                if curr_time - prev_time >= 1.0:  # Update FPS every second
                    fps = frame_count
                    frame_count = 0
                    prev_time = curr_time
                    
                # Process detection results
                for i, det in enumerate(output):  # for each detection result
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        
                        # Rescale keypoints (5 is the number of elements before kpt)
                        if det.shape[1] > 6:
                            num_kpts = (det.shape[1] - 6) // 3  # number of keypoints (each keypoint has 3 values: x, y, conf)
                            for k in range(num_kpts):
                                # Rescale x and y of each keypoint
                                det[:, 6+k*3] = det[:, 6+k*3] * im0.shape[1] / img.shape[3]  # rescale x
                                det[:, 6+k*3+1] = det[:, 6+k*3+1] * im0.shape[0] / img.shape[2]  # rescale y
                        
                        # Draw results and count squats - only consider the first detected person
                        if len(det) > 0:
                            # Get first result (assume it's the main person)
                            *xyxy, conf, cls = det[0, :6]
                            
                            # Keypoints are elements from 6 to end in the first row
                            kpts = det[0, 6:].cpu().numpy() if det.shape[1] > 6 else None
                            
                            # Draw skeleton and keypoints
                            if kpts is not None:
                                plot_skeleton_kpts(im0, kpts, 3)
                                
                                # Detect squat state
                                is_squatting, knee_angle, debug_info = detect_squat(
                                    kpts, 3, 
                                    knee_angle_threshold=KNEE_ANGLE_THRESHOLD, 
                                    conf_threshold=CONFIDENCE_THRESHOLD,
                                    debug=DEBUG
                                )
                                
                                # Track consecutive frames in squat state
                                if is_squatting:
                                    squat_frames += 1
                                else:
                                    # If transitioning from squat to standing and had enough squat frames
                                    if is_squatting_prev and squat_frames >= MIN_SQUAT_FRAMES:
                                        squat_count += 1
                                    squat_frames = 0
                                    
                                # Update previous state
                                is_squatting_prev = is_squatting
                                
                                # Display status and knee angle
                                status_text = "Squatting" if is_squatting else "Standing"
                                status_color = (0, 0, 255) if is_squatting else (0, 255, 0)
                                
                                cv2.putText(im0, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                                cv2.putText(im0, f"Knee angle: {int(knee_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                if squat_frames > 0:
                                    cv2.putText(im0, f"Squat frames: {squat_frames}/{MIN_SQUAT_FRAMES}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                                
                                # Draw debug info if enabled
                                if DEBUG:
                                    im0 = draw_debug_info(im0, debug_info, KNEE_ANGLE_THRESHOLD)
                            
                            # Draw bounding box
                            label = f'person {conf:.2f}'
                            xyxy_list = [float(x) for x in xyxy]
                            plot_one_box(xyxy_list, im0, label=label, color=color, line_thickness=2)
                    
                    # Display FPS and squat count
                    cv2.putText(im0, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(im0, f'Squat count: {squat_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Add instructions
                    cv2.putText(im0, f'Press q to exit', (10, im0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display image with results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        
                        # Press 'q' to exit
                        if cv2.waitKey(1) == ord('q'):
                            cv2.destroyAllWindows()
                            return
                            
                error_count = 0  # Reset error after each successful frame
                        
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
    # Initialize global variables
    DEBUG = False
    
    run()
