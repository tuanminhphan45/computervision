import cv2
import numpy as np
from ..config import SKELETON, SKELETON_COLORS

def plot_skeleton_kpts(im, kpts, steps=3):
    """
    Draw skeleton and keypoints on image
    """
    pose_limb_color = SKELETON_COLORS[9:9+len(SKELETON)]
    pose_kpt_color = SKELETON_COLORS[16:16+17]  # 17 keypoints
    radius = 5
    num_kpts = len(kpts) // steps

    # Draw keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if steps == 3:
            conf = kpts[steps * kid + 2]
            if conf < 0.5:
                continue
        
        if isinstance(x_coord, np.ndarray):
            if x_coord.size > 0 and y_coord.size > 0:
                x_int, y_int = int(x_coord.item()), int(y_coord.item())
                cv2.circle(im, (x_int, y_int), radius, (int(r), int(g), int(b)), -1)
        else:
            x_int, y_int = int(x_coord), int(y_coord)
            if x_int != 0 and y_int != 0:
                cv2.circle(im, (x_int, y_int), radius, (int(r), int(g), int(b)), -1)

    # Draw skeleton connections
    for sk_id, sk in enumerate(SKELETON):
        r, g, b = pose_limb_color[sk_id]
        
        pos1_idx, pos2_idx = (sk[0]-1)*steps, (sk[1]-1)*steps
        
        if pos1_idx >= len(kpts) or pos2_idx >= len(kpts) or pos1_idx+1 >= len(kpts) or pos2_idx+1 >= len(kpts):
            continue
            
        x1, y1 = kpts[pos1_idx], kpts[pos1_idx+1]
        x2, y2 = kpts[pos2_idx], kpts[pos2_idx+1]
        
        if steps == 3 and pos1_idx+2 < len(kpts) and pos2_idx+2 < len(kpts):
            conf1 = kpts[pos1_idx+2]
            conf2 = kpts[pos2_idx+2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
                
        x1_int, y1_int = int(x1), int(y1)
        x2_int, y2_int = int(x2), int(y2)
        
        if x1_int == 0 or y1_int == 0 or x1_int < 0 or y1_int < 0:
            continue
        if x2_int == 0 or y2_int == 0 or x2_int < 0 or y2_int < 0:
            continue
            
        cv2.line(im, (x1_int, y1_int), (x2_int, y2_int), (int(r), int(g), int(b)), thickness=2)
        
    return im

def plot_one_box(x, img, color=(0, 200, 0), label=None, line_thickness=3):
    """
    Draw bounding box on image
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA) 