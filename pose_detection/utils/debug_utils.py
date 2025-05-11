import cv2

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