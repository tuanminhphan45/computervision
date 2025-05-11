import numpy as np
from ..config import DEBUG

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

def detect_squat(kpts, steps=3, knee_angle_threshold=140, conf_threshold=0.3, debug=False):
    """
    Check if person is squatting based on angle between hip-knee-ankle
    
    Returns:
    - is_squatting: True if squatting, False if not
    - knee_angle: Knee angle
    - debug_info: Debug information (dict)
    """
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
        is_squatting = knee_angle < knee_angle_threshold
        
        if debug:
            print(f"Left angle: {left_angle:.1f}, Right angle: {right_angle:.1f}, Threshold: {knee_angle_threshold}")
        
        return is_squatting, knee_angle, debug_info
        
    except Exception as e:
        if debug:
            print(f"Error detecting squat: {e}")
        return False, 0, debug_info 