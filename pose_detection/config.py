# Global configuration variables
DEBUG = False
KNEE_ANGLE_THRESHOLD = 140
HIP_RATIO_THRESHOLD = 1.5
MIN_SQUAT_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.3

# Model configuration
DEFAULT_WEIGHTS = 'yolov7-w6-pose.pt'
DEFAULT_IMG_SIZE = 384
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45

# Drawing configuration
SKELETON_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102),
    (230, 230, 0), (255, 153, 255), (153, 204, 255),
    (255, 102, 255), (255, 51, 255), (102, 178, 255),
    (51, 153, 255), (255, 153, 153), (255, 102, 102),
    (255, 51, 51), (153, 255, 153), (102, 255, 102),
    (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 255)
]

# Skeleton connections
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
] 