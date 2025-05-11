import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression_kpt, scale_coords
from utils.torch_utils import select_device

class PoseDetector:
    def __init__(self, weights, img_size=384, conf_thres=0.25, iou_thres=0.45, device='', cpu_only=False):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize device
        if cpu_only:
            self.device = select_device('cpu')
        else:
            self.device = select_device(device)
            
        # Load model
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        
        # Set model to half precision if using GPU
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
            
        # Run inference once
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
            
        # Set cudnn benchmark
        cudnn.benchmark = True
        
    def detect(self, img):
        """
        Detect poses in image
        Returns:
        - output: Detection results
        - img: Processed image
        """
        # Prepare image
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        with torch.no_grad():
            output, _ = self.model(img)
            
        # Apply NMS
        output = non_max_suppression_kpt(
            output, 
            self.conf_thres, 
            self.iou_thres, 
            nc=self.model.yaml['nc'], 
            nkpt=self.model.yaml['nkpt'], 
            kpt_label=True
        )
        
        return output, img 