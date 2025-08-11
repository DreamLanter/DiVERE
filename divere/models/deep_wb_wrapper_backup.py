"""
Deep White Balance Wrapper for DiVERE
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional
from PIL import Image

# Add the Deep White Balance path
DEEP_WB_PATH = os.path.join(os.path.dirname(__file__), 'Deep_White_Balance', 'PyTorch')
sys.path.append(DEEP_WB_PATH)

try:
    from arch import deep_wb_single_task
    from utilities.deepWB import deep_wb
    from utilities import utils as utls
    DEEP_WB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deep White Balance not available: {e}")
    DEEP_WB_AVAILABLE = False

# 使用绝对导入
try:
    from divere.models.utils.gain_calculator import calculate_rgb_gains_from_images, estimate_illuminant_from_gains
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    try:
        from .utils.gain_calculator import calculate_rgb_gains_from_images, estimate_illuminant_from_gains
    except ImportError:
        print("Warning: Cannot import gain calculator utilities")
        DEEP_WB_AVAILABLE = False


class DeepWBWrapper:
    """Wrapper for Deep White Balance model"""
    
    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize Deep White Balance wrapper
        
        Args:
            model_dir: Directory containing the trained models
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if not DEEP_WB_AVAILABLE:
            raise ImportError("Deep White Balance models are not available")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir or os.path.join(DEEP_WB_PATH, 'models')
        self.net_awb = None
        self._load_model()
    
    def _load_model(self):
        """Load the AWB model"""
        model_path = os.path.join(self.model_dir, 'net_awb.pth')
        
        if not os.path.exists(model_path):
            # Try the combined model
            combined_model_path = os.path.join(self.model_dir, 'net.pth')
            if os.path.exists(combined_model_path):
                from arch import deep_wb_model
                from arch import splitNetworks as splitter
                
                net = deep_wb_model.deepWBNet()
                net.load_state_dict(torch.load(combined_model_path, map_location=self.device))
                net_awb, _, _ = splitter.splitNetworks(net)
                self.net_awb = net_awb
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {combined_model_path}")
        else:
            self.net_awb = deep_wb_single_task.deepWBnet()
            self.net_awb.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.net_awb.to(self.device)
        self.net_awb.eval()
    
    def process_image(self, image: np.ndarray, max_size: int = 656) -> np.ndarray:
        """
        Process an image using Deep White Balance AWB
        
        Args:
            image: Input image array (H, W, 3) in range [0, 255]
            max_size: Maximum dimension for processing
            
        Returns:
            Processed image array (H, W, 3) in range [0, 255]
        """
        if not DEEP_WB_AVAILABLE:
            raise ImportError("Deep White Balance models are not available")
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Process using Deep White Balance
        with torch.no_grad():
            output_awb = deep_wb(pil_image, task='awb', net_awb=self.net_awb, 
                                device=self.device, s=max_size)
        
        # Convert back to numpy array
        output_array = (output_awb * 255).astype(np.uint8)
        
        return output_array
    
    def calculate_auto_gain(self, image: np.ndarray, 
                           method: str = 'simple_ratio') -> Tuple[float, float, float, float, float, float]:
        """
        Calculate auto gain using Deep White Balance
        
        Args:
            image: Input image array (H, W, 3) in range [0, 255]
            method: Method to calculate gains ('simple_ratio', 'log_ratio', 'linear_mapping')
            
        Returns:
            Tuple of (r_gain, g_gain, b_gain, r_illuminant, g_illuminant, b_illuminant)
        """
        # Normalize image to [0, 1] range
        input_image = image.astype(np.float32) / 255.0
        
        # Process image
        output_array = self.process_image(image)
        output_image = output_array.astype(np.float32) / 255.0
        
        # Calculate RGB gains
        gains = calculate_rgb_gains_from_images(input_image, output_image, method)
        
        # Estimate illuminant from gains
        illuminant = estimate_illuminant_from_gains(gains)
        
        return gains + illuminant


def create_deep_wb_wrapper(model_dir: Optional[str] = None, device: str = 'cpu') -> Optional[DeepWBWrapper]:
    """
    Create Deep White Balance wrapper if available
    
    Args:
        model_dir: Directory containing the trained models
        device: Device to run the model on
        
    Returns:
        DeepWBWrapper instance or None if not available
    """
    if not DEEP_WB_AVAILABLE:
        return None
    
    try:
        return DeepWBWrapper(model_dir, device)
    except Exception as e:
        print(f"Failed to create Deep White Balance wrapper: {e}")
        return None 