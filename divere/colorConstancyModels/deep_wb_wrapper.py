"""
Deep White Balance Wrapper for DiVERE - ONNX Version
Independent implementation without Deep_White_Balance submodule dependency
"""

import os
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from sklearn.linear_model import LinearRegression

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ONNX Runtime not available: {e}")
    ONNX_AVAILABLE = False


def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1) """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result


def preprocess_image_for_onnx(image: np.ndarray, max_size: int = 656) -> tuple:
    """
    预处理图像用于 ONNX 推理
    
    Args:
        image: 输入图像数组 (H, W, 3) 范围 [0, 255]
        max_size: 处理的最大尺寸
        
    Returns:
        (预处理后的图像, 原始尺寸)
    """
    # 转换为 PIL Image 进行预处理
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    # 调整尺寸
    w, h = pil_image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h
    
    # 确保尺寸是 16 的倍数（模型要求）
    if w % 16 != 0:
        w = w + (16 - w % 16)
    if h % 16 != 0:
        h = h + (16 - h % 16)
        
    if pil_image.size != (w, h):
        pil_image = pil_image.resize((w, h), Image.Resampling.LANCZOS)
    
    # 转换为 numpy 数组并归一化
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    
    # 转换为 CHW 格式
    img_chw = img_array.transpose(2, 0, 1)
    
    # 添加 batch 维度
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch, (image.shape[1], image.shape[0])  # 返回原始尺寸 (width, height)


def postprocess_onnx_output(output: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    后处理 ONNX 输出
    
    Args:
        output: ONNX 模型输出 (1, 3, H, W)
        original_size: 原始图像尺寸 (width, height)
        
    Returns:
        后处理后的图像数组 (H, W, 3) 范围 [0, 255]
    """
    # 移除 batch 维度并转换为 HWC 格式
    output_hwc = output[0].transpose(1, 2, 0)
    
    # 裁剪到 [0, 1] 范围
    output_hwc = np.clip(output_hwc, 0, 1)
    
    # 转换为 [0, 255] 范围
    output_255 = (output_hwc * 255).astype(np.uint8)
    
    # 调整回原始尺寸
    if output_255.shape[:2] != (original_size[1], original_size[0]):
        pil_output = Image.fromarray(output_255)
        pil_output = pil_output.resize(original_size, Image.Resampling.LANCZOS)
        output_255 = np.array(pil_output)
    
    return output_255


class DeepWBWrapper:
    """ONNX-based Deep White Balance wrapper - Independent implementation"""
    
    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize Deep White Balance wrapper
        
        Args:
            model_dir: Directory containing the ONNX model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available")
        
        # 设置模型路径
        if model_dir is None:
            self.model_dir = os.path.dirname(__file__)
        else:
            self.model_dir = model_dir
            
        self.onnx_model_path = os.path.join(self.model_dir, 'net_awb.onnx')
        
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded from: {self.onnx_model_path}")
        print(f"Using providers: {self.session.get_providers()}")
    
    def process_image(self, image: np.ndarray, max_size: int = 656) -> np.ndarray:
        """
        Process an image using Deep White Balance AWB
        
        Args:
            image: Input image array (H, W, 3) in range [0, 255]
            max_size: Maximum dimension for processing
            
        Returns:
            Processed image array (H, W, 3) in range [0, 255]
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available")
        
        # 预处理
        input_tensor, original_size = preprocess_image_for_onnx(image, max_size)
        
        # ONNX 推理
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        # 后处理
        result = postprocess_onnx_output(output, original_size)
        
        return result
    
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
        
        # Calculate RGB gains using simple ratio method
        if method == 'simple_ratio':
            # Calculate mean values for each channel
            input_means = np.mean(input_image, axis=(0, 1))
            output_means = np.mean(output_image, axis=(0, 1))
            
            # Avoid division by zero
            input_means = np.maximum(input_means, 1e-6)
            
            # Calculate gains as ratio
            gains = output_means / input_means
            
            # Convert to log space
            log_gains = np.log10(gains)
            
            # Clip gains to reasonable range
            log_gains = np.clip(log_gains, -1.0, 1.0)
            
            r_gain, g_gain, b_gain = log_gains
            
        elif method == 'linear_mapping':
            # Use polynomial mapping method
            mapping_func = get_mapping_func(input_image, output_image)
            
            # Apply mapping to neutral gray to get gain factors
            neutral_gray = np.ones((100, 100, 3)) * 0.5  # 50% gray
            mapped_gray = apply_mapping_func(neutral_gray, mapping_func)
            
            # Calculate gains as the ratio of mapped to original
            gains = np.mean(mapped_gray, axis=(0, 1)) / 0.5
            
            # Convert to log space
            log_gains = np.log10(gains)
            
            # Clip gains to reasonable range
            log_gains = np.clip(log_gains, -1.0, 1.0)
            
            r_gain, g_gain, b_gain = log_gains
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Estimate illuminant from gains (simplified)
        # For AWB, we assume the illuminant is the inverse of the gains
        r_illuminant = -r_gain
        g_illuminant = -g_gain
        b_illuminant = -b_gain
        
        return (r_gain, g_gain, b_gain, r_illuminant, g_illuminant, b_illuminant)


def create_deep_wb_wrapper(model_dir: Optional[str] = None, device: str = 'cpu') -> Optional[DeepWBWrapper]:
    """
    Create Deep White Balance wrapper if available
    
    Args:
        model_dir: Directory containing the ONNX model
        device: Device to run the model on
        
    Returns:
        DeepWBWrapper instance or None if not available
    """
    if not ONNX_AVAILABLE:
        return None
    
    try:
        return DeepWBWrapper(model_dir, device)
    except Exception as e:
        print(f"Failed to create Deep White Balance wrapper: {e}")
        return None


if __name__ == "__main__":
    # 测试脚本
    print("Testing ONNX Deep White Balance wrapper...")
    
    # 创建包装器
    wrapper = create_deep_wb_wrapper()
    
    if wrapper:
        print("ONNX wrapper created successfully!")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试处理
        try:
            result = wrapper.process_image(test_image)
            print(f"Test successful! Input shape: {test_image.shape}, Output shape: {result.shape}")
            
            # 测试增益计算
            gains = wrapper.calculate_auto_gain(test_image)
            print(f"Gains calculated: {gains}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("Failed to create ONNX wrapper") 