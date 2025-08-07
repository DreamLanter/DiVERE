"""
核心数据类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class ImageData:
    """图像数据封装"""
    array: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    channels: int = 3
    dtype: np.dtype = np.float32
    color_space: str = "sRGB"
    icc_profile: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    is_proxy: bool = False
    proxy_scale: float = 1.0
    
    def __post_init__(self):
        if self.array is not None:
            self.height, self.width = self.array.shape[:2]
            self.channels = self.array.shape[2] if len(self.array.shape) == 3 else 1
            self.dtype = self.array.dtype
    
    def copy(self):
        """返回此ImageData对象的深拷贝"""
        new_array = self.array.copy() if self.array is not None else None
        return ImageData(
            array=new_array,
            width=self.width,
            height=self.height,
            channels=self.channels,
            dtype=self.dtype,
            color_space=self.color_space,
            icc_profile=self.icc_profile,
            metadata=self.metadata.copy(),
            file_path=self.file_path,
            is_proxy=self.is_proxy,
            proxy_scale=self.proxy_scale
        )

    def copy_with_new_array(self, new_array: np.ndarray):
        """返回一个带有新图像数组的新ImageData实例，同时复制所有其他元数据"""
        return ImageData(
            array=new_array,
            color_space=self.color_space,
            icc_profile=self.icc_profile,
            metadata=self.metadata.copy(),
            file_path=self.file_path,
            is_proxy=self.is_proxy,
            proxy_scale=self.proxy_scale
        )


@dataclass
class ColorGradingParams:
    # ... (rest of the class is unchanged)
    """调色参数配置"""
    # 基础参数
    input_gamma: float = 2.2
    input_gain: float = 1.0
    
    # 密度反相参数
    density_gamma: float = 2.6
    density_gain: float = 1.0
    density_dmax: float = 2.0
    
    # 校正矩阵
    correction_matrix_file: str = ""
    correction_matrix: Optional[np.ndarray] = None
    
    # RGB增益
    rgb_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 密度曲线
    density_curve_points: List[Tuple[float, float]] = field(default_factory=list)
    

    
    # 调试模式参数
    enable_density_inversion: bool = True
    enable_correction_matrix: bool = True
    enable_rgb_gains: bool = True
    enable_density_curve: bool = True
    
    # 曲线调整
    enable_curve: bool = False
    curve_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])  # RGB主曲线
    
    # 单通道曲线
    enable_curve_r: bool = False
    enable_curve_g: bool = False
    enable_curve_b: bool = False
    curve_points_r: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_g: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_b: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    
    # 性能优化参数（已简化，移除快速预览模式）
    small_proxy: bool = True
    low_precision_lut: bool = True  # 使用低精度LUT
    lut_size: int = 16  # LUT大小，默认16x16x16
    proxy_size: str = "1920x1920"  # 代理图像大小
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "input_gamma": self.input_gamma,
            "input_gain": self.input_gain,
            "density_gamma": self.density_gamma,
            "density_gain": self.density_gain,
            "density_dmax": self.density_dmax,
            "correction_matrix_file": self.correction_matrix_file,
            "correction_matrix": self.correction_matrix.tolist() if self.correction_matrix is not None else None,
            "rgb_gains": list(self.rgb_gains),
            "density_curve_points": self.density_curve_points,


            "enable_density_inversion": self.enable_density_inversion,
            "enable_correction_matrix": self.enable_correction_matrix,
            "enable_rgb_gains": self.enable_rgb_gains,
            "enable_density_curve": self.enable_density_curve,
            
            "enable_curve": self.enable_curve,
            "curve_points": self.curve_points,
            "enable_curve_r": self.enable_curve_r,
            "enable_curve_g": self.enable_curve_g,
            "enable_curve_b": self.enable_curve_b,
            "curve_points_r": self.curve_points_r,
            "curve_points_g": self.curve_points_g,
            "curve_points_b": self.curve_points_b,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorGradingParams':
        """从字典反序列化"""
        params = cls()
        params.input_gamma = data.get("input_gamma", 2.2)
        params.input_gain = data.get("input_gain", 1.0)
        params.density_gamma = data.get("density_gamma", 2.6)
        params.density_gain = data.get("density_gain", 1.0)
        params.density_dmax = data.get("density_dmax", 2.0)
        params.correction_matrix_file = data.get("correction_matrix_file", "")
        
        matrix_data = data.get("correction_matrix")
        if matrix_data is not None:
            params.correction_matrix = np.array(matrix_data)
        
        rgb_gains = data.get("rgb_gains", [0.0, 0.0, 0.0])
        params.rgb_gains = tuple(rgb_gains)
        
        params.density_curve_points = data.get("density_curve_points", [])

        params.enable_density_curve = data.get("enable_density_curve", False)
        
        # 曲线参数
        params.enable_curve = data.get("enable_curve", False)
        params.curve_points = data.get("curve_points", [(0.0, 0.0), (1.0, 1.0)])
        params.enable_curve_r = data.get("enable_curve_r", False)
        params.enable_curve_g = data.get("enable_curve_g", False)
        params.enable_curve_b = data.get("enable_curve_b", False)
        params.curve_points_r = data.get("curve_points_r", [(0.0, 0.0), (1.0, 1.0)])
        params.curve_points_g = data.get("curve_points_g", [(0.0, 0.0), (1.0, 1.0)])
        params.curve_points_b = data.get("curve_points_b", [(0.0, 0.0), (1.0, 1.0)])
        
        return params
    
    def copy(self) -> 'ColorGradingParams':
        """返回此ColorGradingParams对象的深拷贝"""
        new_params = ColorGradingParams()
        
        # 复制基础参数
        new_params.input_gamma = self.input_gamma
        new_params.input_gain = self.input_gain
        new_params.density_gamma = self.density_gamma
        new_params.density_gain = self.density_gain
        new_params.density_dmax = self.density_dmax
        new_params.correction_matrix_file = self.correction_matrix_file
        new_params.correction_matrix = self.correction_matrix.copy() if self.correction_matrix is not None else None
        new_params.rgb_gains = self.rgb_gains
        new_params.density_curve_points = self.density_curve_points.copy()
        
        # 复制调试模式参数
        new_params.enable_density_inversion = self.enable_density_inversion
        new_params.enable_correction_matrix = self.enable_correction_matrix
        new_params.enable_rgb_gains = self.enable_rgb_gains
        new_params.enable_density_curve = self.enable_density_curve
        
        # 复制曲线参数
        new_params.enable_curve = self.enable_curve
        new_params.curve_points = self.curve_points.copy()
        new_params.enable_curve_r = self.enable_curve_r
        new_params.enable_curve_g = self.enable_curve_g
        new_params.enable_curve_b = self.enable_curve_b
        new_params.curve_points_r = self.curve_points_r.copy()
        new_params.curve_points_g = self.curve_points_g.copy()
        new_params.curve_points_b = self.curve_points_b.copy()
        
        # 复制性能优化参数
        new_params.small_proxy = self.small_proxy
        new_params.low_precision_lut = self.low_precision_lut
        new_params.lut_size = self.lut_size
        new_params.proxy_size = self.proxy_size
        
        return new_params

@dataclass
class LUT3D:
    # ... (rest of the class is unchanged)
    """3D LUT数据结构"""
    size: int = 32  # LUT大小 (size x size x size)
    data: Optional[np.ndarray] = None  # LUT数据 (size^3, 3)
    
    def __post_init__(self):
        """初始化默认LUT"""
        if self.data is None:
            self.data = self._create_identity_lut()
    
    def _create_identity_lut(self) -> np.ndarray:
        """创建单位LUT"""
        size = self.size
        lut = np.zeros((size**3, 3), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    idx = i * size**2 + j * size + k
                    lut[idx] = [i/(size-1), j/(size-1), k/(size-1)]
        
        return lut
    
    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """将LUT应用到图像"""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 简单的三线性插值LUT应用
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        for y in range(h):
            for x in range(w):
                pixel = image[y, x]
                # 计算LUT索引
                indices = np.clip(pixel * (self.size - 1), 0, self.size - 1)
                # 简单的最近邻插值（简化版本）
                i, j, k = indices.astype(int)
                idx = i * self.size**2 + j * self.size + k
                result[y, x] = self.data[idx]
        
        return result

@dataclass
class Curve:
    # ... (rest of the class is unchanged)
    """密度曲线数据结构"""
    points: List[Tuple[float, float]] = field(default_factory=list)
    interpolation_method: str = "linear"  # linear, cubic, bezier
    
    def add_point(self, x: float, y: float):
        """添加控制点"""
        self.points.append((x, y))
        self.points.sort(key=lambda p: p[0])  # 按x坐标排序
    
    def remove_point(self, index: int):
        """删除控制点"""
        if 0 <= index < len(self.points):
            self.points.pop(index)
    
    def get_interpolated_curve(self, num_points: int = 256) -> np.ndarray:
        """获取插值后的曲线数据"""
        if len(self.points) < 2:
            return np.linspace(0, 1, num_points)
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        
        # 简单的线性插值
        curve_x = np.linspace(0, 1, num_points)
        curve_y = np.interp(curve_x, x_coords, y_coords)
        
        return np.column_stack([curve_x, curve_y])
    
    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """将曲线应用到图像"""
        if len(self.points) < 2:
            return image
        
        curve_data = self.get_interpolated_curve()
        curve_x = curve_data[:, 0]
        curve_y = curve_data[:, 1]
        
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            # 对每个通道应用曲线
            channel = image[:, :, c]
            # 将像素值映射到曲线
            indices = np.clip(channel * (len(curve_x) - 1), 0, len(curve_x) - 1).astype(int)
            result[:, :, c] = curve_y[indices]
        
        return result
