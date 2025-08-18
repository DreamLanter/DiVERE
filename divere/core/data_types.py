"""
核心数据类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class ColorSpaceDefinition:
    """色彩空间定义（数值冗余）"""
    name: str = "sRGB"
    definition: Optional[Dict[str, Any]] = None  # { "primaries_xy": ..., "white_point_xy": ..., "gamma": ... }

@dataclass
class MatrixDefinition:
    """矩阵定义（数值冗余）"""
    name: str = "Identity"
    values: Optional[List[List[float]]] = None

@dataclass
class CurveDefinition:
    """曲线定义（数值冗余）"""
    name: Optional[str] = None
    points: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Preset:
    """
    预设数据结构 (重构版)
    - 遵循“名称+数值冗余”原则
    - 仅包含有效设置，加载时做“部分应用”
    """
    name: str = "默认预设"
    version: int = 2
    input_color_space: Optional[ColorSpaceDefinition] = None
    correction_matrix: Optional[MatrixDefinition] = None
    
    # grading_params 存储 ColorGradingParams 中定义的参数
    grading_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        data = {
            "name": self.name,
            "version": self.version,
            "grading_params": self.grading_params,
        }
        if self.input_color_space:
            data["input_color_space"] = self.input_color_space.__dict__
        if self.correction_matrix:
            data["correction_matrix"] = self.correction_matrix.__dict__
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preset':
        """从字典反序列化"""
        preset = cls(
            name=data.get("name", "未命名预设"),
            version=data.get("version", 2),
            grading_params=data.get("grading_params", {}),
        )
        if "input_color_space" in data and data["input_color_space"]:
            preset.input_color_space = ColorSpaceDefinition(**data["input_color_space"])
        if "correction_matrix" in data and data["correction_matrix"]:
            preset.correction_matrix = MatrixDefinition(**data["correction_matrix"])
        return preset


@dataclass 
class PreviewConfig:
    """预览和代理图像配置 - 统一管理所有预览相关参数"""
    
    # 预览图像尺寸设置
    preview_max_size: int = 2000  # 预览管线最大尺寸
    proxy_max_size: int = 2000    # 代理图像最大尺寸 
    
    # GPU加速阈值
    gpu_threshold: int = 1024 * 1024  # 1M像素以上使用GPU加速
    
    # 预览质量设置
    preview_quality: str = 'linear'  # 'linear', 'cubic', 'nearest'
    
    # LUT预览设置
    preview_lut_size: int = 32       # 预览LUT尺寸（32x32x32）
    full_lut_size: int = 64          # 全精度LUT尺寸（64x64x64）
    
    # 缓存设置
    max_preview_cache: int = 10      # 最大预览缓存数量
    max_lut_cache: int = 20          # 最大LUT缓存数量
    
    def get_proxy_size_tuple(self) -> Tuple[int, int]:
        """获取代理图像尺寸元组"""
        return (self.proxy_max_size, self.proxy_max_size)
    
    def should_use_gpu(self, image_size: int) -> bool:
        """判断是否应该使用GPU加速"""
        return image_size >= self.gpu_threshold


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
    """调色参数配置 (重构版)"""
    # 密度反相参数
    density_gamma: float = 2.6
    density_dmax: float = 2.0
    
    # 校正矩阵 (值由Preset或UI动态提供)
    correction_matrix_file: str = ""
    correction_matrix: Optional[np.ndarray] = None
    
    # RGB增益
    rgb_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 密度曲线
    # RGB主曲线
    curve_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    # 单通道曲线
    curve_points_r: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_g: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_b: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    
    # 调试模式参数 (瞬态，不保存到预设)
    enable_density_inversion: bool = True
    enable_correction_matrix: bool = True
    enable_rgb_gains: bool = True
    enable_density_curve: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (精简版) - 不包含enable_*标志"""
        d = {
            "density_gamma": self.density_gamma,
            "density_dmax": self.density_dmax,
            "rgb_gains": list(self.rgb_gains),
            "curve_points": self.curve_points,
            "curve_points_r": self.curve_points_r,
            "curve_points_g": self.curve_points_g,
            "curve_points_b": self.curve_points_b,
        }
        # 矩阵特殊处理：仅在自定义时保存数值
        if self.correction_matrix_file == "custom" and self.correction_matrix is not None:
             d["correction_matrix"] = self.correction_matrix.tolist()
        
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorGradingParams':
        """
        从字典反序列化 (支持部分更新)
        - data 中存在的键才会更新 params 实例
        """
        params = cls()
        if "density_gamma" in data:
            params.density_gamma = data["density_gamma"]
        if "density_dmax" in data:
            params.density_dmax = data["density_dmax"]
        
        if "correction_matrix" in data:
            matrix_data = data["correction_matrix"]
            if matrix_data is not None:
                params.correction_matrix = np.array(matrix_data)
                params.correction_matrix_file = "custom"
        
        if "rgb_gains" in data:
            rgb_gains = data["rgb_gains"]
            params.rgb_gains = tuple(rgb_gains)
        
        # 曲线参数
        if "curve_points" in data:
            params.curve_points = data["curve_points"]
        if "curve_points_r" in data:
            params.curve_points_r = data["curve_points_r"]
        if "curve_points_g" in data:
            params.curve_points_g = data["curve_points_g"]
        if "curve_points_b" in data:
            params.curve_points_b = data["curve_points_b"]
        
        return params

    def update_from_dict(self, data: Dict[str, Any]):
        """用字典中的值部分更新当前实例"""
        if "density_gamma" in data:
            self.density_gamma = data["density_gamma"]
        if "density_dmax" in data:
            self.density_dmax = data["density_dmax"]

        if "correction_matrix" in data:
            matrix_data = data["correction_matrix"]
            if matrix_data is not None:
                self.correction_matrix = np.array(matrix_data)
                self.correction_matrix_file = "custom"
        
        if "rgb_gains" in data:
            self.rgb_gains = tuple(data["rgb_gains"])

        if "curve_points" in data:
            self.curve_points = data.get("curve_points", [(0.0, 0.0), (1.0, 1.0)])
        if "curve_points_r" in data:
            self.curve_points_r = data.get("curve_points_r", [(0.0, 0.0), (1.0, 1.0)])
        if "curve_points_g" in data:
            self.curve_points_g = data.get("curve_points_g", [(0.0, 0.0), (1.0, 1.0)])
        if "curve_points_b" in data:
            self.curve_points_b = data.get("curve_points_b", [(0.0, 0.0), (1.0, 1.0)])

    def copy(self) -> 'ColorGradingParams':
        """返回此ColorGradingParams对象的深拷贝"""
        new_params = ColorGradingParams()
        
        # 复制基础参数
        new_params.density_gamma = self.density_gamma
        new_params.density_dmax = self.density_dmax
        new_params.correction_matrix_file = self.correction_matrix_file
        new_params.correction_matrix = self.correction_matrix.copy() if self.correction_matrix is not None else None
        new_params.rgb_gains = self.rgb_gains
        
        # 复制曲线参数
        new_params.curve_points = self.curve_points.copy()
        new_params.curve_points_r = self.curve_points_r.copy()
        new_params.curve_points_g = self.curve_points_g.copy()
        new_params.curve_points_b = self.curve_points_b.copy()

        # 复制调试模式参数 (瞬态)
        new_params.enable_density_inversion = self.enable_density_inversion
        new_params.enable_correction_matrix = self.enable_correction_matrix
        new_params.enable_rgb_gains = self.enable_rgb_gains
        new_params.enable_density_curve = self.enable_density_curve
        
        return new_params

@dataclass
class LUT3D:
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
