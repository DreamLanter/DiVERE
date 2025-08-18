"""
核心数据类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class InputTransformationDefinition:
    """输入变换的定义，通常是色彩空间"""
    name: str
    definition: Dict[str, Any]


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
    预设文件的数据结构。
    """
    name: str = "未命名预设"
    version: int = 2

    # Metadata
    raw_file: Optional[str] = None
    orientation: int = 0
    crop: Optional[Tuple[float, float, float, float]] = None  # (x_pct, y_pct, w_pct, h_pct)

    # Input Transformation
    input_transformation: Optional[InputTransformationDefinition] = None

    # Grading Parameters
    grading_params: Dict[str, Any] = field(default_factory=dict)
    density_matrix: Optional[MatrixDefinition] = None
    density_curve: Optional[CurveDefinition] = None

    def to_dict(self) -> Dict[str, Any]:
        """将预设对象序列化为字典。"""
        data = {
            "name": self.name,
            "version": self.version,
        }
        # Metadata
        if self.raw_file:
            data["raw_file"] = self.raw_file
        data["orientation"] = self.orientation
        if self.crop:
            data["crop"] = self.crop

        # Input Transformation
        if self.input_transformation:
            data["input_transformation"] = self.input_transformation.__dict__

        # Grading Parameters
        data["grading_params"] = self.grading_params
        if self.density_matrix:
            data["density_matrix"] = self.density_matrix.__dict__
        if self.density_curve:
            data["density_curve"] = self.density_curve.__dict__

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """从字典反序列化为预设对象。"""
        preset = cls(
            name=data.get("name", "未命名预设"),
            version=data.get("version", 2),
            # Metadata
            raw_file=data.get("raw_file"),
            orientation=data.get("orientation", 0),
            crop=tuple(data["crop"]) if data.get("crop") else None,
            # Grading Parameters
            grading_params=data.get("grading_params", {}),
        )

        # Input Transformation (with backward compatibility)
        if "input_transformation" in data and data["input_transformation"]:
            preset.input_transformation = InputTransformationDefinition(**data["input_transformation"])
        elif "input_color_space" in data and data["input_color_space"]:
            # 旧格式兼容
            preset.input_transformation = InputTransformationDefinition(**data["input_color_space"])

        # Grading Parameters
        if "density_matrix" in data and data["density_matrix"]:
            preset.density_matrix = MatrixDefinition(**data["density_matrix"])
        elif "correction_matrix" in data and data["correction_matrix"]:
            preset.density_matrix = MatrixDefinition(**data["correction_matrix"])
        # Backward compatibility for file-based matrix
        elif "grading_params" in data and "density_matrix_file" in data["grading_params"]:
            preset.density_matrix = MatrixDefinition(name=data["grading_params"]["density_matrix_file"], values=None)
        elif "grading_params" in data and "correction_matrix_file" in data["grading_params"]:
            preset.density_matrix = MatrixDefinition(name=data["grading_params"]["correction_matrix_file"], values=None)

        if "density_curve" in data and data["density_curve"]:
            preset.density_curve = CurveDefinition(**data["density_curve"])
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
    """调色参数的数据类"""
    # Density Inversion
    density_gamma: float = 2.6
    density_dmax: float = 2.0

    # Density Matrix
    density_matrix: Optional[np.ndarray] = None

    # RGB Gains
    rgb_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Density Curve
    curve_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_r: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_g: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_b: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])

    # --- Pipeline Control Flags (transient, not saved in presets) ---
    enable_density_inversion: bool = True
    enable_density_matrix: bool = False
    enable_rgb_gains: bool = True
    enable_density_curve: bool = True

    def __post_init__(self):
        # 确保matrix是ndarray
        if self.density_matrix is not None and not isinstance(self.density_matrix, np.ndarray):
            self.density_matrix = np.array(self.density_matrix)

    def copy(self) -> "ColorGradingParams":
        """返回此ColorGradingParams对象的深拷贝"""
        new_params = ColorGradingParams()
        
        # 复制基础参数
        new_params.density_gamma = self.density_gamma
        new_params.density_dmax = self.density_dmax
        new_params.density_matrix = self.density_matrix.copy() if self.density_matrix is not None else None
        new_params.rgb_gains = self.rgb_gains
        
        # 复制曲线参数
        new_params.curve_points = self.curve_points.copy()
        new_params.curve_points_r = self.curve_points_r.copy()
        new_params.curve_points_g = self.curve_points_g.copy()
        new_params.curve_points_b = self.curve_points_b.copy()

        # 复制 transient 状态
        new_params.enable_density_inversion = self.enable_density_inversion
        new_params.enable_density_matrix = self.enable_density_matrix
        new_params.enable_rgb_gains = self.enable_rgb_gains
        new_params.enable_density_curve = self.enable_density_curve
        
        return new_params

    def to_dict(self) -> Dict[str, Any]:
        """将可保存的参数序列化为字典。"""
        data = {
            'density_gamma': self.density_gamma,
            'density_dmax': self.density_dmax,
            'rgb_gains': self.rgb_gains,
            'curve_points': self.curve_points,
            'curve_points_r': self.curve_points_r,
            'curve_points_g': self.curve_points_g,
            'curve_points_b': self.curve_points_b,
        }
        if self.density_matrix is not None:
            data['density_matrix'] = self.density_matrix.tolist()
        return data

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
        
        if "density_matrix" in data:
            matrix_data = data["density_matrix"]
            if matrix_data is not None:
                params.density_matrix = np.array(matrix_data)
        
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
        
        # Backward compatibility for matrix
        if 'density_matrix' in data:
            params.density_matrix = np.array(data['density_matrix'])
        elif 'correction_matrix' in data:
            params.density_matrix = np.array(data['correction_matrix'])

        return params

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """用字典中的值部分更新当前实例"""
        if "density_gamma" in data:
            self.density_gamma = data["density_gamma"]
        if "density_dmax" in data:
            self.density_dmax = data["density_dmax"]

        if "density_matrix" in data:
            matrix_data = data["density_matrix"]
            if matrix_data is not None:
                self.density_matrix = np.array(matrix_data)
        
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

        # Backward compatibility for matrix
        if 'density_matrix' in data:
            self.density_matrix = np.array(data['density_matrix'])
        elif 'correction_matrix' in data:
            self.density_matrix = np.array(data['correction_matrix'])

        if 'curve_points_r' in data: self.curve_points_r = data['curve_points_r']
        if 'curve_points_g' in data: self.curve_points_g = data['curve_points_g']
        if 'curve_points_b' in data: self.curve_points_b = data['curve_points_b']


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
