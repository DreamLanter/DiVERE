"""
色彩空间管理模块
处理色彩空间转换和管理
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import colour
import json
import os

from .data_types import ImageData


class ColorSpaceManager:
    """色彩空间管理器"""
    
    def __init__(self):
        # 从JSON文件加载色彩空间定义
        self._color_spaces = {}
        self._load_colorspaces_from_json()
        
        # 不再需要预计算转换矩阵，使用在线计算
        # 增加一个简单的转换缓存，加速重复转换
        self._convert_cache: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    
    def _load_colorspaces_from_json(self):
        """从JSON文件加载色彩空间定义"""
        colorspace_dir = Path("config/colorspace")
        if not colorspace_dir.exists():
            print(f"警告：色彩空间配置目录 {colorspace_dir} 不存在")
            return
            
        for json_file in colorspace_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 将primaries转换为numpy数组格式
                if "primaries" in data and isinstance(data["primaries"], dict):
                    primaries = np.array([
                        data["primaries"]["R"],
                        data["primaries"]["G"],
                        data["primaries"]["B"]
                    ])
                    data["primaries"] = primaries
                    
                # 将white_point转换为numpy数组
                if "white_point" in data:
                    data["white_point"] = np.array(data["white_point"])
                    
                # 使用文件名（不含扩展名）作为色彩空间名称
                colorspace_name = json_file.stem
                self._color_spaces[colorspace_name] = data
                
            except Exception as e:
                print(f"加载色彩空间配置文件 {json_file} 时出错: {e}")
    
    def _build_conversion_matrices(self):
        """构建色彩空间转换矩阵"""
        # 现在使用在线计算，不再预计算所有矩阵
        # 只在需要时动态计算转换矩阵和增益向量
        pass
    
    def calculate_color_space_conversion(self, src_space_name: str, dst_space_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        计算色彩空间转换矩阵和增益向量
        
        Args:
            src_space_name: 源色彩空间名称
            dst_space_name: 目标色彩空间名称
            
        Returns:
            tuple: (转换矩阵, 增益向量) - 3x3矩阵和长度为3的向量
        """
        if src_space_name == dst_space_name:
            return np.eye(3), np.array([1.0, 1.0, 1.0])
        
        # 缓存命中
        cache_key = (src_space_name, dst_space_name)
        cached = self._convert_cache.get(cache_key)
        if cached is not None:
            return cached

        # 获取源和目标色彩空间信息
        src_space = self._color_spaces.get(src_space_name)
        dst_space = self._color_spaces.get(dst_space_name)
        
        if src_space is None or dst_space is None:
            print(f"警告: 未找到色彩空间定义，使用单位矩阵")
            return np.eye(3), np.array([1.0, 1.0, 1.0])
        
        try:
            # 计算源色彩空间的RGB到XYZ矩阵
            src_matrix = self._calculate_rgb_to_xyz_matrix(src_space)
            
            # 计算目标色彩空间的RGB到XYZ矩阵
            dst_matrix = self._calculate_rgb_to_xyz_matrix(dst_space)
            
            # 计算XYZ到目标RGB的矩阵（目标矩阵的逆）
            dst_matrix_inv = np.linalg.inv(dst_matrix)
            
            # 计算从源RGB到目标RGB的转换矩阵
            conversion_matrix = np.dot(dst_matrix_inv, src_matrix)
            
            # 计算白点适应增益向量
            gain_vector = self._calculate_white_point_adaptation(src_space, dst_space)
            
            # 缓存结果
            self._convert_cache[cache_key] = (conversion_matrix.astype(np.float32), gain_vector.astype(np.float32))
            return self._convert_cache[cache_key]
            
        except Exception as e:
            print(f"色彩空间转换计算失败: {e}")
            return np.eye(3), np.array([1.0, 1.0, 1.0])
    
    def _calculate_rgb_to_xyz_matrix(self, color_space: dict) -> np.ndarray:
        """
        根据RGB基色和白点计算RGB到XYZ的转换矩阵
        
        Args:
            color_space: 色彩空间信息字典，包含primaries和white_point
            
        Returns:
            3x3的RGB到XYZ转换矩阵
        """
        primaries = color_space['primaries']  # [[Rx,Ry], [Gx,Gy], [Bx,By]]
        white_point = color_space['white_point']  # [Wx, Wy]
        
        # 将xy色度坐标转换为XYZ坐标（假设Y=1）
        def xy_to_XYZ(xy):
            x, y = xy
            if y == 0:
                return np.array([0, 0, 0])
            X = x / y
            Y = 1.0
            Z = (1 - x - y) / y
            return np.array([X, Y, Z])
        
        # 计算RGB基色的XYZ坐标
        R_XYZ = xy_to_XYZ(primaries[0])  # 红色基色
        G_XYZ = xy_to_XYZ(primaries[1])  # 绿色基色
        B_XYZ = xy_to_XYZ(primaries[2])  # 蓝色基色
        
        # 计算白点的XYZ坐标
        W_XYZ = xy_to_XYZ(white_point)
        
        # 构建基色矩阵 [Rx Gx Bx; Ry Gy By; Rz Gz Bz]
        primaries_matrix = np.column_stack([R_XYZ, G_XYZ, B_XYZ])
        
        # 求解标量因子，使得 primaries_matrix * [Sr, Sg, Sb]^T = W_XYZ
        try:
            scaling_factors = np.linalg.solve(primaries_matrix, W_XYZ)
        except np.linalg.LinAlgError:
            print("警告: 基色矩阵奇异，使用默认缩放因子")
            scaling_factors = np.array([1.0, 1.0, 1.0])
        
        # 构建最终的RGB到XYZ转换矩阵
        rgb_to_xyz_matrix = primaries_matrix * scaling_factors[np.newaxis, :]
        
        return rgb_to_xyz_matrix
    
    def _calculate_white_point_adaptation(self, src_space: dict, dst_space: dict) -> np.ndarray:
        """
        计算白点适应增益向量
        
        Args:
            src_space: 源色彩空间信息
            dst_space: 目标色彩空间信息
            
        Returns:
            长度为3的增益向量
        """
        src_white = src_space['white_point']
        dst_white = dst_space['white_point']
        
        # 简化的白点适应：基于白点XYZ坐标的比值
        def xy_to_XYZ_normalized(xy):
            x, y = xy
            if y == 0:
                return np.array([1, 1, 1])
            X = x / y
            Y = 1.0
            Z = (1 - x - y) / y
            return np.array([X, Y, Z])
        
        src_white_XYZ = xy_to_XYZ_normalized(src_white)
        dst_white_XYZ = xy_to_XYZ_normalized(dst_white)
        
        # 计算白点适应增益（避免除零）
        gain_vector = np.divide(dst_white_XYZ, src_white_XYZ, 
                               out=np.ones(3), where=src_white_XYZ!=0)
        
        # 限制增益范围，避免极端值
        gain_vector = np.clip(gain_vector, 0.1, 10.0)
        
        return gain_vector
    
    def _get_colour_space_name(self, space: dict) -> str:
        """获取colour库中的色彩空间名称"""
        # 映射自定义名称到colour库名称
        if 'primaries' in space:
            primaries = space['primaries']
            # sRGB
            if np.allclose(primaries[0], [0.6400, 0.3300], atol=1e-3):
                return 'sRGB'
            # ACEScg  
            elif np.allclose(primaries[0], [0.7130, 0.2930], atol=1e-3):
                return 'ACEScg'
            # Adobe RGB
            elif np.allclose(primaries[0], [0.6400, 0.3300], atol=1e-3) and \
                 np.allclose(primaries[1], [0.2100, 0.7100], atol=1e-3):
                return 'Adobe RGB (1998)'
        
        return 'sRGB'  # 默认
    

    
    def get_available_color_spaces(self) -> list:
        """获取可用的色彩空间列表"""
        return list(self._color_spaces.keys())
    
    def validate_color_space(self, color_space_name: str) -> bool:
        """验证色彩空间名称是否有效"""
        return color_space_name in self._color_spaces
    
    def get_color_space_info(self, color_space_name: str) -> Optional[dict]:
        """获取色彩空间详细信息"""
        return self._color_spaces.get(color_space_name, None)
    
    def set_image_color_space(self, image: ImageData, color_space: str) -> ImageData:
        """设置图像的色彩空间"""
        if not self.validate_color_space(color_space):
            print(f"无效的色彩空间: {color_space}，使用默认值Film_KodakRGB_Linear")
            color_space = "Film_KodakRGB_Linear"
        
        # 创建新的图像数据对象，更新色彩空间信息
        new_image = ImageData(
            array=image.array.copy(),
            width=image.width,
            height=image.height,
            channels=image.channels,
            color_space=color_space,
            file_path=image.file_path,
            is_proxy=image.is_proxy,
            proxy_scale=image.proxy_scale
        )
        print(f"设置图像色彩空间: {image.color_space} -> {color_space}")
        return new_image
    
    def convert_to_working_space(self, image: ImageData, source_profile: str = None) -> ImageData:
        """转换到工作色彩空间（ACEScg Linear）"""
        if image.color_space == "ACEScg":
            return image
        
        # 如果指定了source_profile参数，使用它；否则使用图像的color_space
        source_space = source_profile if source_profile else image.color_space
        
        # 先转换到线性空间
        linear_image = self._convert_to_linear(image, source_space)
        
        # 然后转换到ACEScg
        if source_space != "ACEScg":
            # 使用在线计算的转换矩阵和增益向量
            conversion_matrix, gain_vector = self.calculate_color_space_conversion(source_space, "ACEScg")
            linear_image.array = self._apply_color_conversion(linear_image.array, conversion_matrix, gain_vector)
        
        linear_image.color_space = "ACEScg"
        return linear_image
    
    def convert_to_display_space(self, image: ImageData, target_space: str = "sRGB") -> ImageData:
        """转换到显示色彩空间"""
        if image.color_space == target_space:
            return image
        
        # 从ACEScg转换到目标空间
        if image.color_space == "ACEScg":
            # 使用在线计算的转换矩阵和增益向量
            conversion_matrix, gain_vector = self.calculate_color_space_conversion("ACEScg", target_space)
            image.array = self._apply_color_conversion(image.array, conversion_matrix, gain_vector)
        
        # 应用gamma校正
        image.array = self._apply_gamma(image.array, self._color_spaces[target_space]["gamma"])
        image.color_space = target_space
        
        return image
    
    def _convert_to_linear(self, image: ImageData, source_space: str) -> ImageData:
        """转换到线性空间"""
        gamma = self._color_spaces.get(source_space, {}).get("gamma", 2.2)
        
        # 应用gamma校正（从非线性到线性）
        linear_array = self._apply_gamma(image.array, gamma, inverse=True)
        
        linear_image = ImageData(
            array=linear_array,
            width=image.width,
            height=image.height,
            channels=image.channels,
            dtype=image.dtype,
            color_space=f"{source_space}_Linear",
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=image.is_proxy,
            proxy_scale=image.proxy_scale
        )
        
        return linear_image
    
    def _apply_gamma(self, image_array: np.ndarray, gamma: float, inverse: bool = False) -> np.ndarray:
        """应用gamma校正"""
        # 确保图像数据在[0,1]范围内
        image_array = np.clip(image_array, 0, 1)
        
        if inverse:
            # 从非线性到线性：I_linear = I_nonlinear^gamma
            return np.power(image_array, gamma)
        else:
            # 从线性到非线性：I_nonlinear = I_linear^(1/gamma)
            return np.power(image_array, 1.0 / gamma)
    
    def _apply_color_conversion(self, image_array: np.ndarray, matrix: np.ndarray, gain_vector: np.ndarray) -> np.ndarray:
        """应用色彩矩阵变换和增益校正"""
        # 重塑图像为2D数组以便矩阵乘法
        original_shape = image_array.shape
        if len(original_shape) == 3:
            h, w, c = original_shape
            reshaped = image_array.reshape(-1, c)
            
            # 应用矩阵变换
            transformed = np.dot(reshaped, matrix.T)  # 注意转置，因为我们的矩阵是3x3列向量格式
            
            # 应用白点适应增益
            transformed *= gain_vector[np.newaxis, :]
            
            return transformed.reshape(original_shape)
        else:
            return image_array
    
    def _apply_color_matrix(self, image_array: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """应用色彩矩阵变换（保留用于向后兼容）"""
        return self._apply_color_conversion(image_array, matrix, np.array([1.0, 1.0, 1.0]))
    
    def get_default_color_space(self) -> str:
        """获取默认色彩空间"""
        return "Film_KodakRGB_Linear"
    

    
    def estimate_source_gamma(self, image: ImageData) -> float:
        """估算源图像的gamma值"""
        # 简单的gamma估算方法
        # 基于图像直方图的分布特征
        
        if len(image.array.shape) == 3:
            # 使用亮度通道
            gray = np.dot(image.array[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.array
        
        # 计算累积分布函数
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        cdf = np.cumsum(hist) / np.sum(hist)
        
        # 找到50%分位数
        mid_point = np.argmax(cdf >= 0.5) / 256.0
        
        # 基于中值估算gamma
        if mid_point > 0:
            estimated_gamma = np.log(0.5) / np.log(mid_point)
            return np.clip(estimated_gamma, 1.0, 3.0)
        
        return 2.2  # 默认值
    
    def apply_white_balance(self, image: ImageData, temperature: float, tint: float) -> ImageData:
        """应用白平衡校正"""
        # 简化的白平衡实现
        # temperature: 色温 (K)
        # tint: 色调偏移
        
        # 计算色温转换矩阵
        # 这里使用简化的转换，实际应用中需要更复杂的算法
        if temperature != 6500:  # 6500K为标准白点
            # 简化的色温调整
            ratio = 6500 / temperature
            matrix = np.array([
                [ratio, 0, 0],
                [0, 1, 0], 
                [0, 0, 1/ratio]
            ])
            
            image.array = self._apply_color_matrix(image.array, matrix)
        
        return image 