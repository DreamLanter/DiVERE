"""
IDT计算引擎

实现精确通道分离IDT计算的核心算法，包括：
- 图片RGB值提取
- 3x3线性变换矩阵计算
- 色彩空间原色坐标反推计算
- 使用纯numpy实现，确保数值精度
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import cv2
from PIL import Image
import json


class IDTCalculationEngine:
    """IDT计算引擎"""
    
    def __init__(self):
        self.rgb_values: Dict[str, np.ndarray] = {}  # 存储三个通道的RGB值
        self.ccm_matrix: Optional[np.ndarray] = None  # 优化后的CCM矩阵
        self.source_primaries: Optional[np.ndarray] = None  # 计算出的原始色彩空间原色
        self.source_whitepoint: Optional[np.ndarray] = None  # 计算出的原始色彩空间白点
        
    def load_and_extract_rgb(self, image_path: str, channel_name: str) -> bool:
        """
        加载图片并提取九宫格中心1/9区域的平均RGB值
        
        Args:
            image_path: 图片文件路径
            channel_name: 通道名称 ('red', 'green', 'blue')
            
        Returns:
            是否成功提取
        """
        try:
            # 尝试使用PIL加载图片（支持更多格式）
            with Image.open(image_path) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为numpy数组
                img_array = np.array(img).astype(np.float32) / 255.0
            
            # 获取图片尺寸
            h, w = img_array.shape[:2]
            
            # 计算九宫格中心1/9区域的边界
            center_h_start = h // 3
            center_h_end = 2 * h // 3
            center_w_start = w // 3  
            center_w_end = 2 * w // 3
            
            # 提取中心区域
            center_region = img_array[center_h_start:center_h_end, center_w_start:center_w_end]
            
            # 计算平均RGB值
            avg_rgb = np.mean(center_region, axis=(0, 1))
            
            # 存储结果
            self.rgb_values[channel_name] = avg_rgb
            
            print(f"成功提取{channel_name}通道RGB值: {avg_rgb}")
            return True
            
        except Exception as e:
            print(f"加载图片{image_path}失败: {e}")
            return False
    
    def get_initial_rgb_matrix(self) -> Optional[np.ndarray]:
        """
        获取初始RGB值矩阵（3x3）
        每一行对应一个通道的RGB值
        
        Returns:
            3x3的RGB矩阵，如果数据不完整则返回None
        """
        if len(self.rgb_values) != 3:
            return None
            
        # 按照红、绿、蓝的顺序排列
        channel_order = ['red', 'green', 'blue']
        
        # 检查是否有缺失的通道
        for channel in channel_order:
            if channel not in self.rgb_values:
                return None
        
        # 构建3x3矩阵
        rgb_matrix = np.array([
            self.rgb_values['red'],
            self.rgb_values['green'], 
            self.rgb_values['blue']
        ])
        
        return rgb_matrix
    
    def normalize_ccm_rows(self, ccm: np.ndarray) -> np.ndarray:
        """
        对CCM矩阵的每一行进行归一化，确保行和为1.0
        
        Args:
            ccm: 3x3 CCM矩阵
            
        Returns:
            归一化后的CCM矩阵
        """
        normalized_ccm = ccm.copy()
        
        for i in range(3):
            row_sum = np.sum(normalized_ccm[i, :])
            if abs(row_sum) > 1e-10:  # 避免除零
                normalized_ccm[i, :] /= row_sum
            else:
                # 如果行和接近0，设置为单位矩阵的对应行
                normalized_ccm[i, :] = 0.0
                normalized_ccm[i, i] = 1.0
        
        return normalized_ccm
    
    def calculate_source_colorspace(self, ccm: np.ndarray, target_primaries: np.ndarray, 
                                   target_whitepoint: np.ndarray, target_colorspace: str,
                                   color_space_manager=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从CCM矩阵反推原始色彩空间的原色坐标和白点
        
        数学原理：
        如果 CCM * source_rgb = target_rgb
        那么 source_rgb = CCM^(-1) * target_rgb
        
        Args:
            ccm: 归一化后的3x3 CCM矩阵
            target_primaries: 目标工作空间的原色（3x3矩阵，每行为[R, G, B]）
            target_whitepoint: 目标工作空间的白点RGB值
            target_colorspace: 目标色彩空间名称
            color_space_manager: ColorSpaceManager实例
            
        Returns:
            (source_primaries_xy, source_whitepoint_xy) - 原始色彩空间的原色和白点的xy坐标
        """
        try:
            # 计算CCM的逆矩阵
            ccm_inv = np.linalg.inv(ccm)
            
            # 计算原始色彩空间的RGB原色（每行对应一个原色）
            source_rgb_primaries = ccm_inv @ target_primaries
            
            # 计算原始色彩空间的白点RGB值
            source_rgb_whitepoint = ccm_inv @ target_whitepoint
            
            # 将RGB值转换为xy色度坐标
            # 使用目标色彩空间的转换矩阵
            source_primaries_xy = self._rgb_to_xy_chromaticity(
                source_rgb_primaries, target_colorspace, color_space_manager
            )
            source_whitepoint_xy = self._rgb_to_xy_chromaticity(
                source_rgb_whitepoint.reshape(1, -1), target_colorspace, color_space_manager
            )[0]
            
            return source_primaries_xy, source_whitepoint_xy
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"CCM矩阵不可逆: {e}")
    
    def _rgb_to_xy_chromaticity(self, rgb_values: np.ndarray, 
                                target_colorspace: str = "sRGB",
                                color_space_manager=None) -> np.ndarray:
        """
        将RGB值转换为xy色度坐标
        
        使用目标色彩空间的RGB-to-XYZ转换矩阵进行转换
        
        Args:
            rgb_values: RGB值数组，形状为(N, 3)
            target_colorspace: 目标色彩空间名称
            color_space_manager: ColorSpaceManager实例
            
        Returns:
            xy色度坐标数组，形状为(N, 2)
        """
        # 确保输入是2D数组
        if rgb_values.ndim == 1:
            rgb_values = rgb_values.reshape(1, -1)
        
        # 获取目标色彩空间的xy信息
        primaries_xy, white_point_xy = self.get_target_colorspace_xy_info(
            target_colorspace, color_space_manager
        )
        
        # 构建RGB到XYZ转换矩阵
        rgb_to_xyz_matrix = self.build_rgb_to_xyz_matrix(primaries_xy, white_point_xy)
        
        # 转换为XYZ
        xyz_values = rgb_values @ rgb_to_xyz_matrix.T
        
        # 转换为xy色度坐标
        xy_coords = np.zeros((xyz_values.shape[0], 2))
        
        for i in range(xyz_values.shape[0]):
            X, Y, Z = xyz_values[i]
            total = X + Y + Z
            
            if total > 1e-10:  # 避免除零
                xy_coords[i, 0] = X / total  # x
                xy_coords[i, 1] = Y / total  # y
            else:
                # 如果XYZ接近零，使用目标色彩空间的白点作为默认值
                xy_coords[i, 0] = white_point_xy[0]
                xy_coords[i, 1] = white_point_xy[1]
        
        return xy_coords
    
    def xy_to_XYZ(self, x: float, y: float, Y: float = 1.0) -> np.ndarray:
        """
        将xy色度坐标转换为XYZ坐标
        
        Args:
            x, y: 色度坐标
            Y: Y值（亮度），默认为1.0
            
        Returns:
            XYZ坐标数组
        """
        if y == 0:
            return np.array([0.0, 0.0, 0.0])
        
        X = (x * Y) / y
        Z = ((1 - x - y) * Y) / y
        return np.array([X, Y, Z])
    
    def build_rgb_to_xyz_matrix(self, primaries_xy, white_point_xy: List[float]) -> np.ndarray:
        """
        根据色彩空间的原色和白点构建RGB到XYZ的转换矩阵
        
        Args:
            primaries_xy: 原色的xy坐标，可以是字典格式{"R": [x, y], "G": [x, y], "B": [x, y]}
                        或其他格式
            white_point_xy: 白点的xy坐标 [x, y]
            
        Returns:
            3x3 RGB到XYZ转换矩阵
        """
        try:
            # 处理不同的primaries_xy格式
            if isinstance(primaries_xy, dict):
                # 字典格式
                if 'R' in primaries_xy and 'G' in primaries_xy and 'B' in primaries_xy:
                    R_XYZ = self.xy_to_XYZ(primaries_xy['R'][0], primaries_xy['R'][1], 1.0)
                    G_XYZ = self.xy_to_XYZ(primaries_xy['G'][0], primaries_xy['G'][1], 1.0)
                    B_XYZ = self.xy_to_XYZ(primaries_xy['B'][0], primaries_xy['B'][1], 1.0)
                else:
                    raise ValueError(f'字典格式的primaries_xy缺少必需的键: {primaries_xy.keys()}')
            else:
                # 如果不是字典，尝试其他格式的解析
                raise ValueError(f'不支持的primaries_xy格式: {type(primaries_xy)}, 内容: {primaries_xy}')
            
            # 白点XYZ
            W_XYZ = self.xy_to_XYZ(white_point_xy[0], white_point_xy[1], 1.0)
            
            # 构建原色矩阵
            primaries_matrix = np.column_stack([R_XYZ, G_XYZ, B_XYZ])
            
            # 计算缩放系数，使得 primaries_matrix @ scaling = W_XYZ
            scaling = np.linalg.solve(primaries_matrix, W_XYZ)
            
            # 应用缩放系数
            rgb_to_xyz_matrix = primaries_matrix @ np.diag(scaling)
            
            return rgb_to_xyz_matrix
            
        except Exception as e:
            print(f'构建RGB到XYZ矩阵时出错: {e}')
            print(f'primaries_xy类型: {type(primaries_xy)}')
            print(f'primaries_xy内容: {primaries_xy}')
            print(f'white_point_xy: {white_point_xy}')
            raise
    
    def get_target_colorspace_info(self, colorspace_name: str, 
                                   color_space_manager=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取目标色彩空间的原色RGB值（在该色彩空间内的单位原色）
        
        重要：这个方法返回的是目标色彩空间内的单位原色，即：
        - 红色原色在目标空间中的RGB值 = [1, 0, 0]
        - 绿色原色在目标空间中的RGB值 = [0, 1, 0]  
        - 蓝色原色在目标空间中的RGB值 = [0, 0, 1]
        - 白点在目标空间中的RGB值 = [1, 1, 1]
        
        Args:
            colorspace_name: 色彩空间名称
            color_space_manager: ColorSpaceManager实例
            
        Returns:
            (primaries_rgb, whitepoint_rgb) - 目标色彩空间内的原色RGB矩阵和白点RGB值
        """
        # 在任何线性色彩空间中，原色都是单位向量
        primaries_rgb = np.array([
            [1.0, 0.0, 0.0],  # 红色原色
            [0.0, 1.0, 0.0],  # 绿色原色
            [0.0, 0.0, 1.0]   # 蓝色原色
        ])
        whitepoint_rgb = np.array([1.0, 1.0, 1.0])  # 白点
        
        return primaries_rgb, whitepoint_rgb
    
    def get_target_colorspace_xy_info(self, colorspace_name: str, 
                                     color_space_manager=None) -> Tuple[Dict, List[float]]:
        """
        获取目标色彩空间的xy色度坐标信息
        
        Args:
            colorspace_name: 色彩空间名称
            color_space_manager: ColorSpaceManager实例
            
        Returns:
            (primaries_xy, white_point_xy) - 原色xy坐标字典和白点xy坐标
        """
        # 尝试从ColorSpaceManager获取真实的色彩空间信息
        if color_space_manager:
            try:
                colorspace_info = color_space_manager._color_spaces.get(colorspace_name)
                if colorspace_info and 'primaries' in colorspace_info:
                    primaries_array = colorspace_info['primaries']
                    white_point_array = colorspace_info.get('white_point', [0.3127, 0.3290])
                    
                    # 将numpy数组转换为字典格式
                    if isinstance(primaries_array, np.ndarray) and primaries_array.shape == (3, 2):
                        primaries_xy = {
                            'R': [float(primaries_array[0, 0]), float(primaries_array[0, 1])],
                            'G': [float(primaries_array[1, 0]), float(primaries_array[1, 1])],
                            'B': [float(primaries_array[2, 0]), float(primaries_array[2, 1])]
                        }
                    else:
                        # 如果已经是字典格式，直接使用
                        primaries_xy = primaries_array
                    
                    # 确保白点是列表格式
                    if isinstance(white_point_array, np.ndarray):
                        white_point_xy = white_point_array.tolist()
                    else:
                        white_point_xy = white_point_array
                    
                    return primaries_xy, white_point_xy
            except Exception as e:
                print(f"从ColorSpaceManager获取色彩空间信息失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 回退到硬编码的信息
        if colorspace_name == "ACEScg":
            primaries_xy = {
                "R": [0.7130, 0.2930],
                "G": [0.1650, 0.8300], 
                "B": [0.1280, 0.0440]
            }
            white_point_xy = [0.32168, 0.33767]
            
        elif colorspace_name == "KodakEnduraPremier":
            primaries_xy = {
                "R": [0.6635212396720807, 0.32411816662609266],
                "G": [0.2592092094845883, 0.6664046572656293],
                "B": [0.15456877810311515, 0.049802203945821064]
            }
            white_point_xy = [0.32168, 0.33767]
            
        elif colorspace_name == "sRGB":
            primaries_xy = {
                "R": [0.6400, 0.3300],
                "G": [0.3000, 0.6000],
                "B": [0.1500, 0.0600]
            }
            white_point_xy = [0.3127, 0.3290]
            
        else:
            # 默认使用sRGB
            primaries_xy = {
                "R": [0.6400, 0.3300],
                "G": [0.3000, 0.6000],
                "B": [0.1500, 0.0600]
            }
            white_point_xy = [0.3127, 0.3290]
        
        return primaries_xy, white_point_xy
    
    def set_optimized_ccm(self, ccm: np.ndarray):
        """设置优化后的CCM矩阵"""
        self.ccm_matrix = self.normalize_ccm_rows(ccm)
    
    def calculate_final_colorspace(self, target_colorspace: str, 
                                   color_space_manager=None) -> Dict:
        """
        计算最终的色彩空间信息
        
        Args:
            target_colorspace: 目标工作空间名称
            color_space_manager: ColorSpaceManager实例
            
        Returns:
            包含原色和白点信息的字典
        """
        if self.ccm_matrix is None:
            raise ValueError("尚未设置优化后的CCM矩阵")
        
        # 获取目标色彩空间信息
        target_primaries, target_whitepoint = self.get_target_colorspace_info(
            target_colorspace, color_space_manager
        )
        
        # 计算原始色彩空间
        source_primaries_xy, source_whitepoint_xy = self.calculate_source_colorspace(
            self.ccm_matrix, target_primaries, target_whitepoint, 
            target_colorspace, color_space_manager
        )
        
        # 存储结果
        self.source_primaries = source_primaries_xy
        self.source_whitepoint = source_whitepoint_xy
        
        return {
            'primaries': {
                'R': [float(source_primaries_xy[0, 0]), float(source_primaries_xy[0, 1])],
                'G': [float(source_primaries_xy[1, 0]), float(source_primaries_xy[1, 1])],
                'B': [float(source_primaries_xy[2, 0]), float(source_primaries_xy[2, 1])]
            },
            'white_point': [float(source_whitepoint_xy[0]), float(source_whitepoint_xy[1])],
            'ccm_matrix': self.ccm_matrix.tolist(),
            'target_colorspace': target_colorspace
        }
    
    def clear_data(self):
        """清除所有数据"""
        self.rgb_values.clear()
        self.ccm_matrix = None
        self.source_primaries = None
        self.source_whitepoint = None