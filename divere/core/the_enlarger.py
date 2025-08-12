"""
胶片放大机引擎
负责所有胶片图像处理操作 - 重构版本
使用分离的数学操作和管线处理器
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
import json
from pathlib import Path
import time
import traceback

from .data_types import ImageData, ColorGradingParams, LUT3D, PreviewConfig
from .math_ops import FilmMathOps
from .pipeline_processor import FilmPipelineProcessor

# 尝试导入深度学习白平衡相关模块
try:
    from ..models.deep_wb_wrapper import create_deep_wb_wrapper
    DEEP_WB_AVAILABLE = True
except ImportError as e:
    print("Failed to import deep_wb_wrapper:")
    traceback.print_exc()
    DEEP_WB_AVAILABLE = False


class TheEnlarger:
    """胶片放大机引擎，负责所有图像处理操作 - 重构版本"""

    def __init__(self, preview_config: Optional[PreviewConfig] = None):
        # 校正矩阵管理
        self._correction_matrices = {}
        self._load_default_matrices()
        
        # 预览配置（统一管理）
        self.preview_config = preview_config or PreviewConfig()
        
        # 核心处理组件
        self.math_ops = FilmMathOps(preview_config=self.preview_config)
        self.pipeline_processor = FilmPipelineProcessor(
            self.math_ops, self.preview_config
        )
        
        # GPU加速器（共享math_ops的实例）
        self.gpu_accelerator = self.math_ops.gpu_accelerator
        
        # 设置矩阵加载器
        self.pipeline_processor.set_matrix_loader(self._load_correction_matrix)
        
        # 深度白平衡相关
        self._deep_wb_wrapper = None
        
        # 性能分析
        self._profiling_enabled: bool = False
        
        if not DEEP_WB_AVAILABLE:
            print("Warning: Deep White Balance not available, learning-based auto gain will be disabled")

    def _load_default_matrices(self):
        """加载默认的校正矩阵（支持用户配置优先）"""
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            
            # 获取所有配置文件（用户配置优先）
            config_files = enhanced_config_manager.get_config_files("matrices")
            
            for matrix_file in config_files:
                try:
                    data = enhanced_config_manager.load_config_file(matrix_file)
                    if data is None:
                        continue
                    
                    # 使用文件名作为键，但保留name字段用于显示
                    matrix_key = matrix_file.stem
                    self._correction_matrices[matrix_key] = data
                    
                    # 标记是否为用户配置
                    if matrix_file.parent == enhanced_config_manager.user_matrices_dir:
                        print(f"加载用户矩阵: {matrix_key}")
                    else:
                        print(f"加载内置矩阵: {matrix_key}")
                        
                except Exception as e:
                    print(f"Failed to load matrix {matrix_file}: {e}")
                    
        except ImportError:
            # 如果增强配置管理器不可用，使用原来的方法
            matrix_dir = Path("config/matrices")
            if not matrix_dir.exists():
                return
            for matrix_file in matrix_dir.glob("*.json"):
                try:
                    with open(matrix_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 使用文件名作为键，但保留name字段用于显示
                        matrix_key = matrix_file.stem
                        self._correction_matrices[matrix_key] = data
                except Exception as e:
                    print(f"Failed to load matrix {matrix_file}: {e}")

    def set_profiling_enabled(self, enabled: bool) -> None:
        """启用/关闭预览管线Profiling"""
        self._profiling_enabled = bool(enabled)
        self.pipeline_processor.set_profiling_enabled(enabled)

    def is_profiling_enabled(self) -> bool:
        return self._profiling_enabled

    # =======================
    # 主要处理接口
    # =======================

    def apply_full_pipeline(self, image: ImageData, params: ColorGradingParams, 
                           include_curve: bool = True) -> ImageData:
        """
        应用完整处理管线（保持向后兼容的接口）
        
        Args:
            image: 输入图像
            params: 处理参数
            include_curve: 是否包含曲线处理
            
        Returns:
            处理后的图像
        """
        if image is None:
            return None
            
        # 使用新的全精度管线处理器
        return self.pipeline_processor.apply_full_precision_pipeline(
            image, params, include_curve=include_curve, use_optimization=True
        )

    def apply_preview_pipeline(self, image: ImageData, params: ColorGradingParams,
                              include_curve: bool = True) -> ImageData:
        """
        应用预览管线（新接口）
        
        Args:
            image: 输入图像
            params: 处理参数
            include_curve: 是否包含曲线处理
            
        Returns:
            处理后的预览图像
        """
        if image is None:
            return None
            
        return self.pipeline_processor.apply_preview_pipeline(
            image, params, include_curve=include_curve
        )

    def apply_density_inversion(self, image: ImageData, gamma: float, dmax: float) -> ImageData:
        """应用密度反转（保持向后兼容的接口）"""
        if image.array is None: 
            return image
        
        result_array = self.math_ops.density_inversion(
            image.array, gamma, dmax, use_optimization=True
        )
        
        return image.copy_with_new_array(result_array)

    # =======================
    # 缓存管理
    # =======================
    
    def clear_caches(self) -> None:
        """清空内部缓存（调试用）"""
        self.math_ops.clear_caches()

    # =======================
    # 自动白平衡
    # =======================

    def calculate_auto_gain_legacy(self, image: ImageData, njet: int = 1, p_norm: float = 6.0, sigma: float = 1.0) -> Tuple[float, float, float, float, float, float]:
        """
        使用通用的颜色恒常性算法 (基于 general_cc.m) 计算自动白平衡的RGB增益。
        - njet=0: Shades of Gray
        - njet=1: 1st-order Gray Edge
        
        返回: (r_gain, g_gain, b_gain, r_illuminant, g_illuminant, b_illuminant)
        """
        if image.array is None or image.array.size == 0:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        

        # 确保图像数据在 [0, 255] 范围内
        img_uint8 = image.array.copy()
        if img_uint8.max() <= 1.0:
            img_uint8 = (img_uint8 * 255).astype(np.uint8)

        # 1. 创建饱和点遮罩
        saturation_mask = np.max(img_uint8, axis=2) >= 255
        dilated_mask = binary_dilation(saturation_mask, iterations=1)
        mask = ~dilated_mask

        img_float = img_uint8.astype(np.float32)

        # 2. 计算导数或进行平滑
        if njet > 0:
            # Gray-Edge: 计算梯度幅度
            dx = gaussian_filter(img_float, sigma, order=(0, 1, 0))
            dy = gaussian_filter(img_float, sigma, order=(1, 0, 0))
            processed_data = np.sqrt(dx**2 + dy**2)
        else:
            # Shades of Gray: 应用高斯模糊
            processed_data = gaussian_filter(img_float, sigma, order=0)

        processed_data = np.abs(processed_data)
        
        # 3. Minkowski范数计算 (应用遮罩)
        illuminant_estimate = np.zeros(3)
        for i in range(3):
            channel_data = processed_data[:, :, i]
            masked_channel = channel_data[mask]
            if masked_channel.size == 0: 
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0) # 如果所有像素都被遮罩
            illuminant_estimate[i] = np.power(np.sum(np.power(masked_channel, p_norm)), 1.0 / p_norm)

        # 4. 归一化光源并计算校正因子 (以G通道为参考)
        if np.any(illuminant_estimate < 1e-10): 
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        
        print(f"  正在自动校色，图片光源估计值为：R={illuminant_estimate[0]:.2f}, G={illuminant_estimate[1]:.2f}, B={illuminant_estimate[2]:.2f}")
        
        # 使用G通道作为参考来计算校正因子
        correction_factors = illuminant_estimate[1] / illuminant_estimate

        # 5. 将校正因子转换为对数空间的增益值
        gains = np.log10(correction_factors)

        # 裁剪增益值，避免极端校正
        gains = np.clip(gains, -1.0, 1.0)

        return (gains[0], gains[1], gains[2], illuminant_estimate[0], illuminant_estimate[1], illuminant_estimate[2])

    def calculate_auto_gain_learning_based(self, image: ImageData) -> Tuple[float, float, float, float, float, float]:
        """
        使用深度学习模型计算自动白平衡的RGB增益。 cr: https://github.com/mahmoudnafifi/Deep_White_Balance/tree/master
        
        返回: (r_gain, g_gain, b_gain, r_illuminant, g_illuminant, b_illuminant)
        """
        if not DEEP_WB_AVAILABLE:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        if image.array is None or image.array.size == 0:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        try:
            # 确保图像数据在 [0, 255] 范围内
            t0 = time.time()
            img_uint8 = image.array.copy()
            if img_uint8.max() <= 1.0:
                img_uint8 = (img_uint8 * 255).astype(np.uint8)

            # 使用深度学习模型进行白平衡
            # 缓存与复用模型，避免每次加载
            if self._deep_wb_wrapper is None:
                # 优先尝试GPU
                try:
                    deep_wb_wrapper = create_deep_wb_wrapper(device='cuda')
                except Exception:
                    deep_wb_wrapper = create_deep_wb_wrapper(device='cpu')
                self._deep_wb_wrapper = deep_wb_wrapper
            deep_wb_wrapper = self._deep_wb_wrapper
            if deep_wb_wrapper is None:
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

            # 应用深度学习白平衡
            # 降低推理输入最大边长以提升速度（例如512），保持效果可用
            inference_size = 128
            t1 = time.time()
            result = deep_wb_wrapper.process_image(img_uint8, max_size=inference_size)
            t2 = time.time()
            
            if result is None:
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

            # 计算增益（原始图像与校正后图像的比值）
            original_mean = np.mean(img_uint8, axis=(0, 1))
            corrected_mean = np.mean(result, axis=(0, 1))
            
            # 避免除以零
            corrected_mean = np.maximum(corrected_mean, 1e-10)
            
            # 计算增益
            gains = np.log10(original_mean / corrected_mean)
            
            # 裁剪增益值
            gains = -np.clip(gains, -2.0, 2.0) + gains[1]  # 这里需要负号，somehow拟合出来的增益是反向的
            
            # 计算光源估计（归一化的原始均值）
            illuminant = original_mean / np.sum(original_mean)
            
            t3 = time.time()
            print(f"AI自动校色耗时: 预处理={(t1 - t0)*1000:.1f}ms, 推理={(t2 - t1)*1000:.1f}ms, 统计/收尾={(t3 - t2)*1000:.1f}ms, 总={(t3 - t0)*1000:.1f}ms")
            
            return (gains[0], gains[1], gains[2], illuminant[0], illuminant[1], illuminant[2])
            
        except Exception as e:
            print(f"Deep White Balance error: {e}")
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # =======================
    # 矩阵管理
    # =======================

    def _load_correction_matrix(self, matrix_file):
        """加载校正矩阵"""
        return self._correction_matrices.get(matrix_file)
    
    def _get_correction_matrix_array(self, matrix_file):
        """获取校正矩阵的numpy数组"""
        matrix_data = self._correction_matrices.get(matrix_file)
        if matrix_data and matrix_data.get("matrix_space") == "density":
            return np.array(matrix_data["matrix"])
        return None

    def get_available_matrices(self) -> List[str]:
        """获取可用的校正矩阵列表"""
        return list(self._correction_matrices.keys())
    
    def reload_matrices(self):
        """重新加载矩阵文件"""
        self._correction_matrices = {}
        self._load_default_matrices()

    # =======================
    # LUT生成
    # =======================
    
    def generate_3d_lut(self, params: ColorGradingParams, lut_size: int = 64,
                       include_curve: bool = True) -> np.ndarray:
        """
        生成3D LUT用于外部应用
        
        Args:
            params: 处理参数
            lut_size: LUT大小（每个维度）
            include_curve: 是否包含曲线
            
        Returns:
            3D LUT数组 [lut_size, lut_size, lut_size, 3]
        """
        # 设置矩阵加载器并生成LUT
        original_get_matrix = self.math_ops._get_correction_matrix
        self.math_ops._get_correction_matrix = lambda p: self._get_correction_matrix_from_params(p)
        
        try:
            return self.pipeline_processor.generate_3d_lut(params, lut_size, include_curve)
        finally:
            self.math_ops._get_correction_matrix = original_get_matrix

    def _get_correction_matrix_from_params(self, params: ColorGradingParams) -> Optional[np.ndarray]:
        """从参数中获取校正矩阵"""
        if params.correction_matrix_file == "custom" and params.correction_matrix is not None:
            return np.array(params.correction_matrix)
        
        return self._get_correction_matrix_array(params.correction_matrix_file)

    # =======================
    # 向后兼容的Legacy方法（标记为弃用）
    # =======================
    
    def _process_in_density_space(self, density_array: np.ndarray, params: ColorGradingParams, 
                                 include_curve: bool = True, profile: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Legacy方法：在密度空间处理（为向后兼容保留）"""
        print("Warning: _process_in_density_space is deprecated. Use pipeline_processor instead.")
        
        # 使用新的数学操作来模拟旧行为
        result = density_array.copy()
        
        # 应用校正矩阵
        if params.enable_correction_matrix and params.correction_matrix_file:
            matrix = self._get_correction_matrix_from_params(params)
            if matrix is not None:
                result = self.math_ops.apply_correction_matrix(result, matrix, params.density_dmax)
        
        # 应用RGB增益
        if params.enable_rgb_gains:
            result = self.math_ops.apply_rgb_gains(result, params.rgb_gains)
        
        # 应用曲线
        if include_curve and params.enable_density_curve:
            # RGB通用曲线
            curve_points = None
            if params.enable_curve and params.curve_points and len(params.curve_points) >= 2:
                curve_points = params.curve_points
            
            # 单通道曲线
            channel_curves = {}
            if params.enable_curve_r and params.curve_points_r:
                channel_curves['r'] = params.curve_points_r
            if params.enable_curve_g and params.curve_points_g:
                channel_curves['g'] = params.curve_points_g
            if params.enable_curve_b and params.curve_points_b:
                channel_curves['b'] = params.curve_points_b
            
            result = self.math_ops.apply_density_curve(result, curve_points, channel_curves)
        
        return result
