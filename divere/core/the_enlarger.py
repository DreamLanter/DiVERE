"""
胶片放大机引擎
负责所有胶片图像处理操作
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
import json
from pathlib import Path
from collections import OrderedDict
import time

from .data_types import ImageData, ColorGradingParams, LUT3D

# 尝试导入深度学习白平衡相关模块
try:
    from ..models.deep_wb_wrapper import create_deep_wb_wrapper
    DEEP_WB_AVAILABLE = True
except ImportError:
    DEEP_WB_AVAILABLE = False


class TheEnlarger:
    """胶片放大机引擎，负责所有图像处理操作"""

    def __init__(self):
        self._correction_matrices = {}
        self._load_default_matrices()
        # 预计算与缓存（用于加速预览）
        self._lut1d_cache: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._curve_lut_cache: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._max_cache_size: int = 64
        self._LOG65536: np.float32 = np.float32(np.log10(65536.0))
        # 缓存深度白平衡包装器，避免重复加载模型
        self._deep_wb_wrapper = None
        # 预览Profiling（可开关）
        self._profiling_enabled: bool = False
        self._last_profile: Dict[str, float] = {}
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

    def is_profiling_enabled(self) -> bool:
        return self._profiling_enabled

    def apply_density_inversion(self, image: ImageData, gamma: float, dmax: float) -> ImageData:
        """应用密度反转（优化版：使用整数索引查表）"""
        if image.array is None: 
            return image
        
        pivot = 0.9  # 以三档曝光为转轴
        
        # 使用更高精度的LUT（65536个点）进行直接索引
        # 这样可以避免插值，直接用整数索引查表
        lut_size = 65536
        lut = self._get_density_inversion_lut_fast(gamma, dmax, pivot, size=lut_size)
        
        # 将浮点图像转换为16位整数索引
        # 限制在[0,1]范围并转换为[0,65535]的整数
        img_clipped = np.clip(image.array, 0.0, 1.0)
        
        # 直接转换为整数索引，避免中间的float32转换
        # 使用round而不是floor以获得更准确的结果
        indices = np.round(img_clipped * (lut_size - 1)).astype(np.uint16)
        
        # 使用高级索引直接查表，这比interp快得多
        # np.take在连续内存上特别快
        result_array = np.take(lut, indices)
        
        # 确保输出dtype与输入一致
        if result_array.dtype != image.array.dtype:
            result_array = result_array.astype(image.array.dtype, copy=False)
        
        return image.copy_with_new_array(result_array)

    def _process_in_density_space(self, density_array: np.ndarray, params: ColorGradingParams, include_curve: bool = True, profile: Optional[Dict[str, float]] = None) -> np.ndarray:
        adjusted_density = density_array.copy()
        if profile is not None:
            profile.clear()
        if params.enable_correction_matrix and params.correction_matrix_file:
            # 处理自定义矩阵
            if params.correction_matrix_file == "custom" and params.correction_matrix is not None:
                t_m0 = time.time()
                matrix = params.correction_matrix
                #print(f"  应用自定义矩阵:\n{matrix}")
                adjusted_density = self._apply_matrix_to_image(adjusted_density + params.density_dmax, matrix) - params.density_dmax
                if profile is not None:
                    profile['matrix_ms'] = (time.time() - t_m0) * 1000.0
            # 处理预设矩阵
            else:
                t_m1 = time.time()
                matrix_data = self._load_correction_matrix(params.correction_matrix_file)
                if matrix_data and matrix_data.get("matrix_space") == "density":
                    matrix = np.array(matrix_data["matrix"])
                    #print(f"  应用预设矩阵 {params.correction_matrix_file}:\n{matrix}")
                    adjusted_density = self._apply_matrix_to_image(adjusted_density + params.density_dmax, matrix) - params.density_dmax
                if profile is not None:
                    profile['matrix_ms'] = profile.get('matrix_ms', 0.0) + (time.time() - t_m1) * 1000.0
        if params.enable_rgb_gains:
            # RGB增益在密度空间的应用
            # 正增益 -> 降低密度（变亮）
            # 负增益 -> 增加密度（变暗）
            # 确保正确广播到每个通道
            t_g = time.time()
            for i, gain in enumerate(params.rgb_gains):
                adjusted_density[:, :, i] -= gain
            if profile is not None:
                profile['rgb_gains_ms'] = (time.time() - t_g) * 1000.0
        
        if include_curve and params.enable_density_curve and params.enable_curve and params.curve_points and len(params.curve_points) >= 2:
            lut_size = 1024
            # 使用与UI一致的单调插值算法生成曲线（带缓存）
            t_curve = time.time()
            lut = self._get_curve_lut_cached(params.curve_points, lut_size)

            # 以下是原算法，现在使用LUT查表
            # # 曲线直接作用在密度空间上
            # # - 曲线输入X：[0, 1] 对应密度范围 [4.816, 0] (暗部到亮部，注意反转)
            # # - 曲线输出Y：[0, 1] 对应密度范围 [0, 3.0]
            # # 左下角(0,0) = 暗部输入->低密度输出（保持暗）
            # # 右上角(1,1) = 亮部输入->高密度输出（保持亮）
            
            # # 定义密度映射范围
            # input_density_min = 0.0
            # input_density_max = np.log10(65536)  # ≈ 4.816
            # output_density_min = 0.0
            # output_density_max = np.log10(65536)
            
            # # 将当前密度值归一化到[0,1]用于查找LUT
            # # 注意：负密度值（超亮）会被钳制到0
            # normalized_density = 1 - np.clip((adjusted_density - input_density_min) / (input_density_max - input_density_min), 0, 1)
            
            # # 查找LUT值
            # lut_indices = np.clip(normalized_density * (lut_size - 1), 0, lut_size - 1).astype(int)
            # curve_output = lut[lut_indices]
            
            # # 将曲线输出映射到输出密度范围
            # adjusted_density = (1 - curve_output) * (output_density_max - output_density_min) + output_density_min


            # 常量与缩放（使用预计算的LOG65536）
            inv_range = 1.0 / float(self._LOG65536)
            # 归一化到[0,1]并取反（注意：负密度值将被钳制到0）
            normalized = 1.0 - np.clip(adjusted_density * inv_range, 0.0, 1.0)
            # 计算LUT索引（int32更快，且与原逻辑等效于floor）
            lut_indices = (normalized * (lut_size - 1)).astype(np.int32)
            # 查表（np.take通常更快）
            curve_output = np.take(lut, lut_indices)
            # 将曲线输出映射回密度范围，原地写回，减少分配
            adjusted_density[:] = (1.0 - curve_output) * float(self._LOG65536)
            if profile is not None:
                profile['rgb_curve_ms'] = (time.time() - t_curve) * 1000.0
        
        # 应用单通道曲线（在RGB曲线之后）
        if include_curve and params.enable_density_curve:
            channel_curves = [
                (params.enable_curve_r, params.curve_points_r),  # R通道
                (params.enable_curve_g, params.curve_points_g),  # G通道
                (params.enable_curve_b, params.curve_points_b)   # B通道
            ]
            
            t_ch = time.time()
            inv_range = 1.0 / float(self._LOG65536)
            for channel_idx, (enabled, curve_points) in enumerate(channel_curves):
                if enabled and curve_points and len(curve_points) >= 2:
                    lut_size = 1024
                    # 生成单通道曲线LUT（带缓存）
                    lut_c = self._get_curve_lut_cached(curve_points, lut_size)
                    # 当前通道视图（避免拷贝）
                    channel_density = adjusted_density[:, :, channel_idx]
                    # 归一化
                    normalized_c = 1.0 - np.clip(channel_density * inv_range, 0.0, 1.0)
                    # 索引
                    lut_indices_c = (normalized_c * (lut_size - 1)).astype(np.int32)
                    # 查表并写回（原地）
                    adjusted_density[:, :, channel_idx] = (1.0 - np.take(lut_c, lut_indices_c)) * float(self._LOG65536)
            if profile is not None:
                profile['channel_curves_ms'] = (time.time() - t_ch) * 1000.0
        
        return adjusted_density

    # =====================
    # 缓存与工具函数
    # =====================
    def clear_caches(self) -> None:
        """清空内部缓存（调试用）"""
        self._lut1d_cache.clear()
        self._curve_lut_cache.clear()

    def _cache_put(self, cache: "OrderedDict[Any, np.ndarray]", key: Any, value: np.ndarray) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._max_cache_size:
            cache.popitem(last=False)

    def _get_density_inversion_lut(self, gamma: float, dmax: float, pivot: float, size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        """获取或生成密度反相1D LUT: x∈[0,1] → y∈[0,1]
        y = 10 ** ( pivot + (-log10(max(x,1e-10)) - pivot) * gamma - dmax )
        """
        key = ("dens_inv", round(float(gamma), 6), round(float(dmax), 6), round(float(pivot), 6), int(size))
        lut_y = self._lut1d_cache.get(key)
        if lut_y is None:
            xs = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
            safe = np.maximum(xs, np.float32(1e-10))
            original_density = -np.log10(safe)
            adjusted_density = pivot + (original_density - pivot) * gamma - dmax
            ys = np.power(np.float32(10.0), adjusted_density).astype(np.float32)
            self._cache_put(self._lut1d_cache, key, ys)
        else:
            xs = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
            ys = lut_y
        return xs, ys
    
    def _get_density_inversion_lut_fast(self, gamma: float, dmax: float, pivot: float, size: int = 65536) -> np.ndarray:
        """获取密度反相LUT（仅返回y值用于直接索引）
        专门为高性能整数索引设计的版本
        """
        key = ("dens_inv_fast", round(float(gamma), 6), round(float(dmax), 6), round(float(pivot), 6), int(size))
        lut = self._lut1d_cache.get(key)
        
        if lut is None:
            # 生成输入值 [0, 1]
            xs = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
            
            # 避免log(0)，使用更小的epsilon
            safe = np.maximum(xs, np.float32(1e-10))
            
            # 密度反相公式
            original_density = -np.log10(safe)
            adjusted_density = pivot + (original_density - pivot) * gamma - dmax
            
            # 使用float32以节省内存和提高缓存效率
            lut = np.power(np.float32(10.0), adjusted_density).astype(np.float32)
            
            # 缓存结果
            self._cache_put(self._lut1d_cache, key, lut)
        
        return lut

    def _get_curve_lut_cached(self, control_points: List[Tuple[float, float]], num_samples: int) -> np.ndarray:
        """获取或生成曲线LUT（单调三次插值后的y值，长度为num_samples）"""
        if not control_points or len(control_points) < 2:
            return np.linspace(0.0, 1.0, int(num_samples), dtype=np.float32)
        key_points = tuple((round(float(x), 6), round(float(y), 6)) for x, y in control_points)
        key = ("curve", key_points, int(num_samples))
        lut = self._curve_lut_cache.get(key)
        if lut is None:
            curve_samples = self._generate_monotonic_curve(control_points, int(num_samples))
            lut = np.array([p[1] for p in curve_samples], dtype=np.float32)
            self._cache_put(self._curve_lut_cache, key, lut)
        return lut

    def apply_full_pipeline(self, image: ImageData, params: ColorGradingParams, include_curve: bool = True) -> ImageData:
        if image is None:
            return None
        result = image.copy()
        t_start = time.time()
        stage: Dict[str, float] = {}

        # 密度反相
        if params.enable_density_inversion:
            t0 = time.time()
            result = self.apply_density_inversion(result, params.density_gamma, params.density_dmax)
            stage['density_inversion_ms'] = (time.time() - t0) * 1000.0
        else:
            stage['density_inversion_ms'] = 0.0

        # 转为密度
        t1 = time.time()
        initial_density = -np.log10(np.maximum(result.array, 1e-10))
        stage['to_density_ms'] = (time.time() - t1) * 1000.0

        # 在密度空间处理
        t2 = time.time()
        sub_profile: Dict[str, float] = {}
        final_density = self._process_in_density_space(initial_density, params, include_curve, sub_profile)
        stage['process_density_ms'] = (time.time() - t2) * 1000.0

        # 回到线性
        t3 = time.time()
        final_array = np.power(10, -final_density)
        stage['to_linear_ms'] = (time.time() - t3) * 1000.0

        # 裁剪
        t4 = time.time()
        final_array = np.clip(final_array, 0.0, 1.0)
        stage['clip_ms'] = (time.time() - t4) * 1000.0

        total_ms = (time.time() - t_start) * 1000.0
        stage['total_pipeline_ms'] = total_ms
        for k, v in sub_profile.items():
            stage[f"process/{k}"] = v
        self._last_profile = stage

        if self._profiling_enabled:
            print(
                "预览管线Profiling: "
                f"密度反相={stage['density_inversion_ms']:.1f}ms, "
                f"转为密度={stage['to_density_ms']:.1f}ms, "
                f"密度处理={stage['process_density_ms']:.1f}ms "
                f"(矩阵={stage.get('process/matrix_ms', 0.0):.1f}ms, "
                f"RGB增益={stage.get('process/rgb_gains_ms', 0.0):.1f}ms, "
                f"RGB曲线={stage.get('process/rgb_curve_ms', 0.0):.1f}ms, "
                f"单通道曲线={stage.get('process/channel_curves_ms', 0.0):.1f}ms), "
                f"回到线性={stage['to_linear_ms']:.1f}ms, 裁剪={stage['clip_ms']:.1f}ms, 总={stage['total_pipeline_ms']:.1f}ms"
            )

        return image.copy_with_new_array(final_array)

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
            
            # print(f"  深度学习自动校色，光源估计值为：R={illuminant[0]:.2f}, G={illuminant[1]:.2f}, B={illuminant[2]:.2f}")
            # print(f"  计算出的增益：R={gains[0]:.3f}, G={gains[1]:.3f}, B={gains[2]:.3f}")
            
            return (gains[0], gains[1], gains[2], illuminant[0], illuminant[1], illuminant[2])
            
        except Exception as e:
            print(f"Deep White Balance error: {e}")
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    def _monotonic_cubic_interpolate(self, x: float, points: List[Tuple[float, float]]) -> float:
        """
        单调三次插值，确保曲线单调递增
        基于论文: "Monotonic piecewise cubic interpolation" by Fritsch & Carlson
        """
        if len(points) < 2:
            return x
        
        # 找到x所在的区间
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            if x1 <= x <= x2:
                # 计算区间内的插值
                t = (x - x1) / (x2 - x1) if x2 > x1 else 0
                
                # 使用单调三次插值
                # 计算端点导数（使用有限差分）
                if i == 0:
                    # 第一个区间，使用前向差分
                    h1 = x2 - x1
                    h2 = points[i + 2][0] - x2 if i + 2 < len(points) else h1
                    m0 = (y2 - y1) / h1  # 第一个区间，m0 = m1
                    m1 = (y2 - y1) / h1
                    m2 = (points[i + 2][1] - y2) / h2 if i + 2 < len(points) else m1
                elif i == len(points) - 2:
                    # 最后一个区间，使用后向差分
                    h0 = x1 - points[i - 1][0]
                    h1 = x2 - x1
                    m0 = (y1 - points[i - 1][1]) / h0
                    m1 = (y2 - y1) / h1
                    m2 = m1
                else:
                    # 中间区间，使用中心差分
                    h0 = x1 - points[i - 1][0]
                    h1 = x2 - x1
                    h2 = points[i + 2][0] - x2 if i + 2 < len(points) else h1
                    m0 = (y1 - points[i - 1][1]) / h0
                    m1 = (y2 - y1) / h1
                    m2 = (points[i + 2][1] - y2) / h2 if i + 2 < len(points) else m1
                
                # 计算单调性约束的导数
                if m0 * m1 <= 0:
                    m0 = 0
                if m1 * m2 <= 0:
                    m2 = 0
                
                # 使用Hermite插值
                t2 = t * t
                t3 = t2 * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                
                return h00 * y1 + h10 * h1 * m1 + h01 * y2 + h11 * h1 * m2
        
        # 如果x超出范围，返回最近端点的值
        if x <= points[0][0]:
            return points[0][1]
        else:
            return points[-1][1]

    def _generate_monotonic_curve(self, control_points, num_samples):
        """生成单调曲线样本点"""
        if len(control_points) < 2:
            return [(i / (num_samples - 1), i / (num_samples - 1)) for i in range(num_samples)]
        
        samples = []
        for i in range(num_samples):
            x = i / (num_samples - 1)
            y = self._monotonic_cubic_interpolate(x, control_points)
            samples.append((x, y))
        
        return samples

    def _load_correction_matrix(self, matrix_file):
        """加载校正矩阵"""
        return self._correction_matrices.get(matrix_file)

    def _apply_matrix_to_image(self, image_array, matrix):
        """将矩阵应用到图像"""
        pivot = 0.9 #以三档曝光为转轴 
        applyed_array = pivot + np.dot(image_array.reshape(-1, 3) - pivot, matrix.T).reshape(image_array.shape)

        return applyed_array

    def get_available_matrices(self) -> List[str]:
        """获取可用的校正矩阵列表"""
        return list(self._correction_matrices.keys())
    
    def reload_matrices(self):
        """重新加载矩阵文件"""
        self._correction_matrices = {}
        self._load_default_matrices() 