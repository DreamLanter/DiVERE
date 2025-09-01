"""
RGB到Delta E 1994的转换模块 - 高性能版本
提供色差计算接口，支持多种色彩空间，针对批量计算优化
"""

from typing import Tuple, Union, Dict
import numpy as np
from colour import RGB_COLOURSPACES, delta_E
from functools import lru_cache

# 尝试导入numba进行JIT加速（跨平台兼容）
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# 全局缓存的变换矩阵和白点
_COLORSPACE_CACHE: Dict[str, Dict] = {}

# 优化器专用缓存（针对24色块优化）
_OPTIMIZER_CACHE: Dict[str, np.ndarray] = {}


def _init_colorspace_cache():
    """初始化色彩空间缓存，避免重复查询"""
    global _COLORSPACE_CACHE
    
    if not _COLORSPACE_CACHE:
        for cs_name in ['ACEScg', 'sRGB', 'Display P3']:
            if cs_name in RGB_COLOURSPACES:
                cs = RGB_COLOURSPACES[cs_name]
                # 将xy白点转换为XYZ白点 (Y=1)
                wp_xy = np.array(cs.whitepoint, dtype=np.float32)
                wp_xyz = np.array([wp_xy[0]/wp_xy[1], 1.0, (1-wp_xy[0]-wp_xy[1])/wp_xy[1]], dtype=np.float32)
                
                _COLORSPACE_CACHE[cs_name] = {
                    'matrix_rgb_to_xyz': np.array(cs.matrix_RGB_to_XYZ, dtype=np.float32),
                    'whitepoint_xyz': wp_xyz
                }


@lru_cache(maxsize=128)
def _get_cached_colorspace_data(colorspace: str) -> Tuple[np.ndarray, np.ndarray]:
    """缓存的色彩空间数据获取"""
    if not _COLORSPACE_CACHE:
        _init_colorspace_cache()
    
    if colorspace not in _COLORSPACE_CACHE:
        raise ValueError(f"不支持的色彩空间: {colorspace}")
    
    data = _COLORSPACE_CACHE[colorspace]
    return data['matrix_rgb_to_xyz'], data['whitepoint_xyz']


def calculate_delta_e_1994(
    rgb1: Union[Tuple[float, float, float], np.ndarray],
    rgb2: Union[Tuple[float, float, float], np.ndarray],
    colorspace1: str = 'ACEScg',
    colorspace2: str = 'ACEScg'
) -> float:
    """
    计算两个RGB颜色之间的CIE Delta E 1994色差 - 高性能版本
    
    Args:
        rgb1: 第一个RGB颜色 (R, G, B)
        rgb2: 第二个RGB颜色 (R, G, B)  
        colorspace1: rgb1的色彩空间名称
        colorspace2: rgb2的色彩空间名称
        
    Returns:
        Delta E 1994值，越小表示颜色越接近
    """
    try:
        # 快速验证和转换
        rgb1_array = np.clip(np.asarray(rgb1, dtype=np.float32), 0.0, None)
        rgb2_array = np.clip(np.asarray(rgb2, dtype=np.float32), 0.0, None)
        
        # 快速Lab转换
        lab1 = _rgb_to_lab_fast(rgb1_array, colorspace1)
        lab2 = _rgb_to_lab_fast(rgb2_array, colorspace2)
        
        # 计算Delta E 1994
        delta_e = delta_E(lab1, lab2, method='CIE 1994')
        
        return float(delta_e)
        
    except Exception:
        return 100.0


def _rgb_to_lab_fast(rgb: np.ndarray, colorspace: str) -> np.ndarray:
    """
    高性能RGB到Lab转换，使用缓存的矩阵计算
    
    Args:
        rgb: RGB数组 (3,)
        colorspace: 色彩空间名称
        
    Returns:
        Lab数组 (3,)
    """
    matrix, whitepoint = _get_cached_colorspace_data(colorspace)
    
    # 快速矩阵乘法 RGB → XYZ
    xyz = np.dot(matrix, rgb.astype(np.float32))
    
    # 快速XYZ → Lab转换
    lab = _xyz_to_lab_fast(xyz, whitepoint)
    
    return lab


@jit(nopython=True, cache=True)
def _xyz_to_lab_fast_numba(xyz: np.ndarray, whitepoint: np.ndarray) -> np.ndarray:
    """
    Numba JIT加速的XYZ到Lab转换
    """
    # 归一化到白点
    xyz_norm = xyz / whitepoint
    
    # f函数常数
    delta = 6.0 / 29.0
    delta_cubed = delta * delta * delta
    linear_slope = 1.0 / (3.0 * delta * delta)
    linear_offset = 4.0/29.0
    
    # f函数计算
    fx = xyz_norm[0]**(1.0/3.0) if xyz_norm[0] > delta_cubed else xyz_norm[0] * linear_slope + linear_offset
    fy = xyz_norm[1]**(1.0/3.0) if xyz_norm[1] > delta_cubed else xyz_norm[1] * linear_slope + linear_offset
    fz = xyz_norm[2]**(1.0/3.0) if xyz_norm[2] > delta_cubed else xyz_norm[2] * linear_slope + linear_offset
    
    # Lab计算
    l_val = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return np.array([l_val, a_val, b_val], dtype=np.float32)


def _xyz_to_lab_fast(xyz: np.ndarray, whitepoint: np.ndarray) -> np.ndarray:
    """
    高性能XYZ到Lab转换，优先使用Numba加速
    """
    if NUMBA_AVAILABLE:
        return _xyz_to_lab_fast_numba(xyz, whitepoint)
    
    # Fallback到numpy版本
    xyz_norm = xyz / whitepoint
    
    delta = 6.0 / 29.0
    delta_cubed = delta**3
    linear_slope = 1.0 / (3.0 * delta**2)
    linear_offset = 4.0/29.0
    
    f_values = np.where(xyz_norm > delta_cubed,
                       np.cbrt(xyz_norm),
                       xyz_norm * linear_slope + linear_offset)
    
    fx, fy, fz = f_values[0], f_values[1], f_values[2]
    
    l_val = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return np.array([l_val, a_val, b_val], dtype=np.float32)


def calculate_batch_delta_e_1994(
    rgb_pairs: list,
    colorspace1: str = 'ACEScg',
    colorspace2: str = 'ACEScg'
) -> np.ndarray:
    """
    批量计算Delta E 1994，极致优化版本
    
    Args:
        rgb_pairs: RGB颜色对列表 [(rgb1, rgb2), ...]
        colorspace1: 第一组RGB的色彩空间
        colorspace2: 第二组RGB的色彩空间
        
    Returns:
        Delta E数组
    """
    if not rgb_pairs:
        return np.array([])
    
    # 缓存键
    cache_key = f"{colorspace1}_{colorspace2}_{len(rgb_pairs)}"
    
    try:
        # 快速路径：如果两个色彩空间相同，可以进一步优化
        if colorspace1 == colorspace2:
            return _calculate_same_colorspace_batch(rgb_pairs, colorspace1)
        
        # 不同色彩空间的通用路径
        return _calculate_different_colorspace_batch(rgb_pairs, colorspace1, colorspace2)
        
    except Exception:
        return np.full(len(rgb_pairs), 100.0, dtype=np.float32)


def _calculate_same_colorspace_batch(rgb_pairs: list, colorspace: str) -> np.ndarray:
    """同一色彩空间的快速批量计算"""
    matrix, whitepoint = _get_cached_colorspace_data(colorspace)
    
    # 预分配数组
    n_pairs = len(rgb_pairs)
    
    # 批量提取RGB数据
    rgb1_batch = np.zeros((n_pairs, 3), dtype=np.float32)
    rgb2_batch = np.zeros((n_pairs, 3), dtype=np.float32)
    
    for i, (rgb1, rgb2) in enumerate(rgb_pairs):
        rgb1_batch[i] = np.clip(np.asarray(rgb1, dtype=np.float32), 0.0, None)
        rgb2_batch[i] = np.clip(np.asarray(rgb2, dtype=np.float32), 0.0, None)
    
    # 批量转换 RGB → XYZ → Lab
    xyz1_batch = rgb1_batch @ matrix.T
    xyz2_batch = rgb2_batch @ matrix.T
    
    lab1_batch = _xyz_to_lab_batch_fast(xyz1_batch, whitepoint)
    lab2_batch = _xyz_to_lab_batch_fast(xyz2_batch, whitepoint)
    
    # 批量Delta E计算
    delta_es = delta_E(lab1_batch, lab2_batch, method='CIE 1994')
    
    return delta_es.astype(np.float32)


def _calculate_different_colorspace_batch(rgb_pairs: list, colorspace1: str, colorspace2: str) -> np.ndarray:
    """不同色彩空间的批量计算"""
    # 批量转换RGB到数组
    rgb1_array = np.array([np.clip(np.asarray(pair[0], dtype=np.float32), 0.0, None) 
                          for pair in rgb_pairs])
    rgb2_array = np.array([np.clip(np.asarray(pair[1], dtype=np.float32), 0.0, None)
                          for pair in rgb_pairs])
    
    # 批量转换到Lab
    lab1_array = _rgb_to_lab_batch_fast(rgb1_array, colorspace1)
    lab2_array = _rgb_to_lab_batch_fast(rgb2_array, colorspace2)
    
    # 批量计算Delta E
    delta_es = delta_E(lab1_array, lab2_array, method='CIE 1994')
    
    return delta_es.astype(np.float32)


def calculate_colorchecker_delta_e_optimized(
    reference_patches: Dict[str, Tuple[float, float, float]],
    output_patches: Dict[str, Tuple[float, float, float]],
    weights: Dict[str, float],
    colorspace: str = 'ACEScg'
) -> Tuple[float, Dict[str, float]]:
    """
    ColorChecker专用的极致优化Delta E计算
    针对24色块场景特别优化，避免字典遍历开销
    
    Args:
        reference_patches: 参考色块RGB值
        output_patches: 输出色块RGB值
        weights: 色块权重
        colorspace: 色彩空间
        
    Returns:
        (加权平均Delta E, 各色块Delta E字典)
    """
    # 预先排序的色块ID列表（避免每次重建）
    patch_ids = sorted(reference_patches.keys())
    
    # 批量构建RGB数组
    n_patches = len(patch_ids)
    ref_rgb_array = np.zeros((n_patches, 3), dtype=np.float32)
    out_rgb_array = np.zeros((n_patches, 3), dtype=np.float32)
    weights_array = np.zeros(n_patches, dtype=np.float32)
    
    valid_count = 0
    for i, patch_id in enumerate(patch_ids):
        if patch_id in output_patches:
            ref_rgb_array[valid_count] = np.clip(reference_patches[patch_id], 0.0, None)
            out_rgb_array[valid_count] = np.clip(output_patches[patch_id], 0.0, None)
            weights_array[valid_count] = weights.get(patch_id, 1.0)
            valid_count += 1
    
    if valid_count == 0:
        return float('inf'), {}
    
    # 截取有效数据
    ref_rgb_array = ref_rgb_array[:valid_count]
    out_rgb_array = out_rgb_array[:valid_count]
    weights_array = weights_array[:valid_count]
    
    # 获取转换矩阵
    matrix, whitepoint = _get_cached_colorspace_data(colorspace)
    
    # 批量RGB → XYZ → Lab
    ref_xyz = ref_rgb_array @ matrix.T
    out_xyz = out_rgb_array @ matrix.T
    
    ref_lab = _xyz_to_lab_batch_fast(ref_xyz, whitepoint)
    out_lab = _xyz_to_lab_batch_fast(out_xyz, whitepoint)
    
    # 批量Delta E计算
    delta_es = delta_E(ref_lab, out_lab, method='CIE 1994')
    
    # 加权平均
    total_weight = np.sum(weights_array)
    if total_weight <= 0.0:
        return float('inf'), {}
    
    weighted_avg = float(np.sum(delta_es * weights_array) / total_weight)
    
    # 构建结果字典
    result_dict = {}
    for i, patch_id in enumerate(patch_ids[:valid_count]):
        result_dict[patch_id] = float(delta_es[i])
    
    return weighted_avg, result_dict


def _rgb_to_lab_batch_fast(rgb_array: np.ndarray, colorspace: str) -> np.ndarray:
    """
    批量RGB到Lab转换，向量化计算
    
    Args:
        rgb_array: RGB数组 (N, 3)
        colorspace: 色彩空间名称
        
    Returns:
        Lab数组 (N, 3)
    """
    matrix, whitepoint = _get_cached_colorspace_data(colorspace)
    
    # 批量矩阵乘法 RGB → XYZ
    xyz_array = np.dot(rgb_array, matrix.T)
    
    # 批量XYZ → Lab转换
    lab_array = _xyz_to_lab_batch_fast(xyz_array, whitepoint)
    
    return lab_array


def _xyz_to_lab_batch_fast(xyz_array: np.ndarray, whitepoint: np.ndarray) -> np.ndarray:
    """
    批量XYZ到Lab转换，向量化计算
    
    Args:
        xyz_array: XYZ数组 (N, 3)
        whitepoint: 白点XYZ (3,)
        
    Returns:
        Lab数组 (N, 3)
    """
    # 批量归一化到白点
    xyz_norm = xyz_array / whitepoint[np.newaxis, :]
    
    # 向量化f函数
    delta = 6.0 / 29.0
    delta_cubed = delta**3
    linear_slope = 1.0 / (3.0 * delta**2)
    linear_offset = 4.0/29.0
    
    # 使用where进行向量化条件计算
    f_values = np.where(xyz_norm > delta_cubed,
                       np.cbrt(xyz_norm),
                       xyz_norm * linear_slope + linear_offset)
    
    fx = f_values[:, 0]
    fy = f_values[:, 1] 
    fz = f_values[:, 2]
    
    # 批量Lab计算
    l_val = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return np.column_stack([l_val, a_val, b_val]).astype(np.float32)


# 测试函数
def _test_delta_e():
    """简单的测试函数"""
    import time
    
    # 初始化缓存
    _init_colorspace_cache()
    
    # 测试相同颜色
    white = (1.0, 1.0, 1.0)
    delta_e_same = calculate_delta_e_1994(white, white)
    print(f"相同颜色的Delta E: {delta_e_same}")
    
    # 测试不同颜色
    black = (0.0, 0.0, 0.0)
    delta_e_diff = calculate_delta_e_1994(white, black)
    print(f"黑白的Delta E: {delta_e_diff}")
    
    # 测试边界情况
    negative = (-0.1, 0.5, 1.2)
    delta_e_boundary = calculate_delta_e_1994(negative, white)
    print(f"边界值的Delta E: {delta_e_boundary}")
    
    # 性能测试
    rgb1 = (0.5, 0.5, 0.5)
    rgb2 = (0.6, 0.4, 0.7)
    
    # 单次计算性能
    start = time.time()
    for _ in range(1000):
        calculate_delta_e_1994(rgb1, rgb2)
    single_time = time.time() - start
    print(f"1000次单次计算: {single_time*1000:.2f}ms")
    
    # 批量计算性能
    pairs = [(rgb1, rgb2)] * 1000
    start = time.time()
    calculate_batch_delta_e_1994(pairs)
    batch_time = time.time() - start
    print(f"1000对批量计算: {batch_time*1000:.2f}ms")
    print(f"性能提升: {single_time/batch_time:.1f}x")


if __name__ == "__main__":
    _test_delta_e()