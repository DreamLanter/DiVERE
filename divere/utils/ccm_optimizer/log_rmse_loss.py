#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log-RMSE 损失函数模块

实现基于对数 RMSE 的颜色差异计算，替代传统的 Delta E 1994。
这种方法：
1. 避免了复杂的色彩空间转换
2. 不依赖白点和色度适应
3. 直接在 RGB 空间操作
4. 数学上更稳定且符合人眼感知

损失函数：RMSE(log(c + ε) - log(r + ε))
其中 c 是计算颜色，r 是参考颜色，ε 是防零常数
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union


def calculate_log_rmse(
    color1: Union[Tuple[float, float, float], np.ndarray], 
    color2: Union[Tuple[float, float, float], np.ndarray],
    weights: Optional[np.ndarray] = None,
    epsilon: float = 1e-6
) -> float:
    """
    计算两个颜色之间的 log-RMSE 损失
    
    Args:
        color1: 第一个颜色 (R, G, B)
        color2: 第二个颜色 (R, G, B)
        weights: 已废弃，保留参数仅为兼容性
        epsilon: 防零常数，避免 log(0)
        
    Returns:
        Log-RMSE 值，越小表示颜色越接近
    """
    # 转换为 numpy 数组并确保非负
    c1 = np.asarray(color1, dtype=np.float64)
    c2 = np.asarray(color2, dtype=np.float64)
    
    # 确保颜色值非负（避免负数取对数）
    c1 = np.maximum(c1, 0.0)
    c2 = np.maximum(c2, 0.0)
    
    # 计算对数值（加 epsilon 避免 log(0)）
    log1 = np.log(c1 + epsilon)
    log2 = np.log(c2 + epsilon)
    
    # 计算差值
    diff = log1 - log2
    
    # 移除channel权重机制，直接计算RGB三通道的RMSE
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def calculate_colorchecker_log_rmse(
    reference_patches: Dict[str, Tuple[float, float, float]],
    output_patches: Dict[str, Tuple[float, float, float]], 
    weights: Dict[str, float],
    channel_weights: Optional[np.ndarray] = None,
    epsilon: float = 1e-6
) -> Tuple[float, Dict[str, float]]:
    """
    ColorChecker 专用的批量 log-RMSE 计算
    
    Args:
        reference_patches: 参考色块 RGB 值
        output_patches: 输出色块 RGB 值
        weights: 色块权重
        channel_weights: 已废弃，保留参数仅为兼容性
        epsilon: 防零常数
        
    Returns:
        (加权平均 log-RMSE, 各色块 log-RMSE 字典)
    """
    # 预先排序的色块ID列表
    patch_ids = sorted(reference_patches.keys())
    
    # 批量构建数组
    valid_patches = []
    ref_colors = []
    out_colors = []
    patch_weights = []
    
    for patch_id in patch_ids:
        if patch_id in output_patches:
            valid_patches.append(patch_id)
            ref_colors.append(reference_patches[patch_id])
            out_colors.append(output_patches[patch_id])
            patch_weights.append(weights.get(patch_id, 1.0))
    
    if not valid_patches:
        return float('inf'), {}
    
    # 转换为 numpy 数组
    ref_array = np.array(ref_colors, dtype=np.float64)
    out_array = np.array(out_colors, dtype=np.float64)
    weights_array = np.array(patch_weights, dtype=np.float64)
    
    # 确保颜色值非负
    ref_array = np.maximum(ref_array, 0.0)
    out_array = np.maximum(out_array, 0.0)
    
    # 批量计算对数
    log_ref = np.log(ref_array + epsilon)
    log_out = np.log(out_array + epsilon)
    
    # 计算差值
    diff = log_ref - log_out  # shape: (N, 3)
    
    # 移除channel权重机制，直接计算每个色块的RMSE
    patch_rmse = np.sqrt(np.mean(diff ** 2, axis=1))  # shape: (N,)
    
    # 计算加权平均
    total_weight = np.sum(weights_array)
    if total_weight <= 0.0:
        return float('inf'), {}
    
    weighted_avg_rmse = float(np.sum(patch_rmse * weights_array) / total_weight)
    
    # 构建结果字典
    result_dict = {}
    for i, patch_id in enumerate(valid_patches):
        result_dict[patch_id] = float(patch_rmse[i])
    
    return weighted_avg_rmse, result_dict


def calculate_batch_log_rmse(
    color_pairs: list,
    channel_weights: Optional[np.ndarray] = None,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    批量计算多对颜色的 log-RMSE
    
    Args:
        color_pairs: 颜色对列表 [(color1, color2), ...]
        channel_weights: 可选的RGB通道权重
        epsilon: 防零常数
        
    Returns:
        log-RMSE 数组
    """
    if not color_pairs:
        return np.array([])
    
    # 批量提取颜色数据
    colors1 = np.array([np.maximum(np.asarray(pair[0], dtype=np.float64), 0.0) 
                       for pair in color_pairs])
    colors2 = np.array([np.maximum(np.asarray(pair[1], dtype=np.float64), 0.0)
                       for pair in color_pairs])
    
    # 批量计算对数
    log1 = np.log(colors1 + epsilon)
    log2 = np.log(colors2 + epsilon)
    
    # 计算差值
    diff = log1 - log2
    
    # 移除channel权重机制，直接计算每对的RMSE
    rmse_values = np.sqrt(np.mean(diff ** 2, axis=1))
    
    return rmse_values.astype(np.float64)


def validate_color_range(color: Union[Tuple, np.ndarray], 
                        min_val: float = 0.0, 
                        max_val: float = 1.0) -> bool:
    """
    验证颜色值是否在有效范围内
    
    Args:
        color: 颜色值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        是否在有效范围内
    """
    color_array = np.asarray(color)
    return np.all(color_array >= min_val) and np.all(color_array <= max_val)


# 测试函数
def _test_log_rmse():
    """简单的测试函数"""
    print("测试 log-RMSE 函数")
    
    # 测试相同颜色
    white = (1.0, 1.0, 1.0)
    same_rmse = calculate_log_rmse(white, white)
    print(f"相同颜色的 log-RMSE: {same_rmse:.6f}")
    
    # 测试不同颜色
    black = (0.1, 0.1, 0.1)  # 避免真正的黑色 (0,0,0)
    diff_rmse = calculate_log_rmse(white, black)
    print(f"白色到近黑色的 log-RMSE: {diff_rmse:.6f}")
    
    # 测试带权重
    weighted_rmse = calculate_log_rmse(white, black, weights=np.array([2.0, 1.0, 0.5]))
    print(f"带权重的 log-RMSE: {weighted_rmse:.6f}")
    
    # 测试 ColorChecker 批量计算
    ref_patches = {
        'A1': (0.5, 0.3, 0.2),
        'A2': (0.8, 0.6, 0.4),
        'A3': (0.2, 0.7, 0.3)
    }
    out_patches = {
        'A1': (0.52, 0.31, 0.19),
        'A2': (0.79, 0.58, 0.42),
        'A3': (0.21, 0.71, 0.29)
    }
    weights = {'A1': 1.0, 'A2': 1.0, 'A3': 1.0}
    
    avg_rmse, patch_rmse = calculate_colorchecker_log_rmse(
        ref_patches, out_patches, weights
    )
    print(f"ColorChecker 平均 log-RMSE: {avg_rmse:.6f}")
    for patch_id, rmse in patch_rmse.items():
        print(f"  {patch_id}: {rmse:.6f}")


if __name__ == "__main__":
    _test_log_rmse()