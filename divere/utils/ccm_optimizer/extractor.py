#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ColorChecker色块提取器

从原图中根据四角点提取24个色块的平均RGB值。
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# ColorChecker 24色块布局 (4行6列)
COLORCHECKER_LAYOUT = {
    'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
    'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
    'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
    'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5)
}

def calculate_homography_matrix(corners: List[Tuple[float, float]]) -> np.ndarray:
    """
    计算从单位正方形到四角点的单应性矩阵。
    
    Args:
        corners: 四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 顺序：左上、右上、右下、左下
    
    Returns:
        3x3单应性矩阵
    """
    # 单位正方形的四个角点
    unit_square = np.array([
        [0.0, 0.0],  # 左上
        [1.0, 0.0],  # 右上
        [1.0, 1.0],  # 右下
        [0.0, 1.0]   # 左下
    ], dtype=np.float32)
    
    # 目标四角点
    target_corners = np.array(corners, dtype=np.float32)
    
    # 计算单应性矩阵
    homography_matrix = cv2.getPerspectiveTransform(unit_square, target_corners)
    
    return homography_matrix

def transform_patch_coordinates(patch_row: int, patch_col: int, 
                              homography_matrix: np.ndarray,
                              sample_margin: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    将色块的网格坐标转换为图像像素坐标。
    
    Args:
        patch_row: 色块行号 (0-3)
        patch_col: 色块列号 (0-5)
        homography_matrix: 单应性矩阵
        sample_margin: 采样边距比例 (0.3表示中心30%区域)
    
    Returns:
        (patch_corners, sample_corners): 色块四角和采样区域四角的像素坐标
    """
    # 计算色块在单位坐标系中的边界
    patch_width, patch_height = 1.0 / 6, 1.0 / 4
    left = patch_col * patch_width
    top = patch_row * patch_height
    right = left + patch_width
    bottom = top + patch_height
    
    # 色块四角 (单位坐标)
    patch_unit_corners = np.array([
        [left, top],     # 左上
        [right, top],    # 右上
        [right, bottom], # 右下
        [left, bottom]   # 左下
    ], dtype=np.float32)
    
    # 采样区域四角 (缩小到中心区域)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half_sample_width = patch_width * sample_margin / 2
    half_sample_height = patch_height * sample_margin / 2
    
    sample_unit_corners = np.array([
        [center_x - half_sample_width, center_y - half_sample_height],  # 左上
        [center_x + half_sample_width, center_y - half_sample_height],  # 右上
        [center_x + half_sample_width, center_y + half_sample_height],  # 右下
        [center_x - half_sample_width, center_y + half_sample_height]   # 左下
    ], dtype=np.float32)
    
    # 转换到图像像素坐标
    patch_corners = cv2.perspectiveTransform(
        patch_unit_corners.reshape(-1, 1, 2), homography_matrix
    ).reshape(-1, 2)
    
    sample_corners = cv2.perspectiveTransform(
        sample_unit_corners.reshape(-1, 1, 2), homography_matrix
    ).reshape(-1, 2)
    
    return patch_corners, sample_corners

def extract_patch_rgb(image_array: np.ndarray, sample_corners: np.ndarray) -> Tuple[float, float, float]:
    """
    从图像中提取指定区域的平均RGB值。
    
    Args:
        image_array: 图像数组 (H, W, 3)，值范围 [0.0, 1.0]
        sample_corners: 采样区域四角坐标 (4, 2)
    
    Returns:
        (R, G, B): 平均RGB值
    """
    # 创建掩码
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 将采样区域坐标转换为整数
    corners_int = np.round(sample_corners).astype(np.int32)
    
    # 确保坐标在图像范围内
    corners_int[:, 0] = np.clip(corners_int[:, 0], 0, width - 1)
    corners_int[:, 1] = np.clip(corners_int[:, 1], 0, height - 1)
    
    # 填充多边形掩码
    cv2.fillPoly(mask, [corners_int], 255)
    
    # 计算掩码区域的平均RGB值
    masked_pixels = image_array[mask > 0]
    
    if len(masked_pixels) == 0:
        print(f"Warning: 采样区域为空，使用默认值")
        return (0.5, 0.5, 0.5)
    
    avg_rgb = np.mean(masked_pixels, axis=0)
    
    return tuple(avg_rgb.tolist())

def extract_colorchecker_patches(image_array: np.ndarray, 
                                corners: List[Tuple[float, float]],
                                sample_margin: float = 0.3) -> Dict[str, Tuple[float, float, float]]:
    """
    从原图中提取ColorChecker 24个色块的平均RGB值。
    
    Args:
        image_array: 原图像数组 (H, W, 3)，值范围 [0.0, 1.0]
        corners: ColorChecker四角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 顺序：左上、右上、右下、左下
        sample_margin: 采样边距比例，默认0.3表示使用中心30%区域
    
    Returns:
        字典：{patch_id: (R, G, B)}，包含24个色块的RGB值
    """
    print(f"[DEBUG] === 开始提取ColorChecker色块 ===")
    print(f"[DEBUG] 图像尺寸: {image_array.shape}")
    print(f"[DEBUG] 图像数据类型: {image_array.dtype}")
    print(f"[DEBUG] 图像数值范围: [{image_array.min():.4f}, {image_array.max():.4f}]")
    print(f"[DEBUG] 四角点: {corners}")
    print(f"[DEBUG] 采样边距: {sample_margin}")
    
    # 计算单应性矩阵
    homography_matrix = calculate_homography_matrix(corners)
    print(f"[DEBUG] 单应性矩阵:")
    print(f"[DEBUG] {homography_matrix}")
    
    # 提取各色块
    patches_rgb = {}
    
    # 先提取几个关键色块进行验证
    test_patches = ['A1', 'D1', 'D6', 'B3']  # 深皮肤、白、黑、红色
    
    for patch_id, (row, col) in COLORCHECKER_LAYOUT.items():
        try:
            # 获取色块和采样区域坐标
            patch_corners, sample_corners = transform_patch_coordinates(
                row, col, homography_matrix, sample_margin
            )
            
            # 提取RGB值
            rgb = extract_patch_rgb(image_array, sample_corners)
            patches_rgb[patch_id] = rgb
            
            # 对关键色块输出详细信息
            if patch_id in test_patches:
                print(f"[DEBUG] {patch_id} (第{row+1}行第{col+1}列):")
                print(f"[DEBUG]   采样区域: {sample_corners}")
                print(f"[DEBUG]   RGB值: ({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})")
                
                # 检查RGB值是否合理
                if any(v < 0 or v > 1 for v in rgb):
                    print(f"[DEBUG]   警告: RGB值超出[0,1]范围!")
                if all(abs(v - 0.5) < 0.01 for v in rgb):
                    print(f"[DEBUG]   警告: RGB值接近默认值，可能提取失败!")
            else:
                print(f"[DEBUG] {patch_id}: RGB({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})")
            
        except Exception as e:
            print(f"[DEBUG] Error: 提取色块 {patch_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            patches_rgb[patch_id] = (0.5, 0.5, 0.5)  # 默认值
    
    print(f"[DEBUG] === 色块提取完成，共 {len(patches_rgb)} 个色块 ===")
    
    # 简单的色块数据验证
    valid_patches = sum(1 for rgb in patches_rgb.values() if not all(abs(v - 0.5) < 0.01 for v in rgb))
    print(f"[DEBUG] 有效色块数量: {valid_patches}/{len(patches_rgb)}")
    
    return patches_rgb

def validate_extraction_setup(image_array: Optional[np.ndarray],
                             corners: Optional[List[Tuple[float, float]]]) -> Tuple[bool, str]:
    """
    验证色块提取的前置条件。
    
    Args:
        image_array: 图像数组
        corners: 四角点坐标
    
    Returns:
        (is_valid, error_message): 验证结果和错误信息
    """
    if image_array is None:
        return False, "图像数据为空"
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        return False, f"图像格式错误，期望(H,W,3)，实际{image_array.shape}"
    
    if corners is None or len(corners) != 4:
        return False, f"需要4个角点，当前有{len(corners) if corners else 0}个"
    
    # 检查角点是否在图像范围内
    height, width = image_array.shape[:2]
    for i, (x, y) in enumerate(corners):
        if not (0 <= x < width and 0 <= y < height):
            return False, f"角点{i+1}({x:.1f}, {y:.1f})超出图像范围({width}x{height})"
    
    return True, "验证通过"

# 测试函数
if __name__ == "__main__":
    # 简单的测试代码
    print("ColorChecker色块提取器测试")
    
    # 创建测试图像
    test_image = np.random.rand(600, 800, 3).astype(np.float32)
    test_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
    
    # 验证设置
    is_valid, message = validate_extraction_setup(test_image, test_corners)
    print(f"验证结果: {is_valid}, {message}")
    
    if is_valid:
        # 提取色块
        patches = extract_colorchecker_patches(test_image, test_corners)
        print(f"提取到 {len(patches)} 个色块")
