"""胶片格式识别和自适应间距计算模块"""

from typing import Tuple, Optional
from enum import Enum
import math


class FilmFormat(Enum):
    """胶片格式枚举"""
    FORMAT_35MM = "35mm"
    FORMAT_645 = "645"
    FORMAT_66 = "6x6"
    FORMAT_67 = "6x7"
    OTHER = "other"


def detect_film_format_and_spacing(
    contact_sheet_width: int,
    contact_sheet_height: int,
    crop_width: float,
    crop_height: float,
    crop_orientation: int = 0
) -> Tuple[FilmFormat, float]:
    """检测胶片格式并计算自适应间距
    
    Args:
        contact_sheet_width: Contact sheet宽度（像素）
        contact_sheet_height: Contact sheet高度（像素）
        crop_width: 裁剪框宽度（归一化，0-1）
        crop_height: 裁剪框高度（归一化，0-1）
        crop_orientation: 裁剪方向（0, 90, 180, 270度）
        
    Returns:
        (FilmFormat, 自适应间距值) - 间距值为归一化坐标（0-1）
    """
    # 计算contact sheet的宽高比
    cs_aspect_ratio = contact_sheet_width / contact_sheet_height
    cs_is_landscape = cs_aspect_ratio > 1.0
    
    # 根据crop方向调整crop的宽高比
    if crop_orientation in [90, 270]:
        # 旋转90度或270度时，宽高互换
        effective_crop_aspect = crop_height / crop_width
    else:
        effective_crop_aspect = crop_width / crop_height
    
    crop_is_landscape = effective_crop_aspect > 1.0
    
    # 检测各种胶片格式
    format_type, spacing_factor = _detect_format_by_aspect_ratio(
        effective_crop_aspect, crop_is_landscape, cs_is_landscape
    )
    
    # 计算归一化间距
    if format_type == FilmFormat.FORMAT_35MM:
        # 35mm: 间距 = 长边长度 / 18
        longer_edge = max(crop_width, crop_height)
        spacing = longer_edge / 18.0
    elif format_type == FilmFormat.FORMAT_645:
        # 645: 间距 = 短边长度 × 5/42
        shorter_edge = min(crop_width, crop_height)
        spacing = shorter_edge * 5.0 / 42.0
    elif format_type == FilmFormat.FORMAT_66:
        # 6x6: 间距 = 边长 × 5/56
        edge_length = (crop_width + crop_height) / 2.0  # 正方形取平均
        spacing = edge_length * 5.0 / 56.0
    elif format_type == FilmFormat.FORMAT_67:
        # 6x7: 间距 = 边长 × 5/70
        edge_length = max(crop_width, crop_height)
        spacing = edge_length * 5.0 / 70.0
    else:
        # 其他格式：使用整数等分算法
        spacing = _calculate_optimal_spacing_for_other_format(
            contact_sheet_width, contact_sheet_height, crop_width, crop_height
        )
    
    # 确保间距在合理范围内
    spacing = max(0.005, min(0.1, spacing))  # 0.5% 到 10%
    return format_type, spacing


def _detect_format_by_aspect_ratio(
    crop_aspect: float, 
    crop_is_landscape: bool, 
    cs_is_landscape: bool
) -> Tuple[FilmFormat, float]:
    """根据宽高比检测胶片格式
    
    Args:
        crop_aspect: 裁剪框有效宽高比
        crop_is_landscape: 裁剪框是否为横向
        cs_is_landscape: Contact sheet是否为横向
        
    Returns:
        (格式类型, 间距因子)
    """
    # 35mm全画幅检测：2:3比例，长边与CS长边一致
    if _is_35mm_format(crop_aspect, crop_is_landscape, cs_is_landscape):
        return FilmFormat.FORMAT_35MM, 1.0
    
    # 645画幅检测：56:41.5比例，短边与CS长边一致  
    if _is_645_format(crop_aspect, crop_is_landscape, cs_is_landscape):
        return FilmFormat.FORMAT_645, 1.0
        
    # 6x6画幅检测：1:1比例
    if _is_66_format(crop_aspect):
        return FilmFormat.FORMAT_66, 1.0
        
    # 6x7画幅检测：56:70比例，长边与CS长边一致
    if _is_67_format(crop_aspect, crop_is_landscape, cs_is_landscape):
        return FilmFormat.FORMAT_67, 1.0
    return FilmFormat.OTHER, 1.0


def _is_35mm_format(crop_aspect: float, crop_is_landscape: bool, cs_is_landscape: bool) -> bool:
    """检测是否为35mm格式"""
    # 35mm标准比例：2:3 = 0.667 或 3:2 = 1.5
    target_ratios = [2.0/3.0, 3.0/2.0]
    tolerance = 0.15  # 恢复原有容差
    
    # 检查宽高比匹配
    ratio_match = any(abs(crop_aspect - ratio) <= tolerance for ratio in target_ratios)
    if not ratio_match:
        return False
        
    # 检查方向匹配：长边与contact sheet长边一致
    return crop_is_landscape == cs_is_landscape


def _is_645_format(crop_aspect: float, crop_is_landscape: bool, cs_is_landscape: bool) -> bool:
    """检测是否为645格式"""
    # 645标准比例：56:41.5 ≈ 1.35 或 41.5:56 ≈ 0.741
    target_ratios = [56.0/41.5, 41.5/56.0]
    tolerance = 0.2
    
    # 检查宽高比匹配
    ratio_match = any(abs(crop_aspect - ratio) <= tolerance for ratio in target_ratios)
    if not ratio_match:
        return False
        
    # 检查方向匹配：短边与contact sheet长边一致
    return (not crop_is_landscape) == cs_is_landscape


def _is_66_format(crop_aspect: float) -> bool:
    """检测是否为6x6格式"""
    # 6x6标准比例：1:1 = 1.0
    tolerance = 0.15
    return abs(crop_aspect - 1.0) <= tolerance


def _is_67_format(crop_aspect: float, crop_is_landscape: bool, cs_is_landscape: bool) -> bool:
    """检测是否为6x7格式"""
    # 6x7标准比例：56:70 ≈ 0.8 或 70:56 ≈ 1.25
    target_ratios = [56.0/70.0, 70.0/56.0]
    tolerance = 0.2
    
    # 检查宽高比匹配
    ratio_match = any(abs(crop_aspect - ratio) <= tolerance for ratio in target_ratios)
    if not ratio_match:
        return False
        
    # 检查方向匹配：长边与contact sheet长边一致
    return crop_is_landscape == cs_is_landscape


def _calculate_optimal_spacing_for_other_format(
    cs_width: int, 
    cs_height: int, 
    crop_width: float, 
    crop_height: float
) -> float:
    """为其他格式计算最优间距，使剩余空间刚好放下整数个照片
    
    Args:
        cs_width: Contact sheet宽度（像素）
        cs_height: Contact sheet高度（像素）  
        crop_width: 裁剪框宽度（归一化）
        crop_height: 裁剪框高度（归一化）
        
    Returns:
        归一化间距值（0-1）
    """
    # 预估一行/一列能放下的照片数量
    crops_per_row = max(1, int(1.0 / crop_width))
    crops_per_col = max(1, int(1.0 / crop_height))
    
    # 计算横向和纵向的最优间距
    if crops_per_row > 1:
        # 横向排列：(crops_per_row * crop_width + (crops_per_row-1) * spacing) = 1.0
        available_space = 1.0 - crops_per_row * crop_width
        horizontal_spacing = available_space / (crops_per_row - 1) if crops_per_row > 1 else 0.02
    else:
        horizontal_spacing = 0.02
        
    if crops_per_col > 1:
        # 纵向排列
        available_space = 1.0 - crops_per_col * crop_height  
        vertical_spacing = available_space / (crops_per_col - 1) if crops_per_col > 1 else 0.02
    else:
        vertical_spacing = 0.02
    
    # 对于超宽幅格式（如Xpan），使用更保守的间距策略
    if crop_width > 0.35 and crops_per_row <= 2:
        # 超宽crop，使用固定的小间距
        optimal_spacing = min(0.015, vertical_spacing)  # 1.5%或纵向间距，取较小值
    else:
        # 取较小值作为通用间距，确保两个方向都能整齐排列
        optimal_spacing = min(horizontal_spacing, vertical_spacing)
    
    # 限制在合理范围内
    return max(0.005, min(0.03, optimal_spacing))  # 降低最大限制到3%


def _is_aspect_ratio_close(ratio1: float, ratio2: float, tolerance: float) -> bool:
    """检查两个宽高比是否接近
    
    Args:
        ratio1: 第一个宽高比
        ratio2: 第二个宽高比  
        tolerance: 容差值
        
    Returns:
        是否在容差范围内
    """
    return abs(ratio1 - ratio2) <= tolerance


def _get_aspect_ratio_with_orientation(width: float, height: float, orientation: int) -> float:
    """根据方向获取有效宽高比
    
    Args:
        width: 宽度
        height: 高度
        orientation: 旋转角度（0, 90, 180, 270）
        
    Returns:
        有效宽高比
    """
    if orientation in [90, 270]:
        return height / width
    return width / height