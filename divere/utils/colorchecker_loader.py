#!/usr/bin/env python3
"""
ColorChecker参考色加载器

处理新的ColorChecker JSON schema，包含type判断和色彩变换逻辑：
- type: "XYZ" -> 执行Bradford CAT变换到working colorspace
- type: "DensityExp" -> 检查working colorspace是否与required_working_colorspace字段匹配

职责：
- 加载和验证ColorChecker JSON文件
- 根据type字段选择处理策略
- 执行必要的色彩空间变换
- 提供统一的参考色数据接口
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from divere.core.color_science import bradford_chromatic_adaptation
from divere.utils.app_paths import resolve_data_path


class ColorCheckerLoadError(Exception):
    """ColorChecker加载错误"""
    pass


class WorkspaceValidationError(Exception):
    """工作空间验证错误"""
    pass


def load_colorchecker_reference(
    filename: str,
    working_colorspace: str,
    color_space_manager: Optional[Any] = None
) -> Dict[str, List[float]]:
    """
    加载ColorChecker参考色数据，自动处理白点变换
    
    Args:
        filename: ColorChecker JSON文件名
        working_colorspace: 当前工作色彩空间名称
        color_space_manager: ColorSpaceManager实例，用于获取工作空间白点
        
    Returns:
        Dict[patch_id, [r, g, b]] - 24个色块的RGB值
        
    Raises:
        ColorCheckerLoadError: 文件加载或格式错误
        WorkspaceValidationError: 工作空间验证失败
    """
    # 1. 加载JSON文件
    data = _load_colorchecker_json(filename)
    
    # 2. 验证schema
    _validate_colorchecker_schema(data, filename)
    
    # 3. 获取目标白点（工作空间的白点）
    target_illuminant = _get_working_space_white_point(working_colorspace, color_space_manager)
    
    # 4. 根据type字段选择处理策略
    colorchecker_type = data.get("type")
    
    if colorchecker_type == "XYZ":
        return _process_xyz_type(data, target_illuminant, working_colorspace, color_space_manager)
    elif colorchecker_type == "DensityExp":
        return _process_density_exp_type(data, working_colorspace)
    else:
        raise ColorCheckerLoadError(f"不支持的ColorChecker类型: {colorchecker_type}")


def _get_working_space_white_point(working_colorspace: str, color_space_manager: Optional[Any]) -> str:
    """获取工作空间的白点"""
    if color_space_manager is not None:
        try:
            return color_space_manager.get_colorspace_white_point(working_colorspace)
        except Exception:
            pass
    
    # 硬编码的常见工作空间白点（fallback）
    workspace_white_points = {
        "ACEScg": "D60",
        "Rec2020": "D65", 
        "sRGB": "D65",
        "ProPhoto": "D50",
        "KodakEnduraPremier": "D50"
    }
    
    return workspace_white_points.get(working_colorspace, "D65")


def _load_colorchecker_json(filename: str) -> Dict[str, Any]:
    """加载ColorChecker JSON文件"""
    try:
        # 使用修正后的统一数据路径
        file_path = resolve_data_path("config", "colorchecker", filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ColorCheckerLoadError(f"无法加载ColorChecker文件: {filename}")
    except (json.JSONDecodeError, IOError) as e:
        raise ColorCheckerLoadError(f"ColorChecker文件格式错误: {filename} - {e}")


def _validate_colorchecker_schema(data: Dict[str, Any], filename: str) -> None:
    """验证ColorChecker JSON schema"""
    # 基础必需字段：所有类型都需要
    required_fields = ["type", "data"]
    
    for field in required_fields:
        if field not in data:
            raise ColorCheckerLoadError(f"ColorChecker文件 {filename} 缺少必需字段: {field}")
    
    # 验证type字段
    valid_types = ["XYZ", "DensityExp"]
    colorchecker_type = data["type"]
    if colorchecker_type not in valid_types:
        raise ColorCheckerLoadError(f"无效的type字段: {colorchecker_type}, 支持: {valid_types}")
    
    # 验证white_point字段（仅XYZ类型需要）
    if colorchecker_type == "XYZ":
        if "white_point" not in data:
            raise ColorCheckerLoadError(f"XYZ类型的ColorChecker文件 {filename} 缺少必需字段: white_point")
        
        wb = data["white_point"]
        if not isinstance(wb, str):
            raise ColorCheckerLoadError(f"white_point字段必须是字符串，当前类型: {type(wb)}")
        
        valid_illuminants = ["D50", "D55", "D60", "D65"]
        if wb not in valid_illuminants:
            raise ColorCheckerLoadError(f"无效的white_point值: {wb}, 支持: {valid_illuminants}")
    
    # 验证required_working_colorspace字段（仅DensityExp类型需要）
    if colorchecker_type == "DensityExp":
        if "required_working_colorspace" not in data:
            raise ColorCheckerLoadError(f"DensityExp类型的ColorChecker文件 {filename} 缺少必需字段: required_working_colorspace")
        
        required_ws = data["required_working_colorspace"]
        if not isinstance(required_ws, str):
            raise ColorCheckerLoadError(f"required_working_colorspace字段必须是字符串，当前类型: {type(required_ws)}")
        
        if not required_ws.strip():
            raise ColorCheckerLoadError("required_working_colorspace字段不能为空")
    
    # 验证data字段
    if not isinstance(data["data"], dict):
        raise ColorCheckerLoadError("data字段必须是字典类型")


def _process_xyz_type(
    data: Dict[str, Any], 
    target_illuminant: str, 
    working_colorspace: str,
    color_space_manager: Optional[Any] = None
) -> Dict[str, List[float]]:
    """
    处理XYZ类型的ColorChecker数据
    从源白点执行Bradford CAT变换到目标白点，然后转换为工作色彩空间RGB
    """
    # 从JSON文件中读取源白点
    src_illuminant = data["white_point"]
    
    raw_data = data["data"]
    result = {}
    
    # 执行CAT变换和XYZ到RGB转换
    for patch_id, xyz_values in raw_data.items():
        try:
            # 将XYZ值转换为numpy数组
            xyz_array = np.array(xyz_values, dtype=np.float64)
            
            # 执行色度适应变换：从源白点到目标白点
            if src_illuminant != target_illuminant:
                adapted_xyz = bradford_chromatic_adaptation(
                    xyz_array, src_illuminant, target_illuminant
                )
            else:
                adapted_xyz = xyz_array
            
            # 将适应后的XYZ转换为工作色彩空间RGB
            working_rgb = _convert_xyz_to_working_space_rgb(
                adapted_xyz, working_colorspace, color_space_manager
            )
            
            result[patch_id] = working_rgb.tolist()
            
        except Exception as e:
            raise ColorCheckerLoadError(f"色彩变换失败 (patch {patch_id}): {e}")
    
    return result


def _process_density_exp_type(
    data: Dict[str, Any], 
    working_colorspace: str
) -> Dict[str, List[float]]:
    """
    处理DensityExp类型的ColorChecker数据
    检查working colorspace是否与required_working_colorspace字段匹配
    """
    # 获取要求的工作色彩空间
    required_workspace = data.get("required_working_colorspace", "").strip()
    
    # 检查工作空间（标准化比较）
    normalized_workspace = working_colorspace.strip() if working_colorspace else ""
    if normalized_workspace != required_workspace:
        raise WorkspaceValidationError(
            f"DensityExp类型要求工作色彩空间为{required_workspace}，当前为: {working_colorspace}"
        )
    
    # 工作空间正确，直接返回数据
    raw_data = data["data"]
    return {k: list(v) for k, v in raw_data.items()}


def get_supported_illuminants(filename: str) -> List[str]:
    """获取ColorChecker文件支持的光源列表"""
    try:
        data = _load_colorchecker_json(filename)
        if "white_point" in data:
            return list(data["white_point"].keys())
        return ["D50", "D55", "D60", "D65"]  # 默认支持的光源
    except Exception:
        return ["D50", "D55", "D60", "D65"]


def get_colorchecker_type(filename: str) -> Optional[str]:
    """获取ColorChecker文件的类型"""
    try:
        data = _load_colorchecker_json(filename)
        return data.get("type")
    except Exception:
        return None


def validate_colorchecker_workspace_compatibility(
    filename: str, 
    working_colorspace: str
) -> Tuple[bool, Optional[str]]:
    """
    验证ColorChecker文件与工作空间的兼容性
    
    Returns:
        (is_compatible, error_message)
    """
    try:
        colorchecker_type = get_colorchecker_type(filename)
        
        if colorchecker_type == "DensityExp":
            # 加载JSON文件获取required_working_colorspace
            data = _load_colorchecker_json(filename)
            required_workspace = data.get("required_working_colorspace", "").strip()
            
            # 标准化工作空间名称进行比较（去除空白字符）
            normalized_workspace = working_colorspace.strip() if working_colorspace else ""
            
            if normalized_workspace != required_workspace:
                return False, f"需要将工作色彩空间设为{required_workspace}"
        
        return True, None
        
    except Exception as e:
        return False, f"验证失败: {e}"


def _convert_xyz_to_working_space_rgb(
    xyz: np.ndarray, 
    working_colorspace: str, 
    color_space_manager: Optional[Any] = None
) -> np.ndarray:
    """
    将XYZ值转换为工作色彩空间RGB
    
    Args:
        xyz: XYZ值 (shape可以是(3,)或(N,3))
        working_colorspace: 工作色彩空间名称
        color_space_manager: ColorSpaceManager实例
        
    Returns:
        RGB值，shape与输入相同
    """
    # 使用ColorSpaceManager的XYZ到RGB转换方法
    if color_space_manager is not None:
        try:
            return color_space_manager.convert_xyz_to_working_space_rgb(xyz, working_colorspace)
        except Exception as e:
            raise ColorCheckerLoadError(f"XYZ到RGB转换失败: {e}")
    
    # 如果没有ColorSpaceManager，抛出错误
    raise ColorCheckerLoadError("需要ColorSpaceManager实例进行XYZ到RGB转换")


