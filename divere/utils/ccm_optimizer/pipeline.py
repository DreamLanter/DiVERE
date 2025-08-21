#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiVERE管线模拟器

严格按照DiVERE的处理流程模拟色彩处理管线，用于CCM优化。

处理流程:
1. 注册自定义输入色彩变换（基于优化参数的基色）
2. 输入色彩变换 → 工作色彩空间(ACEScg)
3. 密度反转 (RGB → 密度)
4. RGB增益调整
5. 密度曲线处理 (跳过，因为优化中不包含曲线)
6. 返回线性ACEScg RGB

注意: 不包含曲线处理和输出色彩转换
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import sys

# 添加项目根目录到路径，以便导入divere模块
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from divere.core.color_space import ColorSpaceManager
    from divere.core.data_types import ImageData, ColorGradingParams
    import colour
except ImportError as e:
    print(f"Warning: 无法导入divere模块: {e}")
    print("请确保在DiVERE项目根目录下运行")

class DiVEREPipelineSimulator:
    """DiVERE色彩处理管线模拟器"""
    
    def __init__(self, verbose=False):
        """初始化管线模拟器"""
        self.verbose = verbose  # 控制详细输出

    @staticmethod
    def primaries_to_xyz_matrix(primaries, white_point):
        """从primaries和白点计算到XYZ的转换矩阵"""
        # 确保primaries是正确形状的数组
        primaries = np.asarray(primaries)
        if primaries.ndim == 1:
            # 如果是一维数组，重塑为(3,2)
            primaries = primaries.reshape(3, 2)
        
        # 将xy坐标转换为XYZ
        xyz_primaries = np.zeros((3, 3))
        for i in range(3):
            x, y = primaries[i]
            # 防止除零错误
            if abs(y) < 1e-10:
                y = 1e-10
            z = 1 - x - y
            xyz_primaries[:, i] = [x/y, 1.0, z/y]
        
        # 白点XYZ
        white_point = np.asarray(white_point)
        wx, wy = white_point
        # 防止除零错误
        if abs(wy) < 1e-10:
            wy = 1e-10
        wz = 1 - wx - wy
        white_xyz = np.array([wx/wy, 1.0, wz/wy])
        
        # 计算scaling factors
        scaling = np.linalg.solve(xyz_primaries, white_xyz)
        
        # 构建最终的转换矩阵
        return xyz_primaries * scaling[np.newaxis, :]
    
    def simulate_full_pipeline(self, input_rgb_patches: Dict[str, Tuple[float, float, float]],
                              primaries_xy: np.ndarray,
                              white_point_xy: Optional[np.ndarray] = None,
                              gamma: float = 2.6,
                              dmax: float = 2.0,
                              r_gain: float = 0.0,
                              b_gain: float = 0.0,
                              correction_matrix: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float, float]]:
        """
        一体化DiVERE管线模拟器 - 所有操作在一个函数内完成。
        
        完整流程:
        1. 注册输入色彩变换，转换到工作空间(ACEScg)
        2. original_density = -log10(safe_rgb)  
        3. adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        4. 应用密度校正矩阵 (如果有)
        5. 添加 R/B 增益
        6. rgb = 10^adjusted_density
        
        Args:
            input_rgb_patches: 输入RGB色块字典
            primaries_xy: 自定义色彩空间的RGB基色xy坐标 (3, 2)
            white_point_xy: 白点xy坐标，默认D65
            gamma: 密度反差参数
            dmax: 最大密度参数  
            r_gain: R通道增益
            b_gain: B通道增益
            correction_matrix: 密度校正矩阵 (3x3)
        
        Returns:
            处理后的RGB色块字典
        """
        # ===== 步骤1: 色彩空间注册和转换 =====
        
        # 设置默认白点
        if white_point_xy is None:
            white_point_xy = np.array([0.3127, 0.3290])  # D65
        
        # 直接从primaries和white_point转换到ACEScg
        # ACEScg的基色和白点定义
        acescg_primaries = np.array([
            [0.713, 0.293],  # Red
            [0.165, 0.830],  # Green  
            [0.128, 0.044]   # Blue
        ])
        acescg_white_point = np.array([0.32168, 0.33767])  # ACEScg白点
        
        # 计算转换矩阵
        input_to_xyz = self.primaries_to_xyz_matrix(primaries_xy, white_point_xy)
        acescg_to_xyz = self.primaries_to_xyz_matrix(acescg_primaries, acescg_white_point)
        xyz_to_acescg = np.linalg.inv(acescg_to_xyz)
        
        # 组合转换矩阵：输入空间 -> XYZ -> ACEScg
        input_to_acescg = xyz_to_acescg @ input_to_xyz

        # 与主管线一致：白点适应增益（简化版，匹配 ColorSpaceManager 的实现）
        def _xy_to_XYZ_normalized(xy):
            x, y = float(xy[0]), float(xy[1])
            if abs(y) < 1e-10:
                y = 1e-10
            X = x / y
            Y = 1.0
            Z = (1.0 - x - y) / y
            return np.array([X, Y, Z], dtype=float)

        src_white = np.array(white_point_xy if white_point_xy is not None else [0.3127, 0.3290], dtype=float)
        dst_white = acescg_white_point
        src_white_XYZ = _xy_to_XYZ_normalized(src_white)
        dst_white_XYZ = _xy_to_XYZ_normalized(dst_white)
        # 简化增益：分量比值并裁剪
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_vector = np.divide(dst_white_XYZ, src_white_XYZ)
        gain_vector = np.clip(gain_vector, 0.1, 10.0)
        
        # 转换到工作空间 (ACEScg) 并应用白点增益（与主管线一致在矩阵转换阶段进行）
        working_space_patches = {}
        for patch_id, (r, g, b) in input_rgb_patches.items():
            input_rgb = np.array([r, g, b])
            acescg_rgb = input_to_acescg @ input_rgb
            # 应用白点适应增益（逐分量）
            acescg_rgb = acescg_rgb * gain_vector
            working_space_patches[patch_id] = tuple(acescg_rgb.tolist())
        
        # ===== 步骤2-6: 核心密度处理 =====
        final_rgb_patches = {}
        pivot = 0.9  # DiVERE固定基准点
        
        for patch_id, (r, g, b) in working_space_patches.items():
            # 步骤2: 计算原始密度 (使用软约束避免log(0)和负数)
            # 防止RGB值为负数，确保log10操作的安全性
            r_safe = max(r, 1e-10)  # 确保R值 >= 1e-10
            g_safe = max(g, 1e-10)  # 确保G值 >= 1e-10
            b_safe = max(b, 1e-10)  # 确保B值 >= 1e-10
            
            original_density = np.array([
                -np.log10(r_safe),
                -np.log10(g_safe),
                -np.log10(b_safe)
            ])
            
            # 步骤3: DiVERE密度调整公式
            adjusted_density = pivot + (original_density - pivot) * gamma - dmax
            
            # 步骤4: 应用密度校正矩阵
            if correction_matrix is not None:
                adjusted_density = correction_matrix @ adjusted_density
                    
            # 步骤5: 添加RGB增益 (密度空间中的加法)
            adjusted_density[0] += r_gain  # R通道
            adjusted_density[2] += b_gain  # B通道
            # G通道增益固定为0 (adjusted_density[1] += 0.0)
            
            # 步骤6: 转换回RGB
            final_rgb = 10 ** adjusted_density
            
            # 注意：这里不应该除以65535，因为：
            # 1. 输入已经是0-1范围的RGB值
            # 2. 输出也应该保持在0-1范围
            # 3. 65535是16位整数的最大值，不适用于浮点RGB
            
            final_rgb_patches[patch_id] = tuple(final_rgb.tolist())
        
        if self.verbose:
            print(f"✓ 一体化管线处理完成，处理了 {len(final_rgb_patches)} 个色块")
        return final_rgb_patches

if __name__ == "__main__":
    # 简单的测试代码
    print("DiVERE管线模拟器已加载") 
