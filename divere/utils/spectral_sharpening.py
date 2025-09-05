#!/usr/bin/env python3
"""
光谱锐化（Spectral Sharpening）接口封装

职责：
- 从图像与色卡四角点提取 24 色块（线性RGB）
- 调用现有 CCM 优化器（不改动其算法）
- 返回优化得到的 primaries 与密度/增益参数

约束：不修改核心算法，仅做数据准备与结果封装。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 依赖现有实现（算法不做修改）
from .ccm_optimizer.optimizer import CCMOptimizer
from .ccm_optimizer.extractor import (
    validate_extraction_setup,
    extract_colorchecker_patches,
)


def _to_linear_image_array(image_array: np.ndarray, gamma: float) -> np.ndarray:
    """
    将输入非线性 RGB 数组转换到线性域。[0,1] 浮点。
    仅做 gamma 逆变换，不做矩阵或白点变换，以保持“输入空间”的含义。
    """
    if image_array.dtype != np.float32 and image_array.dtype != np.float64:
        arr = image_array.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        arr = image_array.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    # 逆伽马：I_linear = I_nonlinear^gamma
    gamma = float(gamma) if gamma and gamma > 0 else 1.0
    return np.power(arr, gamma, dtype=np.float32)


def run(
    image_array: np.ndarray,
    cc_corners: List[Tuple[float, float]],
    input_space_gamma: float,
    use_correction_matrix: bool,
    correction_matrix: Optional[np.ndarray],
    optimizer_max_iter: int = 300,
    optimizer_tolerance: float = 1e-8,
    reference_file: str = "colorchecker_acescg_rgb_values.json",
    sharpening_config: Optional[Any] = None,  # SpectralSharpeningConfig
    ui_params: Optional[Dict] = None,  # 来自UI的当前参数
    status_callback: Optional[callable] = None,  # 状态回调函数
) -> Dict[str, Any]:
    """
    执行光谱锐化优化，返回优化结果（不改动核心算法）。

    Args:
        image_array: 原始图像数组（UI层提供）。
        cc_corners: 色卡四角点（左上、右上、右下、左下）。
        input_space_gamma: 当前输入色彩空间的 gamma，用于线性化。
        use_correction_matrix: 是否启用密度校正矩阵参与优化。
        correction_matrix: 3x3 密度校正矩阵（可为 None）。
        optimizer_max_iter: CMA-ES 最大迭代数。
        optimizer_tolerance: 收敛容差。
        reference_file: 参考色卡 RGB 文件名。
        sharpening_config: 光谱锐化配置对象，控制优化参数。
        ui_params: 来自UI的当前参数，用作优化初值。

    Returns:
        dict: {
            'success': bool,
            'rmse': float,
            'parameters': {
                'primaries_xy': ndarray(3,2), 'gamma': float, 'dmax': float,
                'r_gain': float, 'b_gain': float
            },
            'evaluation': dict,  # 评估报告
        }
    """

    is_valid, msg = validate_extraction_setup(image_array, cc_corners)
    if not is_valid:
        raise ValueError(f"色卡提取前置条件不满足: {msg}")

    # 与主管线一致：这里的 input_space_gamma 作为“前置 IDT Gamma”
    linear_img = _to_linear_image_array(image_array, input_space_gamma)

    input_patches = extract_colorchecker_patches(linear_img, cc_corners)
    if not input_patches:
        raise RuntimeError("无法提取色卡数据")

    # 使用配置对象（如果提供）
    if sharpening_config is not None:
        optimizer = CCMOptimizer(
            reference_file=sharpening_config.reference_file,
            sharpening_config=sharpening_config,
            status_callback=status_callback
        )
        max_iter = sharpening_config.max_iter
        tolerance = sharpening_config.tolerance
    else:
        # 向后兼容：使用传统参数
        optimizer = CCMOptimizer(reference_file=reference_file, status_callback=status_callback)
        max_iter = optimizer_max_iter
        tolerance = optimizer_tolerance

    cm = correction_matrix if use_correction_matrix else None
    result = optimizer.optimize(
        input_patches,
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        correction_matrix=cm,
        ui_params=ui_params,
        status_callback=status_callback,
    )

    # 附加评估
    try:
        evaluation = optimizer.evaluate_parameters(result.get('parameters', {}), input_patches, correction_matrix=cm)
        result['evaluation'] = evaluation
    except Exception:
        pass

    return result


