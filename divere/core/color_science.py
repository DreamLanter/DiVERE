"""
色彩科学转换工具
- 提供 Lab/XYZ/DisplayP3 之间常用转换
- 整合色卡参考色从 Lab(D50) → XYZ(D50) → XYZ(D65) → Display P3 (线性/显示)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from colour import RGB_COLOURSPACES, XYZ_to_RGB
from colour.adaptation import chromatic_adaptation_VonKries
from colour.models import eotf_inverse_sRGB


# 常量白点
D50_XYZ = np.array([0.96422, 1.00000, 0.82521], dtype=np.float64)


def lab_d50_to_xyz_d50(lab: Iterable[float]) -> np.ndarray:
    """Lab(D50) → XYZ(D50) 2°观察者。返回 ndarray(3,)."""
    L, a, b = [float(x) for x in lab]
    Xn, Yn, Zn = D50_XYZ
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_inv(f: float) -> float:
        f3 = f * f * f
        return f3 if f3 > 0.008856 else (f - 16.0 / 116.0) / 7.787

    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    return np.array([X, Y, Z], dtype=np.float64)


def xyz_chromatic_adapt_bradford(xyz: np.ndarray, src_wp: np.ndarray, dst_wp: np.ndarray) -> np.ndarray:
    """Bradford 适配 XYZ: src_wp → dst_wp。支持 (3,) 或 (N,3)。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    src_wp = np.asarray(src_wp, dtype=np.float64)
    dst_wp = np.asarray(dst_wp, dtype=np.float64)
    return chromatic_adaptation_VonKries(xyz, src_wp, dst_wp, transform='Bradford')


def xy_to_XYZ_unitY(xy: np.ndarray) -> np.ndarray:
    """将 xy (…,2) 转为 XYZ (…,3)，令 Y=1。"""
    xy = np.asarray(xy, dtype=np.float64)
    x = xy[..., 0]
    y = xy[..., 1]
    Y = np.ones_like(y)
    # 避免 y 为 0
    y_safe = np.where(y == 0, 1e-10, y)
    X = x / y_safe
    Z = (1.0 - x - y) / y_safe
    return np.stack([X, Y, Z], axis=-1)


def xyz_to_display_p3_linear_rgb(xyz_d65: np.ndarray) -> np.ndarray:
    """XYZ(D65) → Display P3 线性 RGB。支持 (3,) 或 (N,3)。"""
    cs = RGB_COLOURSPACES['Display P3']
    return XYZ_to_RGB(xyz_d65, cs.whitepoint, cs.whitepoint, cs.matrix_XYZ_to_RGB)


def encode_display_p3(rgb_linear: np.ndarray) -> np.ndarray:
    """Display P3 编码到显示（采用 sRGB OETF）。返回 [0,1]。"""
    rgb_linear = np.asarray(rgb_linear, dtype=np.float64)
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    encoded = eotf_inverse_sRGB(rgb_linear)
    return np.clip(encoded, 0.0, 1.0)


def get_colorchecker_labs_ordered() -> List[Tuple[float, float, float]]:
    """按 A1..D6 顺序返回 24 个 Lab(D50)。"""
    from divere.utils.colorchecker.calibrate_scanner import get_reference_lab_data
    lab_map = get_reference_lab_data()
    order = [
        'A1','A2','A3','A4','A5','A6',
        'B1','B2','B3','B4','B5','B6',
        'C1','C2','C3','C4','C5','C6',
        'D1','D2','D3','D4','D5','D6'
    ]
    return [tuple(lab_map[k]) for k in order]


def colorchecker_display_p3_qcolors():
    """生成 24 个参考小块颜色（QColor，按 A1..D6），在 Display P3 显示空间下。"""
    from PySide6.QtGui import QColor
    labs = get_colorchecker_labs_ordered()
    # Lab(D50) → XYZ(D50)
    XYZ_D50 = np.stack([lab_d50_to_xyz_d50(lab) for lab in labs], axis=0)
    # D50 → D65
    cs_p3 = RGB_COLOURSPACES['Display P3']
    dst_wp_XYZ = xy_to_XYZ_unitY(np.array(cs_p3.whitepoint))
    XYZ_D65 = xyz_chromatic_adapt_bradford(XYZ_D50, D50_XYZ, dst_wp_XYZ)
    # XYZ(D65) → Display P3 线性 RGB
    RGB_lin = xyz_to_display_p3_linear_rgb(XYZ_D65)
    RGB_lin = np.clip(RGB_lin, 0.0, 1.0)
    # OETF（sRGB 曲线）
    RGB_disp = encode_display_p3(RGB_lin)
    qcolors = [QColor(int(r*255+0.5), int(g*255+0.5), int(b*255+0.5)) for r,g,b in RGB_disp]
    return qcolors


