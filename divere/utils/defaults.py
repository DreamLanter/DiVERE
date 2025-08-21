"""
默认预设加载器
集中式默认值入口：优先从 config/default.json 读取；若不存在，回退到内置 Preset。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from divere.core.data_types import Preset, ColorGradingParams, InputTransformationDefinition, MatrixDefinition, CurveDefinition


_DEFAULT_PRESET_CACHE: Optional[Preset] = None
_DEFAULT_PRESET_LOGGED: bool = False


def load_default_preset() -> Preset:
    global _DEFAULT_PRESET_CACHE, _DEFAULT_PRESET_LOGGED
    if _DEFAULT_PRESET_CACHE is not None:
        return _DEFAULT_PRESET_CACHE

    # 1) 尝试加载项目内默认文件（统一路径解析：可兼容源码与打包环境）
    try:
        from divere.utils.app_paths import resolve_data_path
        default_path = resolve_data_path("config", "default.json")
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _DEFAULT_PRESET_CACHE = Preset.from_dict(data)
        return _DEFAULT_PRESET_CACHE
    except Exception:
        # 解析失败或路径失败，走回退
        pass

    # 2) 回退到内置默认（集中唯一硬编码），避免散落默认值
    # v3: 填充 raw_file、IDT 的 white/primitives，占位 gamma=1.0
    p = Preset(
        name="default",
        raw_file="default.tif",
        input_transformation=InputTransformationDefinition(
            name="CCFLGeneric_Linear",
            definition={
                "gamma": 1.0,
                "white": {"x": 0.3127, "y": 0.3290},
                "primitives": {
                    "r": {"x": 0.6400, "y": 0.3300},
                    "g": {"x": 0.2100, "y": 0.7100},
                    "b": {"x": 0.1500, "y": 0.0600},
                },
            },
        ),
        grading_params=ColorGradingParams(
            density_gamma=1.0,
            density_dmax=2.5,
            rgb_gains=(0.65, 0.0, 0.0),
            density_matrix_name="Cineon_States_M_to_Print_Density",
            density_curve_name="Kodak Endura Premier",
            curve_points=[(0.0, 0.0), (1.0, 1.0)],
        ).to_dict(),
        density_matrix=MatrixDefinition(name="Cineon_States_M_to_Print_Density", values=None),
        density_curve=CurveDefinition(name="Kodak Endura Premier", points=[(0.0, 0.0), (1.0, 1.0)]),
        orientation=0,
    )
    _DEFAULT_PRESET_CACHE = p
    return p


