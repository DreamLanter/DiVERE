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
    # 默认输入空间 CCFLGeneric_Linear；无矩阵；Endura纸曲线（名称示例，可为空）
    p = Preset(
        name="default",
        input_transformation=InputTransformationDefinition(name="CCFLGeneric_Linear", definition={}),
        grading_params=ColorGradingParams().to_dict(),
        density_matrix=MatrixDefinition(name="", values=None),
        density_curve=CurveDefinition(name="Kodak_Endura_Paper", points=[]),
        orientation=0,
    )
    _DEFAULT_PRESET_CACHE = p
    return p


