"""
默认预设加载器
集中式默认值入口：优先从 config/defaults/default.json 读取；若不存在，回退到内置 Preset。
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

    # 1) 尝试加载项目内默认文件（使用与 enhanced_config_manager 相同的路径解析方法）
    try:
        # 使用与其他配置文件相同的加载机制，支持用户配置和多路径回退
        from divere.utils.enhanced_config_manager import enhanced_config_manager
        
        # 候选路径列表，优先级从高到低：
        # 1. 用户配置目录中的 defaults
        # 2. enhanced_config_manager 的 app_config_dir
        # 3. 原始 get_data_dir 方法
        candidate_paths = []
        
        # 用户配置目录中的 defaults（最高优先级）
        user_defaults_dir = enhanced_config_manager.user_config_dir / "config" / "defaults"
        user_default_path = user_defaults_dir / "default.json"
        candidate_paths.append(user_default_path)
        
        # enhanced_config_manager 的 app_config_dir
        app_defaults_dir = enhanced_config_manager.app_config_dir / "defaults"
        app_default_path = app_defaults_dir / "default.json"
        candidate_paths.append(app_default_path)
        
        # 原始方法作为回退
        try:
            from divere.utils.app_paths import get_data_dir
            config_dir = get_data_dir("config")
            fallback_path = config_dir / "defaults" / "default.json"
            candidate_paths.append(fallback_path)
        except Exception:
            pass
        
        # 尝试每个候选路径
        for default_path in candidate_paths:
            if default_path.exists():
                with open(default_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _DEFAULT_PRESET_CACHE = Preset.from_dict(data)
                return _DEFAULT_PRESET_CACHE
        
        # 如果没有找到任何文件，记录日志
        if not _DEFAULT_PRESET_LOGGED:
            print(f"默认配置文件不存在，尝试的路径: {[str(p) for p in candidate_paths]}")
            _DEFAULT_PRESET_LOGGED = True
    except Exception as e:
        # 解析失败或路径失败，记录错误并走回退
        if not _DEFAULT_PRESET_LOGGED:
            print(f"加载默认配置文件失败: {e}")
            _DEFAULT_PRESET_LOGGED = True
        pass

    # 2) 回退到内置默认（集中唯一硬编码），避免散落默认值
    # v3: 填充 raw_file、IDT 的 white/primitives，占位 gamma=1.0
    p = Preset(
        name="default",
        raw_file="default.tif",
        input_transformation=InputTransformationDefinition(
            name="KodakEnduraPremier",
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


def load_film_type_default_preset(film_type: str) -> Optional[Preset]:
    """
    根据胶片类型加载默认预设
    现在使用 FilmTypeController 提供的统一默认值逻辑
    
    Args:
        film_type: 胶片类型 (color_negative_c41, color_reversal, etc.)
        
    Returns:
        对应胶片类型的默认预设
    """
    try:
        from divere.core.film_type_controller import FilmTypeController
        
        # 使用 FilmTypeController 获取胶片类型的默认参数
        controller = FilmTypeController()
        default_params = controller.get_default_params(film_type)
        
        # 尝试从通用默认文件获取基础预设结构
        base_preset = load_default_preset()
        if base_preset:
            # 创建副本并应用胶片类型的默认参数
            preset_dict = base_preset.to_dict()
            
            # 更新胶片类型
            if 'metadata' not in preset_dict:
                preset_dict['metadata'] = {}
            preset_dict['metadata']['film_type'] = film_type
            
            # 更新 cc_params 为胶片类型的默认值
            if 'cc_params' not in preset_dict:
                preset_dict['cc_params'] = {}
            
            cc_params = preset_dict['cc_params']
            cc_params.update({
                'density_gamma': default_params.get('density_gamma', 1.0),
                'density_dmax': default_params.get('density_dmax', 2.5),
                'rgb_gains': list(default_params.get('rgb_gains', (0.0, 0.0, 0.0))),
                'density_matrix': {
                    'name': default_params.get('density_matrix_name', 'Identity'),
                    'values': None
                },
                'density_curve': {
                    'name': default_params.get('density_curve_name', 'linear'),
                    'points': {'rgb' if film_type not in ['b&w_negative', 'b&w_reversal'] else 'gray': [[0.0, 0.0], [1.0, 1.0]]}
                }
            })
            
            preset = Preset.from_dict(preset_dict)
            preset.film_type = film_type
            return preset
        else:
            # 如果没有基础预设，创建最小化预设
            preset = Preset(
                name=f"{film_type} 默认预设",
                version=3,
                film_type=film_type,
                grading_params={
                    'density_gamma': default_params.get('density_gamma', 1.0),
                    'density_dmax': default_params.get('density_dmax', 2.5),
                    'rgb_gains': list(default_params.get('rgb_gains', (0.0, 0.0, 0.0))),
                }
            )
            
            # 设置默认的 density_matrix 和 density_curve
            preset.density_matrix = MatrixDefinition(
                name=default_params.get('density_matrix_name', 'Identity'),
                values=None
            )
            preset.density_curve = CurveDefinition(
                name=default_params.get('density_curve_name', 'linear'),
                points=[[0.0, 0.0], [1.0, 1.0]]
            )
            
            return preset
            
    except Exception as e:
        print(f"加载胶片类型 '{film_type}' 的默认预设失败: {e}")
        return None


def get_available_film_types() -> list[str]:
    """
    获取所有可用的胶片类型（基于配置文件）。
    
    Returns:
        可用胶片类型列表
    """
    film_types = [
        "color_negative_c41",
        "color_negative_ecn2", 
        "color_reversal",
        "b&w_negative",
        "b&w_reversal",
        "digital"
    ]
    
    return film_types


