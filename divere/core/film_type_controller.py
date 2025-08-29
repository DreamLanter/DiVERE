"""
胶片类型控制器
根据胶片类型动态配置pipeline和UI行为
"""

from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

from .data_types import PipelineConfig, UIStateConfig, ColorGradingParams


class FilmTypeController:
    """胶片类型控制器：管理不同胶片类型的处理逻辑和UI状态"""
    
    def __init__(self):
        self._config_data = self._load_film_types_config()
        self._pipeline_configs = {}
        self._ui_state_configs = {}
        self._load_configs_from_data()
    
    def get_pipeline_config(self, film_type: str) -> PipelineConfig:
        """获取指定胶片类型的pipeline配置"""
        return self._pipeline_configs.get(film_type, self._pipeline_configs["color_negative_c41"]).copy()
    
    def get_ui_state_config(self, film_type: str) -> UIStateConfig:
        """获取指定胶片类型的UI状态配置"""
        return self._ui_state_configs.get(film_type, self._ui_state_configs["color_negative_c41"]).copy()
    
    def get_default_params(self, film_type: str) -> Dict[str, Any]:
        """
        获取指定胶片类型的默认参数
        实现基础默认值 + 设备/用户覆盖的逻辑
        """
        pipeline_config = self.get_pipeline_config(film_type)
        
        # 从原始配置数据中提取default_rgb_gains
        film_types_data = self._config_data.get("film_types", {})
        film_config = film_types_data.get(film_type, {})
        pipeline_data = film_config.get("pipeline", {})
        default_rgb_gains = pipeline_data.get("default_rgb_gains", [0.0, 0.0, 0.0])
        
        # 1. 从 film_types_config.json 获取基础默认参数
        base_params = {
            "density_gamma": pipeline_config.default_density_gamma,
            "density_dmax": pipeline_config.default_density_dmax,
            "density_matrix_name": pipeline_config.default_density_matrix_name,
            "density_curve_name": pipeline_config.default_density_curve_name,
            "enable_density_inversion": pipeline_config.enable_density_inversion,
            "enable_density_matrix": pipeline_config.enable_density_matrix,
            "enable_rgb_gains": pipeline_config.enable_rgb_gains,
            "enable_density_curve": pipeline_config.enable_density_curve,
            "default_rgb_gains": default_rgb_gains,  # 从JSON中提取
        }
        
        # 2. 尝试加载用户/设备覆盖配置
        override_params = self._load_default_override_params(film_type)
        
        # 3. 合并参数：基础 + 覆盖
        final_params = base_params.copy()
        if override_params:
            final_params.update(override_params)
        
        return final_params
    
    def _load_default_override_params(self, film_type: str) -> Optional[Dict[str, Any]]:
        """
        加载默认覆盖参数
        优先级：default_imacon.json > default.json
        """
        try:
            # 获取配置文件路径
            current_dir = Path(__file__).parent
            defaults_dir = current_dir.parent / "config" / "defaults"
            
            # 候选覆盖文件，按优先级排序
            candidate_files = [
                "default.json"          # 通用覆盖
            ]
            
            for filename in candidate_files:
                config_file = defaults_dir / filename
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 检查是否匹配当前胶片类型或为通用配置
                        metadata = data.get("metadata", {})
                        config_film_type = metadata.get("film_type")
                        
                        # 如果配置文件指定了胶片类型，必须匹配才能使用
                        if config_film_type and config_film_type != film_type:
                            continue
                            
                        # 提取 cc_params 作为覆盖参数
                        cc_params = data.get("cc_params", {})
                        if cc_params:
                            override_params = {}
                            
                            # 提取相关参数
                            if "density_gamma" in cc_params:
                                override_params["density_gamma"] = cc_params["density_gamma"]
                            if "density_dmax" in cc_params:
                                override_params["density_dmax"] = cc_params["density_dmax"]
                            if "rgb_gains" in cc_params:
                                gains = cc_params["rgb_gains"]
                                if isinstance(gains, list) and len(gains) >= 3:
                                    override_params["rgb_gains"] = tuple(gains[:3])
                            
                            # 处理 density_matrix
                            if "density_matrix" in cc_params:
                                matrix_data = cc_params["density_matrix"]
                                if isinstance(matrix_data, dict) and "name" in matrix_data:
                                    override_params["density_matrix_name"] = matrix_data["name"]
                            
                            # 处理 density_curve
                            if "density_curve" in cc_params:
                                curve_data = cc_params["density_curve"]
                                if isinstance(curve_data, dict) and "name" in curve_data:
                                    override_params["density_curve_name"] = curve_data["name"]
                            
                            print(f"从 {filename} 加载覆盖参数用于胶片类型 {film_type}: {override_params}")
                            return override_params
                            
                    except Exception as e:
                        print(f"解析 {filename} 失败: {e}")
                        continue
            
            return None
            
        except Exception as e:
            print(f"加载默认覆盖配置失败: {e}")
            return None
    
    def should_convert_to_monochrome(self, film_type: str) -> bool:
        """判断是否需要在IDT阶段转换为monochrome"""
        pipeline_config = self.get_pipeline_config(film_type)
        return pipeline_config.convert_to_monochrome_in_idt
    
    def apply_film_type_defaults(self, params: ColorGradingParams, film_type: str, 
                                force_apply: bool = False, only_if_no_preset: bool = False) -> ColorGradingParams:
        """
        对现有参数应用胶片类型的默认值和约束
        
        Args:
            params: 要修改的参数
            film_type: 胶片类型
            force_apply: 是否强制应用所有默认值（忽略当前值）
            only_if_no_preset: 是否只在没有预设的情况下应用（例如新照片加载时）
        """
        pipeline_config = self.get_pipeline_config(film_type)
        default_values = self.get_default_params(film_type)
        
        # 总是应用pipeline启用/禁用设置
        params.enable_density_inversion = pipeline_config.enable_density_inversion
        params.enable_density_matrix = pipeline_config.enable_density_matrix
        params.enable_rgb_gains = pipeline_config.enable_rgb_gains
        params.enable_density_curve = pipeline_config.enable_density_curve
        
        # 应用默认值的逻辑：
        # 1. force_apply=True: 总是应用默认值
        # 2. 否则，仅当当前值是通用默认值时才覆盖
        def should_apply_default(current_value, universal_default):
            return force_apply or current_value == universal_default
        
        # 应用IDT相关默认值（从JSON配置）
        if should_apply_default(params.density_gamma, 1.0):
            params.density_gamma = default_values["density_gamma"]
        if should_apply_default(params.density_dmax, 2.5):
            params.density_dmax = default_values["density_dmax"]
        
        # 应用默认matrix和curve名称
        if should_apply_default(params.density_matrix_name, "Identity"):
            params.density_matrix_name = default_values["density_matrix_name"]
        if should_apply_default(params.density_curve_name, "linear"):
            params.density_curve_name = default_values["density_curve_name"]
        
        # 特殊处理：黑白胶片设置
        if film_type in ["b&w_negative", "b&w_reversal"]:
            # 应用黑白胶片的默认RGB gains
            if should_apply_default(params.rgb_gains, (0.5, 0.0, 0.0)):
                params.rgb_gains = tuple(default_values.get("default_rgb_gains", [0.0, 0.0, 0.0]))
        
        return params
    
    def apply_black_and_white_conversion(self, params: ColorGradingParams, film_type: str) -> ColorGradingParams:
        """
        Apply only the necessary changes for B&W conversion, preserving IDT parameters.
        
        This method only changes:
        - Pipeline enable/disable settings
        - RGB gains (set to 0.0, 0.0, 0.0 for B&W)
        - Density curve name (set to B&W default from JSON, not overrides)
        
        It preserves:
        - IDT gamma (density_gamma)
        - Density dmax
        - Density matrix settings
        - All other user-configured parameters
        """
        pipeline_config = self.get_pipeline_config(film_type)
        
        # Get base defaults from JSON config (not overridden by device-specific files)
        # This ensures we get the proper B&W curve name "Ilford MGFB 2"
        base_params = {
            "density_gamma": pipeline_config.default_density_gamma,
            "density_dmax": pipeline_config.default_density_dmax,
            "density_matrix_name": pipeline_config.default_density_matrix_name,
            "density_curve_name": pipeline_config.default_density_curve_name,
            "default_rgb_gains": self._get_default_rgb_gains_from_json(film_type),
        }
        
        # Apply pipeline enable/disable settings (these should change for B&W)
        params.enable_density_inversion = pipeline_config.enable_density_inversion
        params.enable_density_matrix = pipeline_config.enable_density_matrix
        params.enable_rgb_gains = pipeline_config.enable_rgb_gains
        params.enable_density_curve = pipeline_config.enable_density_curve
        
        # Apply B&W-specific parameter changes
        if film_type in ["b&w_negative", "b&w_reversal"]:
            # Set RGB gains to neutral for B&W (from JSON config)
            params.rgb_gains = tuple(base_params["default_rgb_gains"])
            
            # Set density curve to B&W default from JSON (Ilford MGFB 2)
            params.density_curve_name = base_params["density_curve_name"]
        
        # DO NOT change: density_gamma, density_dmax, density_matrix_name
        # These should be preserved from the current settings
        
        return params
    
    def _get_default_rgb_gains_from_json(self, film_type: str) -> list:
        """Get default RGB gains directly from JSON config, bypassing overrides"""
        film_types_data = self._config_data.get("film_types", {})
        film_config = film_types_data.get(film_type, {})
        pipeline_data = film_config.get("pipeline", {})
        return pipeline_data.get("default_rgb_gains", [0.0, 0.0, 0.0])
    
    def _load_film_types_config(self) -> Dict[str, Any]:
        """从JSON配置文件加载胶片类型配置"""
        try:
            # 获取配置文件路径
            current_dir = Path(__file__).parent
            config_file = current_dir.parent / "config" / "film_types_config.json"
            
            if not config_file.exists():
                raise FileNotFoundError(f"Film types config file not found: {config_file}")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading film types config: {e}")
            # 返回基本的回退配置
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """获取回退配置（当JSON文件加载失败时使用）"""
        return {
            "film_types": {
                "color_negative_c41": {
                    "name": "彩色负片 C41",
                    "pipeline": {
                        "enable_density_inversion": True,
                        "enable_density_matrix": True,
                        "enable_rgb_gains": True,
                        "enable_density_curve": True,
                        "enable_color_space_conversion": True,
                        "convert_to_monochrome_in_idt": False,
                        "default_density_gamma": 1.0,
                        "default_density_dmax": 2.5,
                        "default_density_matrix_name": "Identity",
                        "default_density_curve_name": "linear"
                    },
                    "ui_state": {
                        "density_inversion_enabled": True,
                        "density_matrix_enabled": True,
                        "rgb_gains_enabled": True,
                        "density_curve_enabled": True,
                        "color_space_enabled": True,
                        "density_inversion_visible": True,
                        "density_matrix_visible": True,
                        "rgb_gains_visible": True,
                        "density_curve_visible": True,
                        "color_space_visible": True,
                        "disabled_tooltip": ""
                    }
                }
            }
        }
    
    def _load_configs_from_data(self):
        """从JSON数据创建配置对象"""
        film_types = self._config_data.get("film_types", {})
        
        for film_type, config in film_types.items():
            # 创建Pipeline配置
            pipeline_data = config.get("pipeline", {})
            pipeline_config = PipelineConfig(
                enable_density_inversion=pipeline_data.get("enable_density_inversion", True),
                enable_density_matrix=pipeline_data.get("enable_density_matrix", True),
                enable_rgb_gains=pipeline_data.get("enable_rgb_gains", True),
                enable_density_curve=pipeline_data.get("enable_density_curve", True),
                # 新的IDT字段，向后兼容
                enable_idt_gamma_correction=pipeline_data.get("enable_idt_gamma_correction", 
                                                            pipeline_data.get("enable_color_space_conversion", True)),
                enable_idt_color_space_conversion=pipeline_data.get("enable_idt_color_space_conversion", 
                                                                 pipeline_data.get("enable_color_space_conversion", True)),
                enable_color_space_conversion=pipeline_data.get("enable_color_space_conversion", True),
                convert_to_monochrome_in_idt=pipeline_data.get("convert_to_monochrome_in_idt", False),
                default_density_gamma=pipeline_data.get("default_density_gamma", 1.0),
                default_density_dmax=pipeline_data.get("default_density_dmax", 2.5),
                default_density_matrix_name=pipeline_data.get("default_density_matrix_name", "Identity"),
                default_density_curve_name=pipeline_data.get("default_density_curve_name", "linear")
            )
            self._pipeline_configs[film_type] = pipeline_config
            
            # 创建UI状态配置
            ui_data = config.get("ui_state", {})
            ui_config = UIStateConfig(
                density_inversion_enabled=ui_data.get("density_inversion_enabled", True),
                density_matrix_enabled=ui_data.get("density_matrix_enabled", True),
                rgb_gains_enabled=ui_data.get("rgb_gains_enabled", True),
                density_curve_enabled=ui_data.get("density_curve_enabled", True),
                color_space_enabled=ui_data.get("color_space_enabled", True),
                density_inversion_visible=ui_data.get("density_inversion_visible", True),
                density_matrix_visible=ui_data.get("density_matrix_visible", True),
                rgb_gains_visible=ui_data.get("rgb_gains_visible", True),
                density_curve_visible=ui_data.get("density_curve_visible", True),
                color_space_visible=ui_data.get("color_space_visible", True),
                disabled_tooltip=ui_data.get("disabled_tooltip", "")
            )
            self._ui_state_configs[film_type] = ui_config
    
    def is_monochrome_type(self, film_type: str) -> bool:
        """判断是否为黑白胶片类型"""
        return film_type in ["b&w_negative", "b&w_reversal"]
    
    def get_supported_film_types(self) -> list[str]:
        """获取所有支持的胶片类型"""
        return list(self._pipeline_configs.keys())
    
    def get_film_type_display_name(self, film_type: str) -> str:
        """获取胶片类型的显示名称"""
        film_types = self._config_data.get("film_types", {})
        return film_types.get(film_type, {}).get("name", film_type)