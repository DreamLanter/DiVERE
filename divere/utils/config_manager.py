#!/usr/bin/env python3
"""
配置管理器
用于管理应用程序的配置信息，包括目录记忆功能
"""

import json
import os
from pathlib import Path
from typing import Optional


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config/app_settings.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        default_config = {
            "directories": {
                "open_image": "",
                "save_image": "",
                "save_lut": "",
                "save_matrix": ""
            },
            "ui": {
                "window_size": [1200, 800],
                "window_position": [100, 100]
            },
            "defaults": {
                "input_color_space": "sRGB",
                "output_color_space": "sRGB"
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 合并默认配置和加载的配置
                    self._merge_configs(default_config, loaded_config)
                    return default_config
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        else:
            # 创建配置文件目录
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            # 保存默认配置
            self._save_config(default_config)
            return default_config
    
    def _merge_configs(self, default: dict, loaded: dict):
        """递归合并配置"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def _save_config(self, config: dict = None):
        """保存配置文件"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_directory(self, directory_type: str) -> str:
        """
        获取指定类型的目录路径
        
        Args:
            directory_type: 目录类型 ("open_image", "save_image", "save_lut")
            
        Returns:
            目录路径，如果不存在或无效则返回空字符串
        """
        directory = self.config.get("directories", {}).get(directory_type, "")
        
        # 验证目录是否存在
        if directory and Path(directory).exists():
            return directory
        else:
            return ""
    
    def set_directory(self, directory_type: str, path: str):
        """
        设置指定类型的目录路径
        
        Args:
            directory_type: 目录类型 ("open_image", "save_image", "save_lut")
            path: 目录路径
        """
        if "directories" not in self.config:
            self.config["directories"] = {}
        
        # 确保路径是目录
        path_obj = Path(path)
        if path_obj.is_file():
            path = str(path_obj.parent)
        elif path_obj.is_dir():
            path = str(path_obj)
        else:
            # 如果路径不存在，尝试创建父目录
            parent = path_obj.parent
            if parent.exists() or parent.mkdir(parents=True, exist_ok=True):
                path = str(parent)
            else:
                return  # 无法创建目录，不保存
        
        self.config["directories"][directory_type] = path
        self._save_config()
    
    def get_ui_setting(self, key: str, default=None):
        """获取UI设置"""
        return self.config.get("ui", {}).get(key, default)
    
    def set_ui_setting(self, key: str, value):
        """设置UI设置"""
        if "ui" not in self.config:
            self.config["ui"] = {}
        self.config["ui"][key] = value
        self._save_config()
    
    def get_default_setting(self, key: str, default=None):
        """获取默认设置"""
        return self.config.get("defaults", {}).get(key, default)
    
    def set_default_setting(self, key: str, value):
        """设置默认设置"""
        if "defaults" not in self.config:
            self.config["defaults"] = {}
        self.config["defaults"][key] = value
        self._save_config()


# 全局配置管理器实例
config_manager = ConfigManager() 