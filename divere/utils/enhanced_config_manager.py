#!/usr/bin/env python3
"""
增强配置管理器
支持用户配置目录和配置文件优先级管理
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
import platform
import time


class EnhancedConfigManager:
    """增强配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.app_name = "DiVERE"
        
        # 获取用户配置目录
        self.user_config_dir = self._get_user_config_dir()
        self.user_config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建用户配置子目录
        self.user_colorspace_dir = self.user_config_dir / "config" / "colorspace"
        self.user_curves_dir = self.user_config_dir / "config" / "curves"
        self.user_matrices_dir = self.user_config_dir / "config" / "matrices"
        self.user_models_dir = self.user_config_dir / "models"
        self.user_logs_dir = self.user_config_dir / "logs"
        
        # 创建目录
        for dir_path in [self.user_colorspace_dir, self.user_curves_dir, 
                        self.user_matrices_dir, self.user_models_dir, self.user_logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 应用内置配置目录（统一入口）：优先二进制旁顶层 config，回退到包内 divere/config
        try:
            from .app_paths import get_data_dir
            self.app_config_dir = get_data_dir("config")
        except Exception:
            self.app_config_dir = Path("config")
        
        # 应用设置文件
        self.app_settings_file = self.user_config_dir / "config" / "app_settings.json"
        self.app_settings = self._load_app_settings()
    
    def _get_user_config_dir(self) -> Path:
        """获取用户配置目录"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / self.app_name
        elif system == "Windows":
            return Path.home() / "AppData" / "Local" / self.app_name
        elif system == "Linux":
            return Path.home() / ".config" / self.app_name
        else:
            # 默认使用当前目录
            return Path.cwd() / "user_config"
    
    def _load_app_settings(self) -> Dict[str, Any]:
        """加载应用设置"""
        default_settings = {
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
            },
            "config": {
                "show_user_config_dir": True,
                "auto_backup_config": True
            }
        }
        
        if self.app_settings_file.exists():
            try:
                with open(self.app_settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    return self._merge_configs(default_settings, loaded_settings)
            except Exception as e:
                print(f"加载应用设置失败: {e}")
                return default_settings
        else:
            # 保存默认设置
            self._save_app_settings(default_settings)
            return default_settings
    
    def _save_app_settings(self, settings: Dict[str, Any] = None):
        """保存应用设置"""
        if settings is None:
            settings = self.app_settings
        
        try:
            with open(self.app_settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存应用设置失败: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_files(self, config_type: str) -> List[Path]:
        """
        获取指定类型的配置文件列表（用户配置优先）
        
        Args:
            config_type: 配置类型 ("colorspace", "curves", "matrices")
            
        Returns:
            配置文件路径列表，用户配置在前
        """
        user_dir = getattr(self, f"user_{config_type}_dir")
        app_dir = self.app_config_dir / config_type
        
        config_files = []
        
        # 首先添加用户配置文件
        if user_dir.exists():
            for json_file in user_dir.glob("*.json"):
                config_files.append(json_file)
        
        # 然后添加应用内置配置文件（如果用户没有同名文件）
        if app_dir.exists():
            for json_file in app_dir.glob("*.json"):
                user_file = user_dir / json_file.name
                if not user_file.exists():  # 用户没有同名文件时才添加
                    config_files.append(json_file)
        
        return config_files
    
    def load_config_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """加载单个配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败 {file_path}: {e}")
            return None
    
    def save_user_config(self, config_type: str, name: str, data: Dict[str, Any]) -> bool:
        """
        保存用户配置文件
        
        Args:
            config_type: 配置类型 ("colorspace", "curves", "matrices")
            name: 配置名称（文件名，不含扩展名）
            data: 配置数据
            
        Returns:
            是否保存成功
        """
        user_dir = getattr(self, f"user_{config_type}_dir")
        file_path = user_dir / f"{name}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"用户配置已保存: {file_path}")
            return True
        except Exception as e:
            print(f"保存用户配置失败: {e}")
            return False
    
    def delete_user_config(self, config_type: str, name: str) -> bool:
        """删除用户配置文件"""
        user_dir = getattr(self, f"user_{config_type}_dir")
        file_path = user_dir / f"{name}.json"
        
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"用户配置已删除: {file_path}")
                return True
            except Exception as e:
                print(f"删除用户配置失败: {e}")
                return False
        return False
    
    def copy_default_to_user(self, config_type: str, name: str) -> bool:
        """将默认配置复制到用户配置目录"""
        app_dir = self.app_config_dir / config_type
        user_dir = getattr(self, f"user_{config_type}_dir")
        
        source_file = app_dir / f"{name}.json"
        target_file = user_dir / f"{name}.json"
        
        if source_file.exists() and not target_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"默认配置已复制到用户目录: {target_file}")
                return True
            except Exception as e:
                print(f"复制配置失败: {e}")
                return False
        return False
    
    def get_user_config_dir_path(self) -> str:
        """获取用户配置目录路径（用于显示给用户）"""
        return str(self.user_config_dir)
    
    def open_user_config_dir(self):
        """打开用户配置目录"""
        try:
            if platform.system() == "Darwin":  # macOS
                os.system(f"open '{self.user_config_dir}'")
            elif platform.system() == "Windows":
                os.system(f"explorer '{self.user_config_dir}'")
            elif platform.system() == "Linux":
                os.system(f"xdg-open '{self.user_config_dir}'")
        except Exception as e:
            print(f"打开配置目录失败: {e}")
    
    def backup_user_configs(self) -> bool:
        """备份用户配置"""
        backup_dir = self.user_config_dir / "backup" / f"backup_{int(time.time())}"
        
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.user_config_dir / "config", backup_dir / "config", dirs_exist_ok=True)
            print(f"用户配置已备份到: {backup_dir}")
            return True
        except Exception as e:
            print(f"备份用户配置失败: {e}")
            return False
    
    # 应用设置相关方法
    def get_directory(self, directory_type: str) -> str:
        """获取指定类型的目录路径"""
        directory = self.app_settings.get("directories", {}).get(directory_type, "")
        
        if directory and Path(directory).exists():
            return directory
        else:
            return ""
    
    def set_directory(self, directory_type: str, path: str):
        """设置指定类型的目录路径"""
        if "directories" not in self.app_settings:
            self.app_settings["directories"] = {}
        
        path_obj = Path(path)
        if path_obj.is_file():
            path = str(path_obj.parent)
        elif path_obj.is_dir():
            path = str(path_obj)
        else:
            parent = path_obj.parent
            if parent.exists() or parent.mkdir(parents=True, exist_ok=True):
                path = str(parent)
            else:
                return
        
        self.app_settings["directories"][directory_type] = path
        self._save_app_settings()
    
    def get_ui_setting(self, key: str, default=None):
        """获取UI设置"""
        return self.app_settings.get("ui", {}).get(key, default)
    
    def set_ui_setting(self, key: str, value):
        """设置UI设置"""
        if "ui" not in self.app_settings:
            self.app_settings["ui"] = {}
        self.app_settings["ui"][key] = value
        self._save_app_settings()
    
    def get_default_setting(self, key: str, default=None):
        """获取默认设置"""
        return self.app_settings.get("defaults", {}).get(key, default)
    
    def set_default_setting(self, key: str, value):
        """设置默认设置"""
        if "defaults" not in self.app_settings:
            self.app_settings["defaults"] = {}
        self.app_settings["defaults"][key] = value
        self._save_app_settings()


# 全局增强配置管理器实例
enhanced_config_manager = EnhancedConfigManager()
