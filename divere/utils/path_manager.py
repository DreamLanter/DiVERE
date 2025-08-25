"""
路径管理模块
使用addpath来管理各种配置和资源路径
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

class PathManager:
    """路径管理器，使用addpath方式管理各种路径"""
    
    def __init__(self):
        self._paths: Dict[str, List[str]] = {
            "config": [],
            "defaults": [],
            "colorspace": [],
            "curves": [],
            "matrices": [],
            "assets": [],
            "models": [],
            "test_data": []
        }
        self._initialized = False
        self._initialize_paths()
    
    def _initialize_paths(self):
        """初始化所有路径"""
        if self._initialized:
            return
            
        # 获取项目根目录
        project_root = self._get_project_root()
        
        # 配置路径
        self._paths["config"].extend([
            os.path.join(project_root, "divere", "config"),
            os.path.join(project_root, "divere", "config", "defaults"),
            os.path.join(project_root, "divere", "config", "colorchecker"),
            os.path.join(project_root, "divere", "config", "colorspace"),
            os.path.join(project_root, "divere", "config", "curves"),
            os.path.join(project_root, "divere", "config", "matrices")
        ])
        
        # 默认预设路径
        self._paths["defaults"].extend([
            os.path.join(project_root, "divere", "config", "defaults")
        ])
        
        # 色彩空间路径
        self._paths["colorspace"].extend([
            os.path.join(project_root, "divere", "config", "colorspace"),
            os.path.join(project_root, "divere", "config", "colorspace", "legacy"),
            os.path.join(project_root, "divere", "config", "colorspace", "icc")
        ])
        
        # 曲线路径
        self._paths["curves"].extend([
            os.path.join(project_root, "divere", "config", "curves")
        ])
        
        # 矩阵路径
        self._paths["matrices"].extend([
            os.path.join(project_root, "divere", "config", "matrices")
        ])
        
        # 资源路径
        self._paths["assets"].extend([
            os.path.join(project_root, "divere", "assets")
        ])
        
        # 模型路径
        self._paths["models"].extend([
            os.path.join(project_root, "divere", "models")
        ])
        
        # 测试数据路径
        self._paths["test_data"].extend([
            os.path.join(project_root, "test_scans")
        ])
        
        # 添加所有路径到Python路径
        self._add_all_paths()
        
        self._initialized = True
    
    def _get_project_root(self) -> str:
        """获取项目根目录"""
        # 尝试多种方式获取项目根目录
        current_file = os.path.abspath(__file__)
        
        # 从当前文件向上查找，直到找到包含pyproject.toml的目录
        current_dir = os.path.dirname(current_file)
        while current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # 如果找不到，使用当前工作目录
        return os.getcwd()
    
    def _add_all_paths(self):
        """将所有路径添加到Python路径"""
        for path_list in self._paths.values():
            for path in path_list:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
    
    def add_path(self, category: str, path: str):
        """添加新路径到指定类别"""
        if category not in self._paths:
            self._paths[category] = []
        
        if path not in self._paths[category]:
            self._paths[category].append(path)
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
    
    def remove_path(self, category: str, path: str):
        """从指定类别移除路径"""
        if category in self._paths and path in self._paths[category]:
            self._paths[category].remove(path)
            if path in sys.path:
                sys.path.remove(path)
    
    def get_paths(self, category: str) -> List[str]:
        """获取指定类别的所有路径"""
        return self._paths.get(category, [])
    
    def find_file(self, filename: str, category: str = None) -> Optional[str]:
        """在指定类别中查找文件"""
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    return file_path
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    file_path = os.path.join(path, filename)
                    if os.path.exists(file_path):
                        return file_path
        
        return None
    
    def find_files_by_pattern(self, pattern: str, category: str = None) -> List[str]:
        """根据模式查找文件"""
        import glob
        found_files = []
        
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                search_pattern = os.path.join(path, pattern)
                found_files.extend(glob.glob(search_pattern))
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    search_pattern = os.path.join(path, pattern)
                    found_files.extend(glob.glob(search_pattern))
        
        return found_files
    
    def resolve_path(self, relative_path: str, category: str = None) -> Optional[str]:
        """解析相对路径为绝对路径"""
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                full_path = os.path.join(path, relative_path)
                if os.path.exists(full_path):
                    return full_path
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    full_path = os.path.join(path, relative_path)
                    if os.path.exists(full_path):
                        return full_path
        
        return None
    
    def get_default_preset_path(self, preset_name: str) -> Optional[str]:
        """获取默认预设文件的完整路径"""
        return self.find_file(preset_name, "defaults")
    
    def get_config_path(self, config_name: str) -> Optional[str]:
        """获取配置文件的完整路径"""
        return self.find_file(config_name, "config")
    
    def list_default_presets(self) -> List[str]:
        """列出所有可用的默认预设文件"""
        preset_files = []
        for path in self._paths["defaults"]:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('.json'):
                        preset_files.append(file)
        return sorted(preset_files)
    
    def get_path_info(self) -> Dict[str, Any]:
        """获取路径信息"""
        info = {}
        for category, paths in self._paths.items():
            info[category] = {
                "paths": paths,
                "exists": [os.path.exists(p) for p in paths],
                "files": []
            }
            
            # 统计每个路径下的文件数量
            for path in paths:
                if os.path.exists(path):
                    try:
                        files = os.listdir(path)
                        info[category]["files"].append({
                            "path": path,
                            "count": len(files),
                            "sample_files": files[:5]  # 前5个文件作为示例
                        })
                    except Exception:
                        info[category]["files"].append({
                            "path": path,
                            "count": 0,
                            "sample_files": []
                        })
        
        return info


# 全局路径管理器实例
path_manager = PathManager()

# 便捷函数
def add_path(category: str, path: str):
    """添加路径的便捷函数"""
    path_manager.add_path(category, path)

def find_file(filename: str, category: str = None) -> Optional[str]:
    """查找文件的便捷函数"""
    return path_manager.find_file(filename, category)

def resolve_path(relative_path: str, category: str = None) -> Optional[str]:
    """解析路径的便捷函数"""
    return path_manager.resolve_path(relative_path, category)

def get_default_preset_path(preset_name: str) -> Optional[str]:
    """获取默认预设路径的便捷函数"""
    return path_manager.get_default_preset_path(preset_name)

def list_default_presets() -> List[str]:
    """列出默认预设的便捷函数"""
    return path_manager.list_default_presets()
