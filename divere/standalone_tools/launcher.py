"""
独立工具启动器
提供统一的接口来启动各种独立工具
"""

from typing import Optional
from PySide6.QtWidgets import QWidget

from .file_classification_manager import FileClassificationManager


class ToolsLauncher:
    """独立工具启动器"""
    
    @staticmethod
    def launch_file_classification_manager(parent: Optional[QWidget] = None) -> FileClassificationManager:
        """启动文件分类规则管理器"""
        manager = FileClassificationManager(parent)
        manager.show()
        return manager
    
    @staticmethod
    def get_available_tools() -> list:
        """获取可用的工具列表"""
        return [
            {
                "name": "文件分类规则管理器",
                "description": "管理文件分类规则和默认预设文件",
                "launch_method": ToolsLauncher.launch_file_classification_manager
            }
        ]


# 便捷函数
def launch_file_classification_manager(parent=None):
    """启动文件分类规则管理器的便捷函数"""
    return ToolsLauncher.launch_file_classification_manager(parent)
