"""
独立工具启动器
提供统一的接口来启动各种独立工具
"""

from typing import Optional
from PySide6.QtWidgets import QWidget

from .file_classification_manager import FileClassificationManager
from .idt_calculator import IDTCalculatorWindow


class ToolsLauncher:
    """独立工具启动器"""
    
    @staticmethod
    def launch_file_classification_manager(parent: Optional[QWidget] = None) -> FileClassificationManager:
        """启动文件分类规则管理器"""
        manager = FileClassificationManager(parent)
        manager.show()
        return manager
    
    @staticmethod
    def launch_idt_calculator(parent: Optional[QWidget] = None) -> IDTCalculatorWindow:
        """启动精确通道分离IDT计算工具"""
        calculator = IDTCalculatorWindow(parent)
        calculator.show()
        return calculator
    
    @staticmethod
    def get_available_tools() -> list:
        """获取可用的工具列表"""
        return [
            {
                "name": "文件分类规则管理器",
                "description": "管理文件分类规则和默认预设文件",
                "launch_method": ToolsLauncher.launch_file_classification_manager
            },
            {
                "name": "精确通道分离IDT计算工具",
                "description": "通过三张光谱分离图片计算精确的IDT色彩空间",
                "launch_method": ToolsLauncher.launch_idt_calculator
            }
        ]


# 便捷函数
def launch_file_classification_manager(parent=None):
    """启动文件分类规则管理器的便捷函数"""
    return ToolsLauncher.launch_file_classification_manager(parent)

def launch_idt_calculator(parent=None):
    """启动精确通道分离IDT计算工具的便捷函数"""
    return ToolsLauncher.launch_idt_calculator(parent)
