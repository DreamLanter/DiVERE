"""
精确通道分离IDT计算工具

这个模块提供了一个独立的工具，用于通过三张不同光谱的图片（红光、绿光、蓝光）
计算精确的输入设备转换（IDT）色彩空间。

主要功能：
- 加载三张光谱分离的图片
- 提取九宫格中心区域的RGB值
- 使用CMA-ES算法优化3x3线性变换矩阵
- 计算原始色彩空间的原色坐标
- 保存为标准的IDT色彩空间配置文件
"""

from .idt_calculator_window import IDTCalculatorWindow

__all__ = ['IDTCalculatorWindow']