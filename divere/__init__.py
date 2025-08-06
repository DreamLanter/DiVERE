"""
DiVERE - 专业胶片校色工具
基于ACEScg Linear工作流的数字化胶片后期处理
"""

__version__ = "0.1.0"
__author__ = "V7"
__email__ = "vanadis@yeah.net"

from .core.image_manager import ImageManager
from .core.color_space import ColorSpaceManager
from .core.the_enlarger import TheEnlarger
from .core.lut_processor import LUTProcessor
from .core.curve_editor import CurveEditor

__all__ = [
    "ImageManager",
    "ColorSpaceManager", 
    "TheEnlarger",
    "LUTProcessor",
    "CurveEditor",
] 