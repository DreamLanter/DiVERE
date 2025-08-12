"""
预览配置预设
提供各种常用的预览配置组合
"""

from ..core.data_types import PreviewConfig


class PreviewPresets:
    """预设的预览配置组合"""
    
    @staticmethod
    def high_quality() -> PreviewConfig:
        """高质量预览配置 - 适合最终确认"""
        return PreviewConfig(
            preview_max_size=4096,      # 4K预览
            proxy_max_size=3840,        # 4K代理
            gpu_threshold=512 * 512,    # 512x512就用GPU
            preview_quality='cubic',    # 立方插值
            preview_lut_size=64,        # 高精度LUT
            full_lut_size=128,          # 超高精度LUT
        )
    
    @staticmethod 
    def balanced() -> PreviewConfig:
        """平衡配置 - 默认推荐"""
        return PreviewConfig(
            preview_max_size=2048,      # 2K预览
            proxy_max_size=2000,        # 2K代理  
            gpu_threshold=1024 * 1024,  # 1M像素用GPU
            preview_quality='linear',   # 线性插值
            preview_lut_size=32,        # 标准LUT
            full_lut_size=64,           # 标准精度LUT
        )
    
    @staticmethod
    def fast() -> PreviewConfig:
        """快速预览配置 - 适合实时调整"""
        return PreviewConfig(
            preview_max_size=1024,      # 1K预览
            proxy_max_size=1200,        # 1.2K代理
            gpu_threshold=2048 * 2048,  # 4M像素才用GPU
            preview_quality='linear',   # 线性插值
            preview_lut_size=16,        # 低精度LUT
            full_lut_size=32,           # 中等精度LUT
        )
    
    @staticmethod
    def mobile() -> PreviewConfig:
        """移动设备配置 - 低功耗"""
        return PreviewConfig(
            preview_max_size=768,       # 小预览
            proxy_max_size=960,         # 小代理
            gpu_threshold=4096 * 4096,  # 很大才用GPU（节电）
            preview_quality='nearest',  # 最快插值
            preview_lut_size=16,        # 最小LUT
            full_lut_size=32,           # 中等LUT
        )
    
    @staticmethod
    def gpu_aggressive() -> PreviewConfig:
        """GPU积极使用配置 - 有强GPU时"""
        return PreviewConfig(
            preview_max_size=3072,      # 3K预览
            proxy_max_size=2880,        # 2.8K代理
            gpu_threshold=256 * 256,    # 很小就用GPU
            preview_quality='cubic',    # 高质量插值
            preview_lut_size=64,        # 高精度LUT
            full_lut_size=128,          # 超高精度LUT
        )


# 使用示例
"""
from divere.config.preview_presets import PreviewPresets
from divere.core.the_enlarger import TheEnlarger

# 创建高质量预览配置的放大机
enlarger = TheEnlarger(preview_config=PreviewPresets.high_quality())

# 或者快速预览配置
fast_enlarger = TheEnlarger(preview_config=PreviewPresets.fast())

# 自定义配置
from divere.core.data_types import PreviewConfig
custom_config = PreviewConfig(
    preview_max_size=1536,
    gpu_threshold=800 * 600
)
custom_enlarger = TheEnlarger(preview_config=custom_config)
"""
