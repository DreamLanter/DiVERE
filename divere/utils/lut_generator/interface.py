"""
LUT接口
为DiVERE提供简单的LUT生成接口，不暴露pipeline细节
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

from .core import LUTManager, create_3d_lut, create_1d_lut, save_lut_to_file


class DiVERELUTInterface:
    """DiVERE LUT接口 - 提供简单的LUT生成功能"""
    
    def __init__(self):
        """初始化接口"""
        self._manager = LUTManager()
    
    def generate_pipeline_lut(self, pipeline_config: Dict[str, Any], 
                            output_path: str, 
                            lut_type: str = "3D",
                            size: int = 32) -> bool:
        """
        生成校色pipeline的LUT
        
        Args:
            pipeline_config: pipeline配置（由core提供，接口不关心具体内容）
            output_path: 输出文件路径
            lut_type: LUT类型，"3D"或"1D"
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            if lut_type == "3D":
                # 创建变换函数（这里不关心pipeline的具体实现）
                transform_func = self._create_transform_from_config(pipeline_config)
                lut_info = self._manager.generate_3d_lut(
                    transform_func, size, "DiVERE Pipeline 3D LUT"
                )
            elif lut_type == "1D":
                # 从配置中提取曲线信息
                curves = self._extract_curves_from_config(pipeline_config)
                lut_info = self._manager.generate_1d_lut(
                    curves, size, "DiVERE Pipeline 1D LUT"
                )
            else:
                raise ValueError(f"不支持的LUT类型: {lut_type}")
            
            # 保存LUT
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成pipeline LUT失败: {e}")
            return False
    
    def generate_curve_lut(self, curves: Dict[str, List[Tuple[float, float]]], 
                          output_path: str, 
                          size: int = 1024) -> bool:
        """
        生成曲线LUT
        
        Args:
            curves: 曲线字典，键为'R', 'G', 'B'，值为控制点列表
            output_path: 输出文件路径
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            lut_info = self._manager.generate_1d_lut(
                curves, size, "DiVERE Curve 1D LUT"
            )
            return self._manager.save_lut(lut_info, output_path)
        except Exception as e:
            print(f"生成曲线LUT失败: {e}")
            return False
    
    def generate_identity_lut(self, output_path: str, 
                            lut_type: str = "3D", 
                            size: int = 32) -> bool:
        """
        生成单位LUT
        
        Args:
            output_path: 输出文件路径
            lut_type: LUT类型，"3D"或"1D"
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            if lut_type == "3D":
                # 创建单位变换函数
                def identity_transform(rgb):
                    return rgb
                
                lut_info = self._manager.generate_3d_lut(
                    identity_transform, size, "DiVERE Identity 3D LUT"
                )
            elif lut_type == "1D":
                # 创建单位曲线
                curves = {
                    'R': [(0.0, 0.0), (1.0, 1.0)],
                    'G': [(0.0, 0.0), (1.0, 1.0)],
                    'B': [(0.0, 0.0), (1.0, 1.0)]
                }
                lut_info = self._manager.generate_1d_lut(
                    curves, size, "DiVERE Identity 1D LUT"
                )
            else:
                raise ValueError(f"不支持的LUT类型: {lut_type}")
            
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成单位LUT失败: {e}")
            return False
    
    def load_lut(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        加载LUT文件
        
        Args:
            filepath: LUT文件路径
            
        Returns:
            LUT信息字典，如果加载失败返回None
        """
        return self._manager.load_lut(filepath)
    
    def _create_transform_from_config(self, config: Dict[str, Any]):
        """
        从配置创建变换函数
        """
        def transform(rgb):
            """实际的变换函数"""
            try:
                # 获取必要的组件
                params = config.get('params')
                context = config.get('context')
                the_enlarger = config.get('the_enlarger')
                
                if not all([params, context, the_enlarger]):
                    return rgb
                
                # 确保输入是numpy数组
                import numpy as np
                if not isinstance(rgb, np.ndarray):
                    rgb = np.array(rgb, dtype=np.float32)
                
                # 处理批量RGB值：输入通常是(N, 3)形状
                original_shape = rgb.shape
                
                # 将RGB值重塑为图像格式，处理器期望(H, W, 3)
                if rgb.ndim == 1:
                    # 单个颜色值 (3,) -> (1, 1, 3)
                    fake_array = rgb.reshape(1, 1, 3).astype(np.float32)
                elif rgb.ndim == 2:
                    # 批量颜色值 (N, 3) -> (1, N, 3) 作为一行图像
                    fake_array = rgb.reshape(1, rgb.shape[0], 3).astype(np.float32)
                else:
                    fake_array = rgb.astype(np.float32)
                
                # 创建ImageData对象
                from divere.core.data_types import ImageData
                fake_image = ImageData(
                    array=fake_array,
                    file_path="",
                    metadata={}
                )
                
                # 使用the_enlarger的完整管线处理
                # include_curve参数根据LUT类型决定
                include_curve = params.enable_density_curve
                processed_image = the_enlarger.apply_full_pipeline(
                    fake_image, params, include_curve=include_curve
                )
                
                if processed_image is None or processed_image.array is None:
                    return rgb
                
                # 提取处理后的数组并恢复原始形状
                processed_array = processed_image.array
                
                # 裁剪到有效范围 [0, 1]
                processed_array = np.clip(processed_array, 0.0, 1.0)
                
                if original_shape == (3,):
                    # 单个颜色值
                    return processed_array.flatten()[:3]
                elif len(original_shape) == 2 and original_shape[1] == 3:
                    # 批量颜色值：(1, N, 3) -> (N, 3)
                    return processed_array.reshape(-1, 3)[:original_shape[0]]
                else:
                    return processed_array
                    
            except Exception as e:
                print(f"LUT变换失败: {e}")
                import traceback
                traceback.print_exc()
                return rgb
        
        return transform
    
    def _extract_curves_from_config(self, config: Dict[str, Any]) -> Dict[str, List[Tuple[float, float]]]:
        """
        从配置中提取曲线信息
        """
        curves = {}
        
        # 提取RGB主曲线
        if 'curve_points' in config:
            curves['R'] = config['curve_points']
            curves['G'] = config['curve_points']
            curves['B'] = config['curve_points']
        
        # 提取单通道曲线
        if 'curve_points_r' in config:
            curves['R'] = config['curve_points_r']
        if 'curve_points_g' in config:
            curves['G'] = config['curve_points_g']
        if 'curve_points_b' in config:
            curves['B'] = config['curve_points_b']
        
        return curves


# 便捷函数
def generate_pipeline_lut(pipeline_config: Dict[str, Any], 
                         output_path: str, 
                         lut_type: str = "3D",
                         size: int = 32) -> bool:
    """生成pipeline LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_pipeline_lut(pipeline_config, output_path, lut_type, size)


def generate_curve_lut(curves: Dict[str, List[Tuple[float, float]]], 
                      output_path: str, 
                      size: int = 1024) -> bool:
    """生成曲线LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_curve_lut(curves, output_path, size)


def generate_identity_lut(output_path: str, 
                         lut_type: str = "3D", 
                         size: int = 32) -> bool:
    """生成单位LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_identity_lut(output_path, lut_type, size) 