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
    
    def generate_input_device_transform_lut(self, idt_config: Dict[str, Any],
                                          output_path: str,
                                          size: int = 32) -> bool:
        """
        生成输入设备转换LUT - 包含完整的输入色彩科学处理
        包含：IDT Gamma幂次变换 + 色彩空间矩阵转换 + 增益向量应用
        
        Args:
            idt_config: IDT配置，包含idt_gamma, context, input_colorspace_name
            output_path: 输出文件路径
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            import numpy as np
            
            # 预先获取转换参数以避免重复计算
            context = idt_config.get("context")
            input_colorspace_name = idt_config.get("input_colorspace_name")
            idt_gamma = idt_config.get("idt_gamma", 1.0)
            
            # 获取色彩空间转换矩阵和增益向量
            conversion_matrix = np.eye(3, dtype=np.float32)
            gain_vector = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            
            if context and input_colorspace_name and input_colorspace_name != "ACEScg":
                try:
                    conversion_matrix, gain_vector = context.color_space_manager.calculate_color_space_conversion(
                        input_colorspace_name, "ACEScg"
                    )
                    conversion_matrix = conversion_matrix.astype(np.float32)
                    gain_vector = gain_vector.astype(np.float32)
                    print(f"IDT LUT: 获取到转换矩阵和增益向量，从 {input_colorspace_name} 到 ACEScg")
                    print(f"  - IDT Gamma: {idt_gamma}")
                    print(f"  - 转换矩阵: {conversion_matrix}")
                    print(f"  - 增益向量: {gain_vector}")
                except Exception as e:
                    print(f"获取色彩空间转换参数失败: {e}")
            
            # 创建优化的IDT变换函数
            def idt_transform(rgb):
                try:
                    if not isinstance(rgb, np.ndarray):
                        rgb = np.array(rgb, dtype=np.float32)
                    
                    original_shape = rgb.shape
                    
                    # 重塑为批量处理格式
                    if rgb.ndim == 1:
                        batch_rgb = rgb.reshape(1, 3)
                    elif rgb.ndim == 2:
                        batch_rgb = rgb
                    else:
                        batch_rgb = rgb.reshape(-1, 3)
                    
                    result = batch_rgb.copy().astype(np.float32)
                    
                    # Step 1: 应用IDT Gamma幂次变换
                    if abs(idt_gamma - 1.0) > 1e-6:
                        # 保护数值范围并应用幂次变换
                        result = np.clip(result, 0.0, 1.0)
                        result = np.power(result, idt_gamma, dtype=np.float32)
                    
                    # Step 2: 应用色彩空间矩阵转换
                    if not np.allclose(conversion_matrix, np.eye(3)):
                        # 矩阵乘法: result = result @ conversion_matrix.T
                        result = np.dot(result, conversion_matrix.T)
                    
                    # Step 3: 应用增益向量
                    if not np.allclose(gain_vector, np.array([1.0, 1.0, 1.0])):
                        result = result * gain_vector
                    
                    # 裁剪到有效范围
                    result = np.clip(result, 0.0, 1.0)
                    
                    # 恢复原始形状
                    if original_shape == (3,):
                        return result.flatten()[:3]
                    elif len(original_shape) == 2 and original_shape[1] == 3:
                        return result[:original_shape[0]]
                    else:
                        return result.reshape(original_shape)
                        
                except Exception as e:
                    print(f"IDT变换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return rgb
            
            # 生成3D LUT
            lut_info = self._manager.generate_3d_lut(
                idt_transform, size, "DiVERE Input Device Transform 3D LUT"
            )
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成输入设备转换LUT失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_density_curve_lut(self, curves: Dict[str, List[Tuple[float, float]]],
                                 output_path: str,
                                 size: int = 1024,
                                 screen_glare_compensation: float = 0.0) -> bool:
        """
        生成密度曲线LUT - 在密度空间应用曲线，然后在线性空间应用屏幕反光补偿
        
        Args:
            curves: 曲线字典，键为'R', 'G', 'B'，值为控制点列表
            output_path: 输出文件路径
            size: LUT大小
            screen_glare_compensation: 屏幕反光补偿量(0.0-0.2)，在线性空间应用
            
        Returns:
            是否生成成功
        """
        try:
            import numpy as np
            
            # 创建密度空间曲线变换函数
            def density_curve_transform(rgb):
                try:
                    if not isinstance(rgb, np.ndarray):
                        rgb = np.array(rgb, dtype=np.float32)
                    
                    original_shape = rgb.shape
                    
                    # 重塑为批量处理格式
                    if rgb.ndim == 1:
                        batch_rgb = rgb.reshape(1, 3)
                    elif rgb.ndim == 2:
                        batch_rgb = rgb
                    else:
                        batch_rgb = rgb.reshape(-1, 3)
                    
                    result = np.zeros_like(batch_rgb)
                    
                    # 对每个通道分别处理
                    channel_map = {'R': 0, 'G': 1, 'B': 2}
                    
                    for channel_name, channel_idx in channel_map.items():
                        channel_curves = curves.get(channel_name, [(0.0, 0.0), (1.0, 1.0)])
                        
                        # 检查是否为默认直线
                        if channel_curves == [(0.0, 0.0), (1.0, 1.0)]:
                            result[:, channel_idx] = batch_rgb[:, channel_idx]
                            continue
                        
                        # 线性 -> 密度空间
                        # 避免log(0)
                        safe_input = np.maximum(batch_rgb[:, channel_idx], 1e-10)
                        density = -np.log10(safe_input)
                        
                        # 归一化密度到[0,1]区间进行曲线查找
                        # 使用log10(65536)作为最大密度值 (FilmMathOps中的_LOG65536)
                        LOG65536 = np.log10(65536.0)
                        normalized_density = 1.0 - np.clip(density / LOG65536, 0.0, 1.0)
                        
                        # 应用曲线插值
                        curve_points = np.array(channel_curves, dtype=np.float32)
                        x_points = curve_points[:, 0]
                        y_points = curve_points[:, 1]
                        
                        # 线性插值曲线
                        curve_output = np.interp(normalized_density, x_points, y_points)
                        
                        # 转换回密度空间
                        output_density = (1.0 - curve_output) * LOG65536
                        
                        # 密度 -> 线性空间
                        result[:, channel_idx] = np.power(10.0, -output_density)
                    
                    # 应用屏幕反光补偿（在线性空间）
                    if screen_glare_compensation > 0.0:
                        result = np.maximum(0.0, result - screen_glare_compensation)
                    
                    # 裁剪到有效范围
                    result = np.clip(result, 0.0, 1.0)
                    
                    # 恢复原始形状
                    if original_shape == (3,):
                        return result.flatten()[:3]
                    elif len(original_shape) == 2 and original_shape[1] == 3:
                        return result[:original_shape[0]]
                    else:
                        return result.reshape(original_shape)
                        
                except Exception as e:
                    print(f"密度曲线变换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return rgb
            
            # 生成1D LUT
            lut_info = self._manager.generate_1d_lut(
                curves, size, "DiVERE Density Curve 1D LUT"
            )
            
            # 但是我们需要重新生成数据使用我们的密度变换
            from .core import LUT1DGenerator
            generator = LUT1DGenerator(size)
            
            # 生成x值并应用密度曲线变换
            x = np.linspace(0, 1, size)
            batch_x = np.column_stack([x, x, x])  # 创建RGB批量数据
            
            # 应用密度变换
            lut_data = density_curve_transform(batch_x)
            
            # 更新LUT信息
            lut_info['data'] = lut_data
            lut_info['generator'] = generator
            
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成密度曲线LUT失败: {e}")
            return False
    
    def validate_lut_chain_consistency(self, context, params, 
                                     idt_lut_path: str, 
                                     correction_lut_path: str,
                                     curve_lut_path: str,
                                     test_samples: int = 100) -> Dict[str, float]:
        """
        验证三个LUT串联使用与DiVERE完整管线的数学一致性
        
        Args:
            context: 应用上下文
            params: 处理参数
            idt_lut_path, correction_lut_path, curve_lut_path: 三个LUT文件路径
            test_samples: 测试样本数量
            
        Returns:
            包含误差统计的字典
        """
        try:
            import numpy as np
            from divere.core.data_types import ImageData
            
            # 生成测试样本
            np.random.seed(42)  # 确保可重现
            test_colors = np.random.rand(test_samples, 3).astype(np.float32)
            
            # 方法1：DiVERE完整管线处理
            fake_image = ImageData(
                array=test_colors.reshape(1, test_samples, 3),
                file_path="",
                metadata={}
            )
            
            # 先应用IDT gamma
            idt_gamma = context.get_current_idt_gamma()
            if abs(idt_gamma - 1.0) > 1e-6:
                test_colors_idt = context.the_enlarger.pipeline_processor.math_ops.apply_power(
                    fake_image.array, idt_gamma, use_optimization=False
                )
                fake_image = fake_image.copy_with_new_array(test_colors_idt)
            
            # 色彩空间转换
            converted_image = context.color_space_manager.convert_to_working_space(
                fake_image, params.input_color_space_name, skip_gamma_inverse=True
            )
            
            # 完整管线处理
            divere_result = context.the_enlarger.apply_full_pipeline(
                converted_image, params, include_curve=True, for_export=True,
                convert_to_monochrome_in_idt=False, monochrome_converter=None
            )
            divere_colors = divere_result.array.reshape(-1, 3)
            
            # 方法2：三个LUT串联（这里只是模拟，实际需要加载LUT文件）
            # 由于我们没有实际的LUT加载和应用函数，这里使用我们的变换函数进行验证
            
            # 1. IDT变换
            idt_config = {
                "idt_gamma": idt_gamma,
                "context": context,
                "input_colorspace_name": params.input_color_space_name
            }
            
            def create_idt_transform():
                def idt_transform(rgb):
                    try:
                        fake_array = rgb.reshape(1, -1, 3).astype(np.float32)
                        
                        # IDT Gamma
                        if abs(idt_gamma - 1.0) > 1e-6:
                            fake_array = context.the_enlarger.pipeline_processor.math_ops.apply_power(
                                fake_array, idt_gamma, use_optimization=False
                            )
                        
                        # 色彩空间转换
                        temp_image = ImageData(array=fake_array, file_path="", metadata={})
                        converted_image = context.color_space_manager.convert_to_working_space(
                            temp_image, params.input_color_space_name, skip_gamma_inverse=True
                        )
                        if converted_image and converted_image.array is not None:
                            fake_array = converted_image.array
                        
                        return np.clip(fake_array, 0.0, 1.0).reshape(-1, 3)
                    except Exception:
                        return rgb
                return idt_transform
            
            idt_transform = create_idt_transform()
            step1_colors = idt_transform(test_colors)
            
            # 2. 反相校色变换
            color_params = params.copy()
            color_params.enable_density_curve = False
            color_params.curve_points = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_r = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_g = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_b = [(0.0, 0.0), (1.0, 1.0)]
            
            step1_image = ImageData(array=step1_colors.reshape(1, -1, 3), file_path="", metadata={})
            step2_result = context.the_enlarger.apply_full_pipeline(
                step1_image, color_params, include_curve=False, for_export=True,
                convert_to_monochrome_in_idt=False, monochrome_converter=None
            )
            step2_colors = step2_result.array.reshape(-1, 3)
            
            # 3. 密度曲线变换
            curves = {
                'R': params.curve_points_r or [(0.0, 0.0), (1.0, 1.0)],
                'G': params.curve_points_g or [(0.0, 0.0), (1.0, 1.0)],
                'B': params.curve_points_b or [(0.0, 0.0), (1.0, 1.0)]
            }
            if params.curve_points and params.curve_points != [(0.0, 0.0), (1.0, 1.0)]:
                curves['R'] = curves['G'] = curves['B'] = params.curve_points
            
            def create_density_curve_transform():
                def density_curve_transform(rgb):
                    try:
                        batch_rgb = rgb.reshape(-1, 3)
                        result = np.zeros_like(batch_rgb)
                        
                        channel_map = {'R': 0, 'G': 1, 'B': 2}
                        
                        for channel_name, channel_idx in channel_map.items():
                            channel_curves = curves.get(channel_name, [(0.0, 0.0), (1.0, 1.0)])
                            
                            if channel_curves == [(0.0, 0.0), (1.0, 1.0)]:
                                result[:, channel_idx] = batch_rgb[:, channel_idx]
                                continue
                            
                            # 线性 -> 密度空间
                            safe_input = np.maximum(batch_rgb[:, channel_idx], 1e-10)
                            density = -np.log10(safe_input)
                            
                            # 归一化密度
                            LOG65536 = np.log10(65536.0)
                            normalized_density = 1.0 - np.clip(density / LOG65536, 0.0, 1.0)
                            
                            # 应用曲线
                            curve_points = np.array(channel_curves, dtype=np.float32)
                            x_points = curve_points[:, 0]
                            y_points = curve_points[:, 1]
                            curve_output = np.interp(normalized_density, x_points, y_points)
                            
                            # 转换回线性
                            output_density = (1.0 - curve_output) * LOG65536
                            result[:, channel_idx] = np.power(10.0, -output_density)
                        
                        return np.clip(result, 0.0, 1.0)
                    except Exception:
                        return rgb
                return density_curve_transform
            
            curve_transform = create_density_curve_transform()
            lut_chain_colors = curve_transform(step2_colors)
            
            # 计算误差统计
            abs_error = np.abs(lut_chain_colors - divere_colors)
            rel_error = abs_error / (divere_colors + 1e-10)
            
            stats = {
                'max_abs_error': float(np.max(abs_error)),
                'mean_abs_error': float(np.mean(abs_error)),
                'max_rel_error': float(np.max(rel_error)),
                'mean_rel_error': float(np.mean(rel_error)),
                'rmse': float(np.sqrt(np.mean(abs_error ** 2))),
                'samples_tested': test_samples
            }
            
            return stats
            
        except Exception as e:
            print(f"LUT链验证失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
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
                    fake_image, params, include_curve=include_curve,
                    convert_to_monochrome_in_idt=False, monochrome_converter=None
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