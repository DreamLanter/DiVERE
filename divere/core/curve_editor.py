"""
曲线编辑模块
密度打印曲线编辑器
"""

import json
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import sys

from .data_types import Curve


class CurveEditor:
    """曲线编辑器"""
    
    def __init__(self):
        self._preset_curves = {}
        self._load_preset_curves()
    
    def _load_preset_curves(self):
        """从文件系统加载预设曲线（支持用户配置优先）"""
        self._preset_curves = {}
        
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            
            # 获取所有配置文件（用户配置优先）
            config_files = enhanced_config_manager.get_config_files("curves")
            
            for curve_path in config_files:
                try:
                    curve_data = enhanced_config_manager.load_config_file(curve_path)
                    if curve_data is None:
                        continue
                    
                    name = curve_path.stem
                    self._preset_curves[name] = curve_data
                    
                    # 标记是否为用户配置
                    if curve_path.parent == enhanced_config_manager.user_curves_dir:
                        print(f"加载用户曲线: {name}")
                    else:
                        print(f"加载内置曲线: {name}")
                        
                except Exception as e:
                    print(f"加载曲线文件 {curve_path} 时出错: {e}")
                    
        except ImportError:
            # 如果增强配置管理器不可用，使用原来的方法
            if getattr(sys, 'frozen', False):
                # 编译后环境
                if sys.platform == 'darwin':
                    # macOS .app 包
                    curves_dir = Path(sys.executable).parent / "config" / "curves"
                else:
                    # 其他平台
                    curves_dir = Path(sys.executable).parent / "config" / "curves"
            else:
                # 开发环境
                curves_dir = Path("config/curves")
                
            if not curves_dir.exists():
                return
                
            for curve_path in curves_dir.glob("*.json"):
                try:
                    with open(curve_path, 'r', encoding='utf-8') as f:
                        curve_data = json.load(f)
                        name = curve_path.stem
                        self._preset_curves[name] = curve_data
                except Exception as e:
                    print(f"加载曲线文件 {curve_path} 时出错: {e}")
    
    def create_curve(self, points: List[Tuple[float, float]]) -> Curve:
        """创建曲线"""
        curve = Curve()
        for x, y in points:
            curve.add_point(x, y)
        return curve
    
    def create_preset_curve(self, preset_name: str) -> Optional[Curve]:
        """创建预设曲线"""
        if preset_name in self._preset_curves:
            preset_data = self._preset_curves[preset_name]
            return self.create_curve(preset_data["points"])
        return None
    
    def interpolate_curve(self, curve: Curve, method: str = "linear", num_points: int = 256) -> np.ndarray:
        """插值曲线"""
        if len(curve.points) < 2:
            return np.linspace(0, 1, num_points)
        
        x_coords = [p[0] for p in curve.points]
        y_coords = [p[1] for p in curve.points]
        
        if method == "linear":
            # 线性插值
            curve_x = np.linspace(0, 1, num_points)
            curve_y = np.interp(curve_x, x_coords, y_coords)
            
        elif method == "cubic":
            # 三次样条插值
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(x_coords, y_coords, bc_type='natural')
                curve_x = np.linspace(0, 1, num_points)
                curve_y = cs(curve_x)
            except ImportError:
                # 如果没有scipy，回退到线性插值
                curve_x = np.linspace(0, 1, num_points)
                curve_y = np.interp(curve_x, x_coords, y_coords)
                
        elif method == "bezier":
            # 贝塞尔曲线插值（简化版本）
            curve_x = np.linspace(0, 1, num_points)
            curve_y = self._bezier_interpolation(x_coords, y_coords, curve_x)
            
        else:
            # 默认线性插值
            curve_x = np.linspace(0, 1, num_points)
            curve_y = np.interp(curve_x, x_coords, y_coords)
        
        return np.column_stack([curve_x, curve_y])
    
    def _bezier_interpolation(self, x_points: List[float], y_points: List[float], t_values: np.ndarray) -> np.ndarray:
        """贝塞尔曲线插值"""
        if len(x_points) < 2:
            return np.zeros_like(t_values)
        
        # 简化的贝塞尔曲线实现
        # 对于多点贝塞尔曲线，使用德卡斯特罗算法
        n = len(x_points) - 1
        result = np.zeros_like(t_values)
        
        for i, t in enumerate(t_values):
            # 计算贝塞尔曲线上的点
            x = 0
            y = 0
            for j in range(n + 1):
                # 伯恩斯坦多项式
                coef = self._binomial(n, j) * (t ** j) * ((1 - t) ** (n - j))
                x += coef * x_points[j]
                y += coef * y_points[j]
            result[i] = y
        
        return result
    
    def _binomial(self, n: int, k: int) -> int:
        """计算二项式系数"""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        
        # 使用帕斯卡三角形
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def apply_curve_to_image(self, image: np.ndarray, curve: Curve) -> np.ndarray:
        """将曲线应用到图像"""
        if len(curve.points) < 2:
            return image
        
        # 获取插值后的曲线数据
        curve_data = self.interpolate_curve(curve)
        curve_x = curve_data[:, 0]
        curve_y = curve_data[:, 1]
        
        result = np.zeros_like(image)
        
        if len(image.shape) == 3:
            # 彩色图像
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                # 将像素值映射到曲线
                indices = np.clip(channel * (len(curve_x) - 1), 0, len(curve_x) - 1).astype(int)
                result[:, :, c] = curve_y[indices]
        else:
            # 灰度图像
            indices = np.clip(image * (len(curve_x) - 1), 0, len(curve_x) - 1).astype(int)
            result = curve_y[indices]
        
        return result
    
    def save_curve(self, curve: Curve, path: str) -> bool:
        """保存曲线"""
        try:
            curve_data = {
                "name": Path(path).stem,
                "description": f"自定义曲线: {Path(path).stem}",
                "points": curve.points,
                "interpolation_method": curve.interpolation_method
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(curve_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存曲线失败: {e}")
            return False
    
    def load_curve(self, path: str) -> Optional[Curve]:
        """加载曲线"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                curve_data = json.load(f)
            
            curve = Curve()
            curve.points = curve_data.get("points", [])
            curve.interpolation_method = curve_data.get("interpolation_method", "linear")
            
            return curve
        except Exception as e:
            print(f"加载曲线失败: {e}")
            return None
    
    def get_preset_curves(self) -> List[str]:
        """获取预设曲线列表"""
        return list(self._preset_curves.keys())
    
    def get_curve_info(self, curve: Curve) -> dict:
        """获取曲线信息"""
        return {
            "num_points": len(curve.points),
            "interpolation_method": curve.interpolation_method,
            "points": curve.points,
            "range": {
                "x_min": min(p[0] for p in curve.points) if curve.points else 0,
                "x_max": max(p[0] for p in curve.points) if curve.points else 1,
                "y_min": min(p[1] for p in curve.points) if curve.points else 0,
                "y_max": max(p[1] for p in curve.points) if curve.points else 1
            }
        }
    
    def create_film_response_curve(self, film_type: str = "color_negative") -> Curve:
        """创建胶片响应曲线"""
        if film_type == "color_negative":
            # 彩色负片特征曲线
            points = [
                (0.0, 0.0),
                (0.1, 0.05),
                (0.2, 0.15),
                (0.4, 0.35),
                (0.6, 0.65),
                (0.8, 0.85),
                (0.9, 0.95),
                (1.0, 1.0)
            ]
        elif film_type == "color_positive":
            # 彩色正片特征曲线
            points = [
                (0.0, 0.0),
                (0.2, 0.1),
                (0.4, 0.25),
                (0.6, 0.5),
                (0.8, 0.75),
                (1.0, 1.0)
            ]
        else:
            # 默认线性曲线
            points = [(0.0, 0.0), (1.0, 1.0)]
        
        return self.create_curve(points)
    
    def create_contrast_curve(self, contrast: float) -> Curve:
        """创建对比度调整曲线"""
        # contrast > 1 增加对比度，< 1 减少对比度
        if contrast == 1.0:
            return self.create_curve([(0.0, 0.0), (1.0, 1.0)])
        
        # 创建S型曲线来调整对比度
        if contrast > 1.0:
            # 增加对比度
            mid_point = 0.5
            offset = (contrast - 1.0) * 0.2
            points = [
                (0.0, 0.0),
                (0.25, mid_point - offset),
                (0.75, mid_point + offset),
                (1.0, 1.0)
            ]
        else:
            # 减少对比度
            mid_point = 0.5
            offset = (1.0 - contrast) * 0.2
            points = [
                (0.0, 0.0),
                (0.25, mid_point + offset),
                (0.75, mid_point - offset),
                (1.0, 1.0)
            ]
        
        return self.create_curve(points) 