"""
可视化曲线编辑器组件
支持拖拽编辑控制点，实时预览曲线效果
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QGroupBox, QDoubleSpinBox,
                            QGridLayout, QSizePolicy, QFileDialog, QInputDialog, QMessageBox)
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..core.data_types import Curve


class CurveEditWidget(QWidget):
    """曲线编辑画布"""
    
    curve_changed = Signal(list)  # 当曲线改变时发出信号，传递控制点列表
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 曲线控制点 [(x, y), ...]
        # 支持多通道曲线：RGB, R, G, B
        self.curves = {
            'RGB': [(0.0, 0.0), (1.0, 1.0)],
            'R': [(0.0, 0.0), (1.0, 1.0)],
            'G': [(0.0, 0.0), (1.0, 1.0)],
            'B': [(0.0, 0.0), (1.0, 1.0)]
        }
        self.current_channel = 'RGB'  # 当前编辑的通道
        self.control_points = self.curves[self.current_channel]  # 向后兼容
        self.selected_point = -1
        self.dragging = False
        
        # 样式设置 - 更细致、低调的外观
        self.grid_color = QColor(220, 220, 220)
        self.curve_color = QColor(80, 80, 80)  # 深灰色，低调
        self.point_color = QColor(100, 100, 100)  # 灰色控制点
        self.selected_point_color = QColor(60, 60, 60)  # 深灰色选中点
        self.background_color = QColor(250, 250, 250)  # 更浅的背景
        
        # 通道曲线颜色
        self.channel_colors = {
            'RGB': QColor(80, 80, 80),      # 深灰色
            'R': QColor(200, 80, 80, 128),   # 半透明红色
            'G': QColor(80, 200, 80, 128),   # 半透明绿色
            'B': QColor(80, 80, 200, 128)    # 半透明蓝色
        }
        
        self.dmax = 4.0  # 默认最大密度值
        self.gamma = 1.0 # 默认gamma值
        
        # 曲线分辨率
        self.curve_resolution = 256
        
        self.setMouseTracking(True)
    
    def set_dmax(self, dmax: float):
        """设置最大密度值用于坐标轴显示"""
        self.dmax = dmax
        self.update()

    def set_gamma(self, gamma: float):
        """设置gamma值用于坐标轴显示"""
        self.gamma = gamma
        self.update()
    
    def set_current_channel(self, channel: str):
        """设置当前编辑的通道"""
        if channel in self.curves:
            self.current_channel = channel
            self.control_points = self.curves[channel]
            self.selected_point = -1
            self.update()
            self.curve_changed.emit(self.control_points)
    
    def set_curve_points(self, points: List[Tuple[float, float]], channel: str = None, emit_signal: bool = True):
        """设置曲线控制点"""
        if channel is None:
            channel = self.current_channel
        
        if channel in self.curves:
            # 确保所有点都是tuple格式，并按x坐标排序
            normalized_points = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    normalized_points.append((float(point[0]), float(point[1])))
            
            self.curves[channel] = sorted(normalized_points, key=lambda p: p[0])
            if channel == self.current_channel:
                self.control_points = self.curves[channel]
                self.selected_point = -1
                self.update()
                if emit_signal:
                    self.curve_changed.emit(self.control_points)
    
    def get_curve_points(self, channel: str = None) -> List[Tuple[float, float]]:
        """获取曲线控制点"""
        if channel is None:
            channel = self.current_channel
        return self.curves.get(channel, [(0.0, 0.0), (1.0, 1.0)])
    
    def get_all_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """获取所有通道的曲线"""
        return self.curves.copy()
    
    def set_all_curves(self, curves: Dict[str, List[Tuple[float, float]]], emit_signal: bool = True):
        """设置所有通道的曲线"""
        for channel, points in curves.items():
            if channel in self.curves:
                # 确保所有点都是tuple格式，并按x坐标排序
                normalized_points = []
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        normalized_points.append((float(point[0]), float(point[1])))
                
                self.curves[channel] = sorted(normalized_points, key=lambda p: p[0])
        
        # 更新当前显示的曲线
        if self.current_channel in self.curves:
            self.control_points = self.curves[self.current_channel]
            self.selected_point = -1
            self.update()
            if emit_signal:
                self.curve_changed.emit(self.control_points)
    
    def add_point(self, x: float, y: float):
        """添加控制点"""
        # 确保坐标在[0,1]范围内
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        # 检查是否过于接近现有点
        min_distance = 0.05  # 最小距离阈值
        for i, (px, py) in enumerate(self.control_points):
            if abs(px - x) < min_distance:
                # 如果太接近现有点，就选中现有点而不是添加新点
                self.selected_point = i
                self.update()
                return
        
        # 插入到正确位置保持x坐标排序
        inserted = False
        for i, (px, py) in enumerate(self.control_points):
            if x < px:
                self.control_points.insert(i, (x, y))
                self.selected_point = i
                inserted = True
                break
        
        if not inserted:
            self.control_points.append((x, y))
            self.selected_point = len(self.control_points) - 1
        
        self.update()
        self.curve_changed.emit(self.control_points)
    
    def remove_selected_point(self):
        """删除选中的控制点"""
        if (self.selected_point >= 0 and 
            self.selected_point < len(self.control_points) and
            len(self.control_points) > 2):
            # 不允许删除第一个和最后一个点
            if 0 < self.selected_point < len(self.control_points) - 1:
                del self.control_points[self.selected_point]
                self.selected_point = -1
                self.update()
                self.curve_changed.emit(self.control_points)
    
    def _get_draw_rect(self) -> QRectF:
        """获取绘制区域的矩形（考虑边距）"""
        rect = self.rect()
        left_margin = 40
        top_margin = 20
        right_margin = 20
        bottom_margin = 50
        
        return QRectF(left_margin, top_margin, 
                     rect.width() - left_margin - right_margin, 
                     rect.height() - top_margin - bottom_margin)
    
    def _widget_to_curve_coords(self, widget_x: int, widget_y: int) -> Tuple[float, float]:
        """将组件坐标转换为曲线坐标(0-1)"""
        draw_rect = self._get_draw_rect()
        
        curve_x = (widget_x - draw_rect.left()) / draw_rect.width()
        curve_y = 1.0 - (widget_y - draw_rect.top()) / draw_rect.height()
        
        return max(0.0, min(1.0, curve_x)), max(0.0, min(1.0, curve_y))
    
    def _curve_to_widget_coords(self, curve_x: float, curve_y: float) -> Tuple[int, int]:
        """将曲线坐标转换为组件坐标"""
        draw_rect = self._get_draw_rect()
        
        widget_x = draw_rect.left() + curve_x * draw_rect.width()
        widget_y = draw_rect.top() + (1.0 - curve_y) * draw_rect.height()
        
        return int(widget_x), int(widget_y)
    
    def _find_point_near(self, x: int, y: int) -> int:
        """查找鼠标附近的控制点"""
        for i, (px, py) in enumerate(self.control_points):
            wx, wy = self._curve_to_widget_coords(px, py)
            if abs(wx - x) <= 8 and abs(wy - y) <= 8:
                return i
        return -1
    
    def _interpolate_curve(self, points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """插值生成平滑曲线 - 使用单调三次插值（类似Photoshop）"""
        if points is None:
            points = self.control_points
            
        if len(points) < 2:
            return points
        
        if len(points) == 2:
            # 只有两个点时使用线性插值
            curve_points = []
            for i in range(self.curve_resolution + 1):
                t = i / self.curve_resolution
                x = points[0][0] + t * (points[1][0] - points[0][0])
                y = points[0][1] + t * (points[1][1] - points[0][1])
                curve_points.append((x, y))
            return curve_points
        
        # 使用单调三次样条插值（更接近Photoshop的行为）
        curve_points = []
        
        # 按x坐标对控制点排序
        sorted_points = sorted(points, key=lambda p: p[0])
        
        # 生成插值点
        for i in range(self.curve_resolution + 1):
            x = i / self.curve_resolution
            y = self._monotonic_cubic_interpolate(x, sorted_points)
            curve_points.append((x, y))
        
        return curve_points
    
    def _monotonic_cubic_interpolate(self, x: float, points: List[Tuple[float, float]]) -> float:
        """单调三次插值（类似Photoshop的曲线插值）"""
        if not points:
            return x  # 默认线性
        
        # 找到x所在的区间
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            if x1 <= x <= x2:
                if x2 - x1 == 0:
                    return y1
                
                # 计算局部切线斜率（避免过冲）
                if i == 0:
                    # 第一段：使用当前段的斜率
                    m1 = (y2 - y1) / (x2 - x1)
                else:
                    # 中间段：使用相邻段的平均斜率
                    x0, y0 = points[i - 1]
                    m1 = ((y1 - y0) / (x1 - x0) + (y2 - y1) / (x2 - x1)) * 0.5
                
                if i == len(points) - 2:
                    # 最后一段：使用当前段的斜率
                    m2 = (y2 - y1) / (x2 - x1)
                else:
                    # 中间段：使用相邻段的平均斜率
                    x3, y3 = points[i + 2]
                    m2 = ((y2 - y1) / (x2 - x1) + (y3 - y2) / (x3 - x2)) * 0.5
                
                # Hermite插值（更平滑，类似Photoshop）
                t = (x - x1) / (x2 - x1)
                t2 = t * t
                t3 = t2 * t
                
                h00 = 2*t3 - 3*t2 + 1
                h10 = t3 - 2*t2 + t
                h01 = -2*t3 + 3*t2
                h11 = t3 - t2
                
                result = (h00 * y1 + h10 * (x2 - x1) * m1 + h01 * y2 + h11 * (x2 - x1) * m2)
                # 限制结果在[0,1]范围内
                return max(0.0, min(1.0, result))
        
        # 如果x在范围外，返回最近端点的y值
        if x <= points[0][0]:
            return points[0][1]
        else:
            return points[-1][1]
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = event.position().x(), event.position().y()
            point_index = self._find_point_near(x, y)
            
            if point_index >= 0:
                # 选中已存在的点
                self.selected_point = point_index
                self.dragging = True
            else:
                # 添加新点
                curve_x, curve_y = self._widget_to_curve_coords(x, y)
                self.add_point(curve_x, curve_y)
                self.dragging = True
            
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键删除点
            x, y = event.position().x(), event.position().y()
            point_index = self._find_point_near(x, y)
            if point_index >= 0:
                self.selected_point = point_index
                self.remove_selected_point()
                self.update()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point >= 0:
            x, y = event.position().x(), event.position().y()
            curve_x, curve_y = self._widget_to_curve_coords(x, y)
            
            # 限制Y坐标在[0,1]范围内
            curve_y = max(0.0, min(1.0, curve_y))
            
            # 允许编辑端部点，但限制X坐标在[0,1]范围内
            curve_x = max(0.0, min(1.0, curve_x))
            
            # 对于中间点，限制在相邻点之间（保持顺序）
            if 0 < self.selected_point < len(self.control_points) - 1:
                left_x = self.control_points[self.selected_point - 1][0]
                right_x = self.control_points[self.selected_point + 1][0]
                curve_x = max(left_x + 0.01, min(right_x - 0.01, curve_x))
            
            # 直接更新选中点，不重新排序
            self.control_points[self.selected_point] = (curve_x, curve_y)
            
            self.update()
            self.curve_changed.emit(self.control_points)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.remove_selected_point()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        draw_rect = self._get_draw_rect()
        
        # 绘制背景
        painter.fillRect(draw_rect, self.background_color)
        
        # 绘制网格（以密度为单位）
        painter.setPen(QPen(self.grid_color, 1))
        
        # 横向网格线（对应Y轴密度值）
        # Y轴从上到下：0.0到3.0密度，每0.3一格
        y_density_max = 3.0
        y_density_step = 0.3
        num_y_lines = int(y_density_max / y_density_step) + 1
        
        for i in range(num_y_lines):
            density = i * y_density_step
            # 将密度映射到[0,1]范围（0密度在顶部，3.0密度在底部）
            norm_y = density / y_density_max
            y = draw_rect.top() + norm_y * draw_rect.height()
            if y <= draw_rect.bottom():
                painter.drawLine(int(draw_rect.left()), int(y), 
                               int(draw_rect.right()), int(y))
        
        # 垂直网格线（对应X轴密度值）
        # X轴从左到右：0到log10(65536)≈4.816密度，每0.3一格
        x_density_max = np.log10(65536)  # ≈ 4.816
        x_density_step = 0.3
        num_x_lines = int(x_density_max / x_density_step) + 1
        
        for i in range(num_x_lines):
            density = i * x_density_step
            # 将密度映射到[0,1]范围
            norm_x = density / x_density_max
            x = draw_rect.left() + norm_x * draw_rect.width()
            if x <= draw_rect.right():
                painter.drawLine(int(x), int(draw_rect.top()), 
                               int(x), int(draw_rect.bottom()))
        
        # 绘制边框
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(draw_rect)
        
        # 绘制曲线
        # 如果在RGB模式，先绘制暗淡的R、G、B曲线
        if self.current_channel == 'RGB':
            for channel in ['R', 'G', 'B']:
                points = self.curves.get(channel, [])
                if len(points) >= 2:
                    curve_points = self._interpolate_curve(points)
                    if curve_points:
                        polygon = QPolygonF()
                        for x, y in curve_points:
                            wx, wy = self._curve_to_widget_coords(x, y)
                            polygon.append(QPointF(wx, wy))
                        
                        # 使用半透明的通道颜色
                        painter.setPen(QPen(self.channel_colors[channel], 1.0))
                        painter.drawPolyline(polygon)
        
        # 绘制当前通道的曲线（主曲线）
        if len(self.control_points) >= 2:
            curve_points = self._interpolate_curve(self.control_points)
            
            if curve_points:
                polygon = QPolygonF()
                for x, y in curve_points:
                    wx, wy = self._curve_to_widget_coords(x, y)
                    polygon.append(QPointF(wx, wy))
                
                # 使用当前通道的颜色，更粗一些
                color = self.channel_colors.get(self.current_channel, self.curve_color)
                if self.current_channel != 'RGB':
                    # 单通道模式下使用不透明的颜色
                    color = QColor(color.red(), color.green(), color.blue(), 255)
                painter.setPen(QPen(color, 2.0))  # 主曲线更粗
                painter.drawPolyline(polygon)
        
        # 绘制控制点
        for i, (x, y) in enumerate(self.control_points):
            wx, wy = self._curve_to_widget_coords(x, y)
            
            color = self.selected_point_color if i == self.selected_point else self.point_color
            painter.setPen(QPen(color, 1))  # 更细的边框
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(wx - 3), int(wy - 3), 6, 6)  # 更小的控制点
        
        # 绘制坐标标签
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # 绘制Y轴标签（密度值）
        y_density_max = 3.0
        y_density_step = 0.3
        
        for i in range(int(y_density_max / y_density_step) + 1):
            density = i * y_density_step
            if density <= y_density_max:
                # 将密度映射到曲线坐标（0密度在顶部=y_val=0，3.0密度在底部=y_val=1）
                y_val = density / y_density_max
                _wx, wy = self._curve_to_widget_coords(0, y_val)
                # 显示时Y轴是反向的（0在顶部）
                painter.drawText(5, wy + 4, f"{density:.1f}")

        # 绘制X轴标签（密度值）
        x_density_max = np.log10(65536)  # ≈ 4.816
        x_density_step = 0.3
        
        for i in range(int(x_density_max / x_density_step) + 1):
            density = i * x_density_step
            if density <= x_density_max:
                # 将密度映射到曲线坐标
                x_val = density / x_density_max
                wx, _wy = self._curve_to_widget_coords(x_val, 0)
                text_width = painter.fontMetrics().horizontalAdvance(f"{density:.1f}")
                painter.drawText(wx - text_width // 2, int(draw_rect.bottom()) + 15, f"{density:.1f}")
        
        # 绘制轴标题
        font.setPointSize(10)
        painter.setFont(font)
        
        # X轴标题
        x_title = "输入密度"
        x_title_width = painter.fontMetrics().horizontalAdvance(x_title)
        painter.drawText(int(draw_rect.center().x() - x_title_width // 2), 
                        int(draw_rect.bottom()) + 35, x_title)
        
        # Y轴标题（垂直绘制）
        painter.save()
        painter.translate(15, int(draw_rect.center().y()))
        painter.rotate(-90)
        y_title = "输出密度"
        y_title_width = painter.fontMetrics().horizontalAdvance(y_title)
        painter.drawText(-y_title_width // 2, -5, y_title)
        painter.restore()


class CurveEditorWidget(QWidget):
    """完整的曲线编辑器组件"""
    
    curve_changed = Signal(str, list)  # 曲线名称和控制点
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preset_curves = {}
        self.current_curve_name = "自定义"
        
        self._load_preset_curves()
        self._setup_ui()
        self._connect_signals()
    
    def _load_preset_curves(self):
        """从curves目录加载已保存的曲线"""
        self.preset_curves = {}
        curves_dir = Path("config/curves")
        
        if curves_dir.exists():
            for json_file in curves_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 支持新旧两种格式
                        if 'curves' in data and isinstance(data['curves'], dict):
                            # 新格式（多通道）
                            self.preset_curves[json_file.stem] = {
                                "name": data.get("name", json_file.stem),
                                "description": data.get("description", ""),
                                "curves": data["curves"]
                            }
                        elif 'points' in data:
                            # 旧格式（单曲线）- 转换为新格式
                            points = data["points"]
                            self.preset_curves[json_file.stem] = {
                                "name": data.get("name", json_file.stem),
                                "description": data.get("description", ""),
                                "curves": {
                                    "RGB": points,
                                    "R": [(0.0, 0.0), (1.0, 1.0)],
                                    "G": [(0.0, 0.0), (1.0, 1.0)],
                                    "B": [(0.0, 0.0), (1.0, 1.0)]
                                }
                            }
                except Exception as e:
                    print(f"加载曲线文件 {json_file} 时出错: {e}")
    
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 通道选择（放在曲线控件上方）
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("通道:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("RGB", "RGB")
        self.channel_combo.addItem("红", "R")
        self.channel_combo.addItem("绿", "G")
        self.channel_combo.addItem("蓝", "B")
        self.channel_combo.setMaximumWidth(150)
        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch()
        
        layout.addLayout(channel_layout)
        
        # 曲线编辑画布
        self.curve_edit_widget = CurveEditWidget()
        layout.addWidget(self.curve_edit_widget, 1)
        
        # 曲线控制（已保存曲线和操作按钮）
        control_layout = QHBoxLayout()
        
        # 已保存曲线选择
        control_layout.addWidget(QLabel("已保存曲线:"))
        self.curve_combo = QComboBox()
        self.curve_combo.addItem("自定义", "custom")
        for curve_key, curve_data in self.preset_curves.items():
            self.curve_combo.addItem(curve_data["name"], curve_key)
        self.curve_combo.setMaximumWidth(200)
        control_layout.addWidget(self.curve_combo)
        
        # 设置默认选择为"Kodak Endura Paper"
        kodak_index = self.curve_combo.findText("Kodak Endura Paper")
        if kodak_index >= 0:
            self.curve_combo.setCurrentIndex(kodak_index)
        
        control_layout.addStretch()
        
        # 操作按钮
        self.reset_button = QPushButton("重置为线性")
        self.save_button = QPushButton("保存曲线")
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.save_button)
        
        layout.addLayout(control_layout)
        
        # 使用说明
        help_label = QLabel(
            "使用说明：左键点击添加/选择控制点，拖拽移动点，右键删除点，Delete键删除选中点"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(help_label)
    
    def _connect_signals(self):
        """连接信号"""
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self.curve_combo.currentTextChanged.connect(self._on_preset_curve_changed)
        self.reset_button.clicked.connect(self._reset_to_linear)
        self.save_button.clicked.connect(self._save_curve)
        
        self.curve_edit_widget.curve_changed.connect(self._on_curve_changed)
    
    def _on_channel_changed(self):
        """通道选择改变"""
        channel = self.channel_combo.currentData()
        self.curve_edit_widget.set_current_channel(channel)
    
    def _on_preset_curve_changed(self):
        """预设曲线改变"""
        curve_key = self.curve_combo.currentData()
        
        if curve_key == "custom":
            self.current_curve_name = "自定义"
            return
        
        if curve_key in self.preset_curves:
            curve_data = self.preset_curves[curve_key]
            self.current_curve_name = curve_data["name"]
            
            # 加载所有通道的曲线，不触发信号避免跳转到"自定义"
            if "curves" in curve_data:
                self.curve_edit_widget.set_all_curves(curve_data["curves"], emit_signal=False)
            else:
                # 兼容旧格式
                points = curve_data.get("points", [(0.0, 0.0), (1.0, 1.0)])
                self.curve_edit_widget.set_curve_points(points, emit_signal=False)
            
            # 手动触发曲线改变信号，通知主窗口更新预览
            self.curve_changed.emit(self.current_curve_name, self.curve_edit_widget.get_curve_points())
    
    def _reset_to_linear(self):
        """重置为线性曲线"""
        linear_points = [(0.0, 0.0), (1.0, 1.0)]
        # 重置所有通道为线性曲线
        all_curves = {
            'RGB': linear_points,
            'R': linear_points,
            'G': linear_points,
            'B': linear_points
        }
        self.curve_edit_widget.set_all_curves(all_curves)
        self.curve_combo.setCurrentText("自定义")
        self.current_curve_name = "自定义"
    
    def _save_curve(self):
        """保存当前曲线到文件"""
        # 获取曲线名称
        name, ok = QInputDialog.getText(
            self, 
            "保存曲线", 
            "请输入曲线名称:",
            text=self.current_curve_name if self.current_curve_name != "自定义" else ""
        )
        
        if not ok or not name:
            return
        
        # 获取描述
        description, ok = QInputDialog.getText(
            self, 
            "保存曲线", 
            "请输入曲线描述（可选）:"
        )
        
        if not ok:
            return
        
        # 准备数据
        all_curves = self.curve_edit_widget.get_all_curves()
        curve_data = {
            "name": name,
            "description": description,
            "version": 2,
            "curves": all_curves
        }
        
        # 生成文件名（去除特殊字符）
        safe_filename = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存曲线文件",
            f"config/curves/{safe_filename}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                # 确保文件有.json扩展名
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(curve_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "成功", f"曲线已保存到：\n{file_path}")
                
                # 重新加载曲线列表
                self._load_preset_curves()
                self._refresh_curve_combo()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存曲线时出错：\n{str(e)}")
    
    def _refresh_curve_combo(self):
        """刷新曲线下拉列表"""
        # 保存当前选择
        current_data = self.curve_combo.currentData()
        
        # 清空并重新填充
        self.curve_combo.clear()
        self.curve_combo.addItem("自定义", "custom")
        for curve_key, curve_data in self.preset_curves.items():
            self.curve_combo.addItem(curve_data["name"], curve_key)
        
        # 恢复选择
        index = self.curve_combo.findData(current_data)
        if index >= 0:
            self.curve_combo.setCurrentIndex(index)
    
    def _on_curve_changed(self, points):
        """曲线改变时的处理"""
        # 当内部曲线变化时，自动将预设设置为"自定义"
        if self.curve_combo.currentData() != "custom":
            self.curve_combo.blockSignals(True)
            self.curve_combo.setCurrentText("自定义")
            self.curve_combo.blockSignals(False)
        self.current_curve_name = "自定义"

        # 发出曲线改变信号，包含名称和点
        self.curve_changed.emit(self.current_curve_name, points)
    
    def set_curve(self, points: List[Tuple[float, float]]):
        """设置曲线"""
        # 数据验证和保护
        if not isinstance(points, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in points):
            print(f"警告: set_curve 收到无效的数据格式: {points}，将重置为线性。")
            points = [(0.0, 0.0), (1.0, 1.0)]
        
        self.curve_edit_widget.set_curve_points(points)
    
    def get_curve_points(self) -> List[Tuple[float, float]]:
        """获取曲线控制点"""
        return self.curve_edit_widget.get_curve_points()
    
    def get_all_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """获取所有通道的曲线"""
        return self.curve_edit_widget.get_all_curves()
    
    def set_all_curves(self, curves: Dict[str, List[Tuple[float, float]]]):
        """设置所有通道的曲线"""
        for channel, points in curves.items():
            self.curve_edit_widget.set_curve_points(points, channel)