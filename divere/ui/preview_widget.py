"""
预览组件
用于显示图像预览
"""

import numpy as np
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton, QCheckBox
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QCursor, QPolygonF

from divere.core.data_types import ImageData


class PreviewCanvas(QLabel):
    """自绘制画布：基于 pan/zoom 稳定绘制，不改变自身尺寸"""
    def __init__(self):
        super().__init__()
        self._source_pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # 简单缩放缓存以提升性能
        self._scaled_pixmap: Optional[QPixmap] = None
        self._scaled_zoom: float = 1.0
        # 叠加绘制回调
        self.overlay_drawer = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        # 清空文本避免覆盖
        self.setText("")
        # 源变更需重建缩放缓存
        self._scaled_pixmap = None
        self._scaled_zoom = 1.0
        self.update()

    def set_view(self, zoom: float, pan_x: float, pan_y: float) -> None:
        self._zoom = float(zoom)
        self._pan_x = float(pan_x)
        self._pan_y = float(pan_y)
        self.update()

    def _ensure_scaled_cache(self) -> None:
        if self._source_pixmap is None:
            self._scaled_pixmap = None
            return
        # 仅在缩放不为1时缓存
        if abs(self._zoom - 1.0) < 1e-6:
            self._scaled_pixmap = None
            self._scaled_zoom = 1.0
            return
        # 使用不取整的绘制缩放路径，暂不生成缓存，避免滚轮缩放时跳动
        self._scaled_pixmap = None
        return
        # 超大尺寸时放弃缓存，避免内存/卡顿（直接使用绘制时缩放）
        if target_w <= 0 or target_h <= 0:
            self._scaled_pixmap = None
            return
        if target_w > 4096 or target_h > 4096 or target_w * target_h > 16_000_000:
            self._scaled_pixmap = None
            return
        if self._scaled_pixmap is None or self._scaled_zoom != self._zoom:
            self._scaled_pixmap = self._source_pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled_zoom = self._zoom

    def paintEvent(self, event):
        # 先让 QLabel 按样式绘制背景
        super().paintEvent(event)
        if self._source_pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.translate(QPointF(self._pan_x, self._pan_y))
        # 为避免滚轮缩放时因缓存缩放尺寸取整导致的锚点漂移，这里统一走绘制时缩放路径
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._source_pixmap)
        if self.overlay_drawer is not None:
            try:
                self.overlay_drawer(painter)
            except Exception as e:
                print(f"overlay绘制失败: {e}")
        painter.end()


class PreviewWidget(QWidget):
    """图像预览组件"""
    
    # 发送图像旋转信号，参数为旋转方向：1=左旋，-1=右旋
    image_rotated = Signal(int)
    
    def __init__(self):
        super().__init__()
        
        self.current_image: Optional[ImageData] = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # 拖动相关状态
        self.dragging = False
        self.last_mouse_pos = None
        self.drag_start_pos = None
        self.original_pan_pos = None
        
        # 平滑拖动相关
        self.smooth_drag_timer = QTimer()
        self.smooth_drag_timer.timeout.connect(self._smooth_drag_update)
        self.smooth_drag_timer.setInterval(16)  # ~60fps
        
        self._create_ui()
        self._setup_mouse_controls()
        self._setup_keyboard_controls()

        # 视图边界（避免留白）的开关（按需限制）。
        # 根据需求改为默认关闭：允许自由拖放，不做边界约束。
        self._enable_pan_clamp = False

        # 缩放上下限
        self._min_zoom: float = 0.05
        self._max_zoom: float = 16.0

        # 旋转锚点状态（用于保持以预览中心为轴旋转）
        self._rotate_anchor_active: bool = False
        self._rotate_direction: int = 0  # 1=左旋, -1=右旋
        self._rotate_anchor_p = None  # (px, py) 旋转前图像坐标
        self._rotate_old_wh = None  # (W_old, H_old)

        # 色卡选择器状态
        self.cc_enabled: bool = False
        self.cc_corners = None  # [(x,y) * 4]
        self.cc_drag_idx: int | None = None
        self.cc_handle_radius_disp: float = 8.0
        self.cc_ref_qcolors = None  # 24项，参考色（QColor）

    # ============ 内部工具 ============
    def _get_viewport_size(self):
        """获取可视区域尺寸（viewport 尺寸）"""
        if hasattr(self, 'scroll_area') and self.scroll_area is not None:
            return self.scroll_area.viewport().size()
        return self.size()

    def _get_scaled_image_size(self):
        """返回当前缩放下图像尺寸 (Wi, Hi)"""
        if not self.current_image or self.current_image.array is None:
            return 0, 0
        h, w = self.current_image.array.shape[:2]
        Wi = int(round(w * float(self.zoom_factor)))
        Hi = int(round(h * float(self.zoom_factor)))
        return Wi, Hi

    def _clamp_pan(self):
        """根据 viewport 与缩放后图像尺寸对 pan 进行边界约束"""
        if not self._enable_pan_clamp:
            return
        Wv = self._get_viewport_size().width()
        Hv = self._get_viewport_size().height()
        Wi, Hi = self._get_scaled_image_size()

        # 当图像大于视口：允许范围 [Wv - Wi, 0]
        # 当图像小于视口：允许范围 [0, Wv - Wi]，以便可实现居中
        if Wi >= Wv:
            min_tx, max_tx = Wv - Wi, 0
        else:
            min_tx, max_tx = 0, Wv - Wi
        if Hi >= Hv:
            min_ty, max_ty = Hv - Hi, 0
        else:
            min_ty, max_ty = 0, Hv - Hi

        if self.pan_x < min_tx:
            self.pan_x = int(min_tx)
        elif self.pan_x > max_tx:
            self.pan_x = int(max_tx)
        if self.pan_y < min_ty:
            self.pan_y = int(min_ty)
        elif self.pan_y > max_ty:
            self.pan_y = int(max_ty)
    
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        # 旋转按钮
        self.rotate_left_btn = QPushButton("← 左旋")
        self.rotate_right_btn = QPushButton("右旋 →")
        self.rotate_left_btn.setMaximumWidth(80)
        self.rotate_right_btn.setMaximumWidth(80)
        self.rotate_left_btn.clicked.connect(self.rotate_left)
        self.rotate_right_btn.clicked.connect(self.rotate_right)
        
        # 视图控制按钮
        self.fit_window_btn = QPushButton("适应窗口")
        self.center_btn = QPushButton("居中")
        self.fit_window_btn.setMaximumWidth(80)
        self.center_btn.setMaximumWidth(80)
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        self.center_btn.clicked.connect(self.center_image)
        # 色卡选择器（默认隐藏，由参数面板联动显示/控制）
        self.cc_checkbox = QCheckBox("色卡选择器")
        self.cc_checkbox.toggled.connect(self._on_cc_toggled)
        self.cc_checkbox.setVisible(False)
        
        # 添加按钮到布局
        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addWidget(self.fit_window_btn)
        button_layout.addWidget(self.center_btn)
        button_layout.addWidget(self.cc_checkbox)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.image_label = PreviewCanvas()
        self.image_label.setObjectName('imageCanvas')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setText("请加载图像")
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    # ===== 色卡选择器：UI与绘制 =====
    def _on_cc_toggled(self, checked: bool):
        self.cc_enabled = bool(checked)
        if self.cc_enabled and self.current_image and self.current_image.array is not None:
            if not self.cc_corners:
                self._init_default_colorchecker()
            self._ensure_cc_reference_colors()
            self.image_label.overlay_drawer = self._draw_colorchecker_overlay
        else:
            self.image_label.overlay_drawer = None
        self._update_display()

    def _init_default_colorchecker(self):
        h, w = self.current_image.array.shape[:2]
        max_w = w * 0.6
        max_h = h * 0.6
        target_w = min(max_w, max_h * 1.5)
        target_h = target_w / 1.5
        cx = w / 2.0; cy = h / 2.0
        x0 = cx - target_w / 2.0; y0 = cy - target_h / 2.0
        x1 = cx + target_w / 2.0; y1 = cy + target_h / 2.0
        self.cc_corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _draw_colorchecker_overlay(self, painter: QPainter):
        if not (self.cc_enabled and self.cc_corners):
            return
        pts = [QPointF(*p) for p in self.cc_corners]
        painter.setPen(QPen(QColor(255, 255, 0, 220), 2))
        painter.setBrush(Qt.NoBrush)
        for i in range(4):
            painter.drawLine(pts[i], pts[(i+1)%4])
        painter.setBrush(QColor(255,255,0,200))
        for p in pts:
            painter.drawEllipse(p, 5, 5)

        # 绘制网格与采样框
        import numpy as np
        import cv2
        src = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
        dst = np.array(self.cc_corners, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        def map_rect(x0,y0,x1,y1):
            R = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
            R_h = np.hstack([R, np.ones((4,1), dtype=np.float32)])
            P = (H @ R_h.T).T
            P = P[:,:2] / P[:,2:3]
            return [QPointF(float(px), float(py)) for px,py in P]

        margin = 0.18  # 采样区域内缩比例
        ref_margin = 0.33  # 参考色小块的内缩比例（相对整个网格单元）
        for r in range(4):
            for c in range(6):
                gx0 = c/6.0; gx1=(c+1)/6.0
                gy0 = r/4.0; gy1=(r+1)/4.0
                poly = map_rect(gx0,gy0,gx1,gy1)
                painter.setPen(QPen(QColor(255,255,255,120),1))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(QPolygonF(poly))
                sx0 = gx0 + margin*(gx1-gx0)
                sx1 = gx1 - margin*(gx1-gx0)
                sy0 = gy0 + margin*(gy1-gy0)
                sy1 = gy1 - margin*(gy1-gy0)
                s_poly = map_rect(sx0,sy0,sx1,sy1)
                painter.setPen(QPen(QColor(0,255,0,200),1))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(QPolygonF(s_poly))
                # 参考色小块（填充，无描边）
                rx0 = gx0 + ref_margin*(gx1-gx0)
                rx1 = gx1 - ref_margin*(gx1-gx0)
                ry0 = gy0 + ref_margin*(gy1-gy0)
                ry1 = gy1 - ref_margin*(gy1-gy0)
                r_poly = map_rect(rx0, ry0, rx1, ry1)
                if self.cc_ref_qcolors is not None:
                    idx = r*6 + c
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(self.cc_ref_qcolors[idx])
                    painter.drawPolygon(QPolygonF(r_poly))

    def _ensure_cc_reference_colors(self):
        """生成24个参考颜色（QColor，按A1..D6），以 Display P3 显示。"""
        try:
            from divere.core.color_science import colorchecker_display_p3_qcolors
            self.cc_ref_qcolors = colorchecker_display_p3_qcolors()
        except Exception as e:
            print(f"生成参考颜色失败: {e}")
            self.cc_ref_qcolors = None
    
    def _setup_mouse_controls(self):
        """设置鼠标控制"""
        # 将鼠标事件绑定到image_label而不是scroll_area
        self.image_label.wheelEvent = self._wheel_event
        self.image_label.mousePressEvent = self._mouse_press_event
        self.image_label.mouseMoveEvent = self._mouse_move_event
        self.image_label.mouseReleaseEvent = self._mouse_release_event
        
        # 设置鼠标样式
        self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
    def _setup_keyboard_controls(self):
        """设置键盘控制"""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def set_image(self, image_data: ImageData):
        """设置显示的图像"""
        self.current_image = image_data
        self._update_display()
        # 如果存在旋转锚点，应用
        if self._rotate_anchor_active:
            self.apply_rotate_anchor()

    def get_current_image_data(self) -> Optional[ImageData]:
        """返回当前显示的ImageData对象"""
        return self.current_image
    
    def _update_display(self):
        """更新显示"""
        if not self.current_image or self.current_image.array is None:
            self.image_label.setText("请加载图像")
            return
        
        try:
            # 设置源图与视图参数，自绘中按 pan/zoom 绘制
            pixmap = self._array_to_pixmap(self.current_image.array)
            self._clamp_pan()
            self.image_label.set_source_pixmap(pixmap)
            self.image_label.set_view(self.zoom_factor, self.pan_x, self.pan_y)
            
        except Exception as e:
            print(f"更新显示失败: {e}")
            self.image_label.setText(f"显示错误: {str(e)}")
    
    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """将numpy数组转换为QPixmap"""
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        
        # 确保数组是连续的内存布局（旋转后可能不连续）
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        
        height, width = array.shape[:2]
        
        if len(array.shape) == 3:
            channels = array.shape[2]
            if channels == 3:
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
            elif channels == 4:
                qimage = QImage(array.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
            else: # Fallback for other channel counts
                array = array[:, :, :3]
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
        
        # 为DisplayP3图像应用Qt内置色彩管理
        if hasattr(self, 'current_image') and self.current_image and self.current_image.color_space == "Rec2020":
            from PySide6.QtGui import QColorSpace
            # 创建色彩空间（DisplayP3）
            displayp3_space = QColorSpace(QColorSpace.NamedColorSpace.DisplayP3)
            # 应用色彩空间到QImage
            qimage.setColorSpace(displayp3_space)
        
        return QPixmap.fromImage(qimage)
    
    def _wheel_event(self, event):
        """鼠标滚轮事件 - 缩放"""
        if not self.current_image: 
            event.accept()
            return
            
        delta = event.angleDelta().y()
        
        # 获取鼠标在 label 坐标系中的位置（使用浮点坐标保证精度）
        m = event.position()

        # 计算围绕鼠标的缩放：保持鼠标下像素不动
        zoom_factor_change = 1.05 if delta > 0 else 1/1.05
        old_zoom = float(self.zoom_factor)
        new_zoom = float(self.zoom_factor * zoom_factor_change)
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        # 图像坐标 p（以当前 pan, zoom 映射）
        # p = (m - t) / s
        p_x = (float(m.x()) - float(self.pan_x)) / old_zoom
        p_y = (float(m.y()) - float(self.pan_y)) / old_zoom

        # 新平移 t' = m - p * s'
        new_pan_x = float(m.x()) - p_x * new_zoom
        new_pan_y = float(m.y()) - p_y * new_zoom

        self.zoom_factor = new_zoom
        # 使用浮点 pan，避免取整引入的锚点跳变
        self.pan_x = new_pan_x
        self.pan_y = new_pan_y

        # 边界约束（基于新 zoom）
        self._clamp_pan()

        # 更新显示
        self._update_display()
        
        event.accept()
    
    def _mouse_press_event(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 色卡角点命中测试（优先）
            if self.cc_enabled and self.cc_corners:
                m = event.position()
                for idx, (ix, iy) in enumerate(self.cc_corners):
                    dx = (ix * float(self.zoom_factor) + float(self.pan_x)) - float(m.x())
                    dy = (iy * float(self.zoom_factor) + float(self.pan_y)) - float(m.y())
                    if (dx*dx + dy*dy) ** 0.5 <= self.cc_handle_radius_disp:
                        self.cc_drag_idx = idx
                        event.accept(); return
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.drag_start_pos = event.pos()
            
            # 记录开始拖动时的平移位置（使用浮点坐标）
            self.original_pan_pos = QPointF(float(self.pan_x), float(self.pan_y))
            
            # 改变鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            
        event.accept()
    
    def _mouse_move_event(self, event):
        """鼠标移动事件"""
        if self.cc_enabled and self.cc_drag_idx is not None:
            m = event.position()
            ix = (float(m.x()) - float(self.pan_x)) / float(self.zoom_factor)
            iy = (float(m.y()) - float(self.pan_y)) / float(self.zoom_factor)
            self.cc_corners[self.cc_drag_idx] = (ix, iy)
            self._update_display(); event.accept(); return
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            
            # 直接更新平移偏移
            # 使用浮点累加，避免取整抖动
            self.pan_x += float(delta.x())
            self.pan_y += float(delta.y())

            # 边界约束（M3）
            self._clamp_pan()
            
            # 更新显示
            self._update_display()
            
            self.last_mouse_pos = event.pos()
            
        event.accept()
    
    def _mouse_release_event(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.cc_drag_idx = None
            self.dragging = False
            self.last_mouse_pos = None
            self.drag_start_pos = None
            self.original_pan_pos = None
            
            # 恢复鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            
        event.accept()
    
    def _smooth_drag_update(self):
        """平滑拖动更新（预留功能）"""
        # 这里可以实现更平滑的拖动效果
        pass
    
    def reset_view(self):
        """重置视图"""
        self.zoom_factor = 1.0
        self.center_image()

    def fit_to_window(self):
        """适应窗口大小"""
        if not self.current_image: 
            return
            
        # 使用viewport的可用区域尺寸，避免滚动条/边距带来的误差
        widget_size = self.scroll_area.viewport().size()
        image_size = self.current_image.array.shape[:2][::-1]
        scale_x = widget_size.width() / image_size[0]
        scale_y = widget_size.height() / image_size[1]
        self.zoom_factor = min(scale_x, scale_y, 1.0)

        # 居中平移
        Wi, Hi = self._get_scaled_image_size()
        Wv, Hv = widget_size.width(), widget_size.height()
        self.pan_x = int(round((Wv - Wi) / 2))
        self.pan_y = int(round((Hv - Hi) / 2))
        self._clamp_pan()

        self._update_display()
    
    def center_image(self):
        """居中显示图像"""
        if not self.current_image:
            return

        widget_size = self._get_viewport_size()
        Wi, Hi = self._get_scaled_image_size()
        Wv, Hv = widget_size.width(), widget_size.height()
        self.pan_x = int(round((Wv - Wi) / 2))
        self.pan_y = int(round((Hv - Hi) / 2))
        self._clamp_pan()

        self._update_display()

    # ===== 色卡选择器：读取24色块平均色（XYZ） =====
    def read_colorchecker_xyz(self):
        """读取当前色卡选择器24块的平均颜色（XYZ），返回 ndarray (24, 3)。行序为4行x6列。"""
        if not (self.cc_enabled and self.cc_corners and self.current_image and self.current_image.array is not None):
            return None
        import numpy as np
        import cv2
        from colour import RGB_COLOURSPACES, RGB_to_XYZ
        arr = self.current_image.array
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        H_img, W_img = arr.shape[:2]
        src = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
        dst = np.array(self.cc_corners, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        margin = 0.18
        xyz_list = []
        for r in range(4):
            for c in range(6):
                gx0 = c/6.0; gx1=(c+1)/6.0
                gy0 = r/4.0; gy1=(r+1)/4.0
                sx0 = gx0 + margin*(gx1-gx0)
                sx1 = gx1 - margin*(gx1-gx0)
                sy0 = gy0 + margin*(gy1-gy0)
                sy1 = gy1 - margin*(gy1-gy0)
                rect = np.array([[sx0,sy0],[sx1,sy0],[sx1,sy1],[sx0,sy1]], dtype=np.float32)
                rect_h = np.hstack([rect, np.ones((4,1), dtype=np.float32)])
                poly = (H @ rect_h.T).T
                poly = poly[:,:2] / poly[:,2:3]
                poly_int = np.round(poly).astype(np.int32)
                mask = np.zeros((H_img, W_img), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_int], 255)
                m = mask.astype(bool)
                if not np.any(m):
                    rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    rgb = arr[m].reshape(-1, arr.shape[2]).mean(axis=0)
                # 假设预览为 DisplayP3
                try:
                    cs = RGB_COLOURSPACES.get('Display P3')
                    XYZ = RGB_to_XYZ(rgb, cs.whitepoint, cs.whitepoint, cs.matrix_RGB_to_XYZ)
                except Exception:
                    XYZ = rgb
                xyz_list.append(XYZ.astype(np.float32))
        return np.stack(xyz_list, axis=0)

    # ===== 旋转锚点支持 =====
    def prepare_rotate(self, direction: int):
        """在图像实际旋转前调用，捕获以预览中心为锚的图像坐标。"""
        if not self.current_image:
            return
        self._rotate_direction = int(direction)
        self._rotate_anchor_active = True
        # 视口中心
        Wv = self._get_viewport_size().width()
        Hv = self._get_viewport_size().height()
        m0x = float(Wv) / 2.0
        m0y = float(Hv) / 2.0
        s = float(self.zoom_factor)
        t_x = float(self.pan_x)
        t_y = float(self.pan_y)
        # 以当前视图参数反解中心对应的图像坐标 p
        p_x = (m0x - t_x) / s
        p_y = (m0y - t_y) / s
        self._rotate_anchor_p = (p_x, p_y)
        h, w = self.current_image.array.shape[:2]
        self._rotate_old_wh = (float(w), float(h))

    def apply_rotate_anchor(self):
        """在图像旋转加载完成后调用，调整平移使预览中心保持为旋转轴。"""
        if not (self._rotate_anchor_active and self.current_image and self.current_image.array is not None):
            # 清理状态
            self._rotate_anchor_active = False
            self._rotate_direction = 0
            self._rotate_anchor_p = None
            self._rotate_old_wh = None
            return
        try:
            W_old, H_old = self._rotate_old_wh
            px, py = self._rotate_anchor_p
            direction = self._rotate_direction
            # 旋转后的对应坐标 p'
            if direction == 1:  # 左旋 90° CCW
                ppx = py
                ppy = W_old - 1.0 - px
            elif direction == -1:  # 右旋 90° CW
                ppx = H_old - 1.0 - py
                ppy = px
            else:
                ppx, ppy = px, py
            # 以当前视口中心求新的平移
            Wv = self._get_viewport_size().width()
            Hv = self._get_viewport_size().height()
            m0x = float(Wv) / 2.0
            m0y = float(Hv) / 2.0
            s = float(self.zoom_factor)
            self.pan_x = m0x - ppx * s
            self.pan_y = m0y - ppy * s
            self._clamp_pan()
            self._update_display()
        finally:
            # 清理状态
            self._rotate_anchor_active = False
            self._rotate_direction = 0
            self._rotate_anchor_p = None
            self._rotate_old_wh = None
        
    def rotate_left(self):
        """左旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 发送信号通知主窗口执行旋转
            self.image_rotated.emit(1)  # 1表示左旋
            
    def rotate_right(self):
        """右旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 发送信号通知主窗口执行旋转
            self.image_rotated.emit(-1)  # -1表示右旋
            
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key.Key_Left:
            self.rotate_left()
        elif event.key() == Qt.Key.Key_Right:
            self.rotate_right()
        elif event.key() == Qt.Key.Key_0:  # 数字0键重置视图
            self.reset_view()
        elif event.key() == Qt.Key.Key_F:  # F键适应窗口
            self.fit_to_window()
        elif event.key() == Qt.Key.Key_C:  # C键居中
            self.center_image()
        else:
            super().keyPressEvent(event)
