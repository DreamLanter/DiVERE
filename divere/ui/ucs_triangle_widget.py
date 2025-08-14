from __future__ import annotations

from typing import Dict, Tuple
from PySide6 import QtCore, QtGui, QtWidgets
from divere.core.color_space import uv_to_xy


class UcsTriangleWidget(QtWidgets.QWidget):
    coordinatesChanged = QtCore.Signal(dict)
    resetPointRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # 背景图：统一通过 app_paths 解析，优先二进制旁顶层 assets，回退到包内
        from divere.utils.app_paths import resolve_data_path
        bg_path = resolve_data_path("assets", "CIE_1976_UCS.png")
        self._background = QtGui.QPixmap(str(bg_path))

        # 坐标换算参数
        # 用户已校正：(0,0) 位于像素 (105, 100)（自左下角），每 0.1 = 297 px → 每 1.0 = 2970 px
        self._origin_from_left = 105.0
        self._origin_from_bottom = 100.0
        self._pixels_per_unit = 2970.0

        # 交互配置
        self._point_radius = 8.0
        self._grab_tolerance = 12.0

        # 初始三点（u', v'）
        self._points_uv: Dict[str, QtCore.QPointF] = {
            "R": QtCore.QPointF(0.5, 0.5),
            "G": QtCore.QPointF(0.16, 0.55),
            "B": QtCore.QPointF(0.2, 0.1),
        }

        self._dragging_key: str | None = None

        self.setMouseTracking(True)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        # 内容缩放比例（四周留白）
        self._content_scale = 0.70

    # 尺寸策略（保持纵横比）
    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        iw = max(1, self._background.width())
        ih = max(1, self._background.height())
        return int(round(width * ih / iw))

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        iw = max(1, self._background.width())
        ih = max(1, self._background.height())
        target_w = min(800, iw)
        target_h = int(round(target_w * ih / iw))
        return QtCore.QSize(target_w, target_h)

    def minimumSizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        iw = max(1, self._background.width())
        ih = max(1, self._background.height())
        target_w = min(300, iw)
        target_h = int(round(target_w * ih / iw))
        return QtCore.QSize(target_w, target_h)

    # 公共接口
    def get_uv_coordinates(self) -> Dict[str, Tuple[float, float]]:
        return {key: (pt.x(), pt.y()) for key, pt in self._points_uv.items()}

    def set_uv_coordinates(self, coords: Dict[str, Tuple[float, float]]) -> None:
        for key in ("R", "G", "B"):
            if key in coords:
                u, v = coords[key]
                self._points_uv[key] = QtCore.QPointF(float(u), float(v))
        self.update()
        self.coordinatesChanged.emit(self.get_uv_coordinates())

    # 换算与几何
    def _image_size(self) -> QtCore.QSize:
        return self._background.size()

    def _origin_top_left(self) -> QtCore.QPointF:
        h = float(self._image_size().height())
        return QtCore.QPointF(self._origin_from_left, h - self._origin_from_bottom)

    def _uv_to_img(self, uv: QtCore.QPointF) -> QtCore.QPointF:
        origin = self._origin_top_left()
        x = origin.x() + uv.x() * self._pixels_per_unit
        y = origin.y() - uv.y() * self._pixels_per_unit
        return QtCore.QPointF(x, y)

    def _img_to_uv(self, img: QtCore.QPointF) -> QtCore.QPointF:
        origin = self._origin_top_left()
        u = (img.x() - origin.x()) / self._pixels_per_unit
        v = (origin.y() - img.y()) / self._pixels_per_unit
        return QtCore.QPointF(u, v)

    def _compute_draw_rect(self) -> QtCore.QRectF:
        if self._background.isNull():
            return QtCore.QRectF(0, 0, self.width(), self.height())
        iw = float(self._background.width())
        ih = float(self._background.height())
        ww = float(self.width())
        wh = float(self.height())
        if iw <= 0 or ih <= 0 or ww <= 0 or wh <= 0:
            return QtCore.QRectF(0, 0, self.width(), self.height())
        s = min(ww / iw, wh / ih)
        s *= float(self._content_scale)
        dw = iw * s
        dh = ih * s
        left = (ww - dw) * 0.5
        top = (wh - dh) * 0.5
        return QtCore.QRectF(left, top, dw, dh)

    def _img_to_disp(self, img: QtCore.QPointF) -> QtCore.QPointF:
        r = self._compute_draw_rect()
        iw = float(self._background.width())
        s = r.width() / iw if iw > 0 else 1.0
        x = r.left() + img.x() * s
        y = r.top() + img.y() * s
        return QtCore.QPointF(x, y)

    def _disp_to_img(self, disp: QtCore.QPointF) -> QtCore.QPointF:
        r = self._compute_draw_rect()
        iw = float(self._background.width())
        s = r.width() / iw if iw > 0 else 1.0
        x = (disp.x() - r.left()) / s
        y = (disp.y() - r.top()) / s
        return QtCore.QPointF(x, y)

    # 绘制与交互
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform, True)

        target_rect = self._compute_draw_rect()
        src_rect = QtCore.QRectF(self._background.rect())
        painter.drawPixmap(target_rect, self._background, src_rect)

        # 三角形与点
        points_px: Dict[str, QtCore.QPointF] = {}
        for key, uv in self._points_uv.items():
            img_pt = self._uv_to_img(uv)
            points_px[key] = self._img_to_disp(img_pt)

        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        r = points_px["R"]; g = points_px["G"]; b = points_px["B"]
        path = QtGui.QPainterPath(QtCore.QPointF(r))
        path.lineTo(g); path.lineTo(b); path.lineTo(r)
        painter.drawPath(path)

        painter.setBrush(QtGui.QColor(255, 255, 255, 50))
        painter.drawPolygon(QtGui.QPolygonF([r, g, b]))

        for key, px in points_px.items():
            color = {"R": QtGui.QColor(255, 64, 64), "G": QtGui.QColor(64, 220, 64), "B": QtGui.QColor(64, 128, 255)}[key]
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 200), 3))
            painter.setBrush(color)
            painter.drawEllipse(px, self._point_radius, self._point_radius)
            uv = self._points_uv[key]
            label = f"{key}  u'={uv.x():.4f}  v'={uv.y():.4f}"
            text_rect = QtCore.QRectF(px.x() + 10, px.y() - 22, 240, 22)
            painter.fillRect(text_rect, QtGui.QColor(0, 0, 0, 140))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            painter.drawText(text_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, label)

        # 十字参考：(0,0)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 180), 1))
        origin_img = self._origin_top_left(); origin_disp = self._img_to_disp(origin_img)
        ox, oy = origin_disp.x(), origin_disp.y()
        painter.drawLine(ox - 8, oy, ox + 8, oy)
        painter.drawLine(ox, oy - 8, ox, oy + 8)
        # 十字参考：(0.6, 0.6)
        ref_uv = QtCore.QPointF(0.6, 0.6)
        ref_img = self._uv_to_img(ref_uv); ref_disp = self._img_to_disp(ref_img)
        rx, ry = ref_disp.x(), ref_disp.y()
        painter.drawLine(rx - 8, ry, rx + 8, ry)
        painter.drawLine(rx, ry - 8, rx, ry + 8)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = QtCore.QPointF(event.position())
        closest_key = None; closest_dist2 = float("inf")
        for key, uv in self._points_uv.items():
            px = self._img_to_disp(self._uv_to_img(uv))
            d2 = (px.x() - pos.x()) ** 2 + (px.y() - pos.y()) ** 2
            if d2 < closest_dist2:
                closest_key = key; closest_dist2 = d2
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if closest_key is not None and closest_dist2 <= (self._grab_tolerance ** 2):
                self.resetPointRequested.emit(closest_key)
            return
        if closest_key is not None and closest_dist2 <= (self._grab_tolerance ** 2):
            self._dragging_key = closest_key
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        else:
            self._dragging_key = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = QtCore.QPointF(event.position())
        if self._dragging_key is not None:
            img_pt = self._disp_to_img(pos)
            uv = self._img_to_uv(img_pt)
            self._points_uv[self._dragging_key] = uv
            self.coordinatesChanged.emit(self.get_uv_coordinates())
            self.update()
        else:
            hover = False
            for uv in self._points_uv.values():
                px = self._img_to_disp(self._uv_to_img(uv))
                d2 = (px.x() - pos.x()) ** 2 + (px.y() - pos.y()) ** 2
                if d2 <= (self._grab_tolerance ** 2):
                    hover = True; break
            self.setCursor(QtCore.Qt.OpenHandCursor if hover else QtCore.Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging_key is not None:
            self._dragging_key = None
            self.setCursor(QtCore.Qt.ArrowCursor)


