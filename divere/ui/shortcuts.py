# shortcuts.py
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QApplication

import sys


class ShortcutsBinder(QObject):
    """
    统一注册快捷键并内置动作逻辑。
    依赖 host 提供以下属性/方法：
        host.context.rotate(deg)
        host.context.get_current_params() -> params or None
        host.context.update_params(params)
        host.preview_widget.context.folder_navigator.navigate_previous()
        host.preview_widget.context.folder_navigator.navigate_next()
        host._show_status_message(str)
        host._reset_parameters()
        host._set_folder_default()
    """
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self._shortcuts = []  # 持有引用，避免被GC

    # ---------- 公共入口 ----------
    def setup_default_shortcuts(self):
        add = self._add

        # 导航：左右箭头
        add(Qt.Key_Left,  self._act_go_prev)
        add(Qt.Key_Right, self._act_go_next)

        # 旋转：[ / ]
        add(Qt.Key_BracketLeft,  self._act_rotate_left)
        add(Qt.Key_BracketRight, self._act_rotate_right)

        # AI校色：空格 / Ctrl+空格
        add(Qt.Key_Space, self._act_auto_color)
        if sys.platform == "darwin":
            add("Shift+Space", self._act_auto_color_multi)
        else:
            add("Ctrl+Space", self._act_auto_color_multi)

        # Ctrl+V：粘贴默认；Ctrl+C：复制为默认
        add("Ctrl+V", self._act_reset_parameters)
        add("Ctrl+C", self._act_set_folder_default)

        # 参数调节：Q/E/A/D/W/S/R/F
        add(Qt.Key_Q, self._act_R_down)  # R降曝光（增青）
        add(Qt.Key_E, self._act_R_up)    # R增曝光（增红）

        add(Qt.Key_A, self._act_B_down)  # B降曝光（增黄）
        add(Qt.Key_D, self._act_B_up)    # B增曝光（增蓝）

        add(Qt.Key_W, self._act_dmax_down)  # dmax降低（提升曝光）
        add(Qt.Key_S, self._act_dmax_up)    # dmax增大（降低曝光）

        add(Qt.Key_R, self._act_gamma_up)   # gamma增大（增加反差）
        add(Qt.Key_F, self._act_gamma_down) # gamma减小（降低反差）

    # ---------- 内部：工具 ----------
    def _add(self, seq, slot, context=Qt.ApplicationShortcut):
        sc = QShortcut(QKeySequence(seq), self.host)
        sc.setContext(context)
        sc.activated.connect(slot)
        self._shortcuts.append(sc)
        return sc

    def _step(self):
        mods = QApplication.keyboardModifiers()
        return 0.001 if (mods & Qt.ControlModifier) else 0.01

    def _with_params(self, fn):
        params = self.host.context.get_current_params()
        if params is None:
            return False
        fn(params)
        self.host.context.update_params(params)
        return True

    # ---------- 内置动作（不再依赖主窗实现） ----------
    # 导航
    def _act_go_prev(self):
        self.host.preview_widget.context.folder_navigator.navigate_previous()
        self.host._show_status_message("已切换到上一张照片")

    def _act_go_next(self):
        self.host._show_status_message("⏳正在切换到下一张照片...")
        self.host.preview_widget.context.folder_navigator.navigate_next()
        self.host._show_status_message("已切换到下一张照片")

    # 旋转
    def _act_rotate_left(self):
        self.host.context.rotate(90)
        self.host._show_status_message("左旋转 90°")

    def _act_rotate_right(self):
        self.host.context.rotate(-90)
        self.host._show_status_message("右旋转 90°")

    # 参数重置/默认
    def _act_reset_parameters(self):
        self.host._reset_parameters()
        self.host._show_status_message("参数已重置")

    def _act_set_folder_default(self):
        self.host._set_folder_default()
        self.host._show_status_message("已设为文件夹默认")

    # R通道
    def _act_R_down(self):
        step = self._step()
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r - step)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    def _act_R_up(self):
        step = self._step()
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r + step)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    # B通道
    def _act_B_down(self):
        step = self._step()
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b - step)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    def _act_B_up(self):
        step = self._step()
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b + step)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    # dmax
    def _act_dmax_down(self):
        step = self._step()
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax - step))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    def _act_dmax_up(self):
        step = self._step()
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax + step))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    # gamma
    def _act_gamma_up(self):
        step = self._step()
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma + step))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    def _act_gamma_down(self):
        step = self._step()
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma - step))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    def _act_auto_color(self):
        self.host._on_auto_color_requested()
        self.host._show_status_message("校色一次")

    def _act_auto_color_multi(self):
        self.host._on_auto_color_iterative_requested()
        self.host._show_status_message("校色一次")


class ImeBracketFilter(QObject):
    """
    输入法全角括号兜底过滤器：捕捉【】并调用旋转动作。
    """
    def __init__(self, binder: ShortcutsBinder):
        super().__init__(binder.host)
        self._binder = binder

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            t = event.text()
            if t == "【":
                self._binder._act_rotate_left()
                return True
            elif t == "】":
                self._binder._act_rotate_right()
                return True
        return False


def install_ime_brackets_fallback(binder: ShortcutsBinder, install_on_app=True):
    """
    安装【】兜底过滤器到 QApplication（默认）或 host。
    返回过滤器对象（需在外部持有引用）。
    """
    filt = ImeBracketFilter(binder)
    if install_on_app:
        QApplication.instance().installEventFilter(filt)
    else:
        binder.host.installEventFilter(filt)
    return filt