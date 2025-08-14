from __future__ import annotations

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor


def _build_dark_palette() -> QPalette:
    palette = QPalette()
    window = QColor(40, 40, 40)
    base = QColor(35, 35, 35)
    alt_base = QColor(50, 50, 50)
    text = QColor(230, 230, 230)
    mid_text = QColor(200, 200, 200)
    disabled_text = QColor(140, 140, 140)
    button = QColor(55, 55, 55)
    button_text = text
    highlight = QColor(70, 120, 210)
    highlighted_text = QColor(255, 255, 255)

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, alt_base)
    palette.setColor(QPalette.ToolTipBase, alt_base)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, button)
    palette.setColor(QPalette.ButtonText, button_text)
    palette.setColor(QPalette.BrightText, highlighted_text)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, highlighted_text)
    palette.setColor(QPalette.PlaceholderText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    return palette


def _build_light_palette() -> QPalette:
    # 明亮主题：显式设置，以避免在系统暗黑模式下继承系统暗色
    palette = QPalette()
    window = QColor(245, 245, 245)
    base = QColor(255, 255, 255)
    alt_base = QColor(245, 245, 245)
    text = QColor(20, 20, 20)
    disabled_text = QColor(140, 140, 140)
    button = QColor(250, 250, 250)
    button_text = QColor(30, 30, 30)
    highlight = QColor(45, 110, 200)
    highlighted_text = QColor(255, 255, 255)

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, alt_base)
    palette.setColor(QPalette.ToolTipBase, base)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, button)
    palette.setColor(QPalette.ButtonText, button_text)
    palette.setColor(QPalette.BrightText, highlighted_text)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, highlighted_text)
    palette.setColor(QPalette.PlaceholderText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    return palette


def _dark_qss() -> str:
    return """
    QToolBar#mainToolBar {
        background: #2d2d2d;
        border: 1px solid #404040;
        spacing: 6px;
        padding: 3px;
    }
    QToolButton {
        color: #e6e6e6;
        background-color: #3a3a3a;
        border: 1px solid #505050;
        border-radius: 3px;
        padding: 4px 8px;
    }
    QToolButton:hover { background-color: #474747; border-color: #6a6a6a; }
    QToolButton:pressed { background-color: #3f3f3f; border-color: #808080; }

    QGroupBox { border: 1px solid #444; margin-top: 6px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #dddddd; }

    QScrollArea, QScrollArea > QWidget > QWidget { background: #282828; }
    QScrollArea { border: 1px solid #404040; }
    QDockWidget { border: 1px solid #404040; }
    QDockWidget::title { background: #333333; padding: 4px 6px; color: #dddddd; }
    QSplitter::handle { background: #404040; }
    QSplitter::handle:hover { background: #5a5a5a; }
    QSplitter::handle:horizontal { width: 6px; margin: 0 2px; }
    QSplitter::handle:vertical { height: 6px; margin: 2px 0; }
    QTabBar::tab { background: #3a3a3a; color: #dddddd; padding: 5px 10px; border: 1px solid #505050; border-bottom: none; }
    QTabBar::tab:selected { background: #2f2f2f; }
    QTabBar::tab:hover { background: #454545; }
    QTabWidget::pane { border: 1px solid #505050; top: -1px; }

    QLabel#noteLabel {
        color: #c8c8c8; font-size: 11px; padding: 8px;
        background-color: #2f2f2f; border: 1px solid #505050; border-radius: 4px;
    }
    QLabel#imageCanvas { background-color: #2b2b2b; color: #ffffff; border: 1px solid #404040; }
    """


def _light_qss() -> str:
    return """
    QToolBar#mainToolBar {
        background: #f5f5f5;
        border: 1px solid #dcdcdc;
        spacing: 6px;
        padding: 3px;
    }
    QToolButton {
        color: #222222;
        background-color: #f8f8f8;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 4px 8px;
    }
    QToolButton:hover { background-color: #e8e8e8; border-color: #aaaaaa; }
    QToolButton:pressed { background-color: #dcdcdc; border-color: #888888; }

    QGroupBox { border: 1px solid #d0d0d0; margin-top: 6px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #333333; }

    QScrollArea, QScrollArea > QWidget > QWidget { background: #f5f5f5; }
    QScrollArea { border: 1px solid #dcdcdc; }
    QDockWidget { border: 1px solid #dcdcdc; }
    QDockWidget::title { background: #f1f1f1; padding: 4px 6px; color: #333333; }
    QSplitter::handle { background: #dcdcdc; }
    QSplitter::handle:hover { background: #c8c8c8; }
    QSplitter::handle:horizontal { width: 6px; margin: 0 2px; }
    QSplitter::handle:vertical { height: 6px; margin: 2px 0; }
    QTabBar::tab { background: #efefef; color: #222222; padding: 5px 10px; border: 1px solid #d0d0d0; border-bottom: none; }
    QTabBar::tab:selected { background: #ffffff; }
    QTabBar::tab:hover { background: #e8e8e8; }
    QTabWidget::pane { border: 1px solid #d0d0d0; top: -1px; }

    QLabel#noteLabel {
        color: #666666; font-size: 11px; padding: 8px;
        background-color: #f8f8f8; border: 1px solid #dddddd; border-radius: 4px;
    }
    QLabel#imageCanvas { background-color: #eaeaea; color: #000000; border: 1px solid #d0d0d0; }
    """


def apply_theme(app: QApplication, theme: str) -> None:
    theme = (theme or "dark").lower()
    # 使用Fusion样式，避免平台/系统主题干扰
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    if theme == "dark":
        app.setPalette(_build_dark_palette())
        base_qss = _dark_qss()
    else:
        app.setPalette(_build_light_palette())
        base_qss = _light_qss()
    # 叠加到已有样式表（不强制替换第三方控件样式）
    app.setStyleSheet(base_qss)
    # 记录当前主题
    app.setProperty("_divere_theme", theme)


def current_theme(app: QApplication) -> str:
    t = app.property("_divere_theme")
    return t if isinstance(t, str) else "dark"


