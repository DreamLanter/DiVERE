"""
DiVERE 主应用程序入口
"""

import sys
import os
from pathlib import Path

def get_app_root():
    """获取应用程序根目录，兼容开发环境和Nuitka编译后环境"""
    if getattr(sys, 'frozen', False):
        # Nuitka 编译后的环境
        if sys.platform == 'darwin':
            # macOS .app 包
            app_bundle = Path(sys.executable).parent.parent.parent
            return app_bundle / "Contents" / "MacOS"
        else:
            # 其他平台
            return Path(sys.executable).parent
    else:
        # 开发环境
        return Path(__file__).parent.parent

# 设置应用程序根目录
app_root = get_app_root()
sys.path.insert(0, str(app_root))

# 设置工作目录为应用程序根目录，确保相对路径正确
os.chdir(app_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from divere.ui.main_window import MainWindow


def main():
    """主函数"""
    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setApplicationName("DiVERE")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("DiVERE Team")
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("icons/app_icon.png"))
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 