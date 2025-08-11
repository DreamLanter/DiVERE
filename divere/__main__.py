"""
DiVERE 主应用程序入口
"""

import sys
import os
from pathlib import Path

# 将工作目录切换到可执行文件所在目录（适配 .app/Contents/MacOS 与独立二进制）
try:
    executable_dir = Path(sys.argv[0]).resolve().parent
    os.chdir(executable_dir)
except Exception:
    pass

# 添加项目根目录到Python路径（开发环境下使用）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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