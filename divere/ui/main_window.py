"""
主窗口界面
"""

import sys
import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QMenuBar, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QSplitter, QLabel, QDockWidget, QDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
import numpy as np

from divere.core.image_manager import ImageManager
from divere.core.color_space import ColorSpaceManager
from divere.core.the_enlarger import TheEnlarger
from divere.core.lut_processor import LUTProcessor

from divere.core.data_types import ImageData, ColorGradingParams
from divere.utils.enhanced_config_manager import enhanced_config_manager

from .preview_widget import PreviewWidget
from .save_dialog import SaveImageDialog
from .parameter_panel import ParameterPanel


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化核心组件
        self.image_manager = ImageManager()
        self.color_space_manager = ColorSpaceManager()
        self.the_enlarger = TheEnlarger()
        self.lut_processor = LUTProcessor(self.the_enlarger)
        
        # 当前状态
        self.current_image: Optional[ImageData] = None
        self.current_proxy: Optional[ImageData] = None
        self.current_params = ColorGradingParams()
        self.input_color_space: str = "Film_KodakRGB_Linear"  # 默认输入色彩空间
        
        # 设置窗口
        self.setWindowTitle("DiVERE - 专业胶片校色工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建界面
        self._create_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        
        # 初始化默认色彩空间
        self._initialize_color_space_info()
        
        # 实时预览更新（智能延迟机制）
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(10)  # 10ms延迟，超快响应
        
        # 拖动状态跟踪
        self.is_dragging = False
        
        # 最后，初始化参数面板的默认值
        self.parameter_panel.initialize_defaults()
        
        # 自动加载测试图像（可选）
        # self._load_demo_image()
        
    def _create_ui(self):
        """创建用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧参数面板
        self.parameter_panel = ParameterPanel(self)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        parameter_dock = QDockWidget("调色参数", self)
        parameter_dock.setWidget(self.parameter_panel)
        parameter_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, parameter_dock)
        
        # 中央预览区域
        self.preview_widget = PreviewWidget()
        self.preview_widget.image_rotated.connect(self._on_image_rotated)
        splitter.addWidget(self.preview_widget)
        
        # 设置分割器比例
        splitter.setSizes([300, 800])
        

    
    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 打开图像
        open_action = QAction("打开图像", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # 选择输入色彩空间
        colorspace_action = QAction("设置输入色彩空间", self)
        colorspace_action.triggered.connect(self._select_input_color_space)
        file_menu.addAction(colorspace_action)
        
        file_menu.addSeparator()
        
        # 保存图像
        save_action = QAction("保存图像", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        # 重置参数
        reset_action = QAction("重置参数", self)
        reset_action.triggered.connect(self._reset_parameters)
        edit_menu.addAction(reset_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 显示原始图像
        show_original_action = QAction("显示原始图像", self)
        show_original_action.setCheckable(True)
        show_original_action.triggered.connect(self._toggle_original_view)
        view_menu.addAction(show_original_action)
        
        view_menu.addSeparator()
        
        # 视图控制
        reset_view_action = QAction("重置视图", self)
        reset_view_action.setShortcut(QKeySequence("0"))
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        # 估算胶片类型
        estimate_film_action = QAction("估算胶片类型", self)
        estimate_film_action.triggered.connect(self._estimate_film_type)
        tools_menu.addAction(estimate_film_action)

        # 配置管理
        tools_menu.addSeparator()
        config_manager_action = QAction("配置管理器", self)
        config_manager_action.triggered.connect(self._open_config_manager)
        tools_menu.addAction(config_manager_action)

        # 启用预览Profiling
        tools_menu.addSeparator()
        profiling_action = QAction("启用预览Profiling", self)
        profiling_action.setCheckable(True)
        profiling_action.toggled.connect(self._toggle_profiling)
        tools_menu.addAction(profiling_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 设置工具栏样式，让按钮更和谐
        toolbar.setStyleSheet("""
            QToolBar {
                spacing: 6px;
                padding: 3px;
            }
            QToolButton {
                min-width: 60px;
                min-height: 24px;
                font-size: 12px;
                font-weight: normal;
                padding: 4px 8px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #f8f8f8;
            }
            QToolButton:hover {
                background-color: #e8e8e8;
                border-color: #999999;
            }
            QToolButton:pressed {
                background-color: #d8d8d8;
                border-color: #666666;
            }
        """)
        
        # 打开图像
        open_action = QAction("打开", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)
        
        # 保存图像
        save_action = QAction("保存", self)
        save_action.triggered.connect(self._save_image)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 重置参数
        reset_action = QAction("重置", self)
        reset_action.triggered.connect(self._reset_parameters)
        toolbar.addAction(reset_action)
        

    
    def _create_statusbar(self):
        """创建状态栏"""
        self.statusBar().showMessage("就绪")
    
    def _open_image(self):
        """打开图像文件"""
        # 获取上次打开的目录
        last_directory = enhanced_config_manager.get_directory("open_image")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图像文件",
            last_directory,
            "图像文件 (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp)"
        )
        
        if file_path:
            # 保存当前目录
            enhanced_config_manager.set_directory("open_image", file_path)
            
            try:
                # 加载图像
                self.current_image = self.image_manager.load_image(file_path)
                
                                # 生成代理
                self.current_proxy = self.image_manager.generate_proxy(self.current_image)
                
                # 设置输入色彩空间
                self.current_proxy = self.color_space_manager.set_image_color_space(
                    self.current_proxy, self.input_color_space
                )
                print(f"设置输入色彩空间: {self.input_color_space}")
                
                # 转换到工作色彩空间
                self.current_proxy = self.color_space_manager.convert_to_working_space(
                    self.current_proxy
                )
                print(f"转换到工作色彩空间: {self.current_proxy.color_space}")
                
                # 生成更小的代理图像用于实时预览
                proxy_size_val = self.current_params.proxy_size
                try:
                    size = int(proxy_size_val.split('x')[0])
                except:
                    size = 1920 # 默认值
                
                self.current_proxy = self.image_manager.generate_proxy(self.current_proxy, (size, size))
                print(f"生成实时预览代理: {self.current_proxy.width}x{self.current_proxy.height}")
                
                # 触发预览
                self._update_preview()
                
                # 自动适应窗口大小
                self.preview_widget.fit_to_window()
                
                self.statusBar().showMessage(f"已加载图像: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
    
    def _select_input_color_space(self):
        """选择输入色彩空间"""
        from PySide6.QtWidgets import QInputDialog
        
        # 获取可用的色彩空间列表
        available_spaces = self.color_space_manager.get_available_color_spaces()
        
        # 显示选择对话框
        color_space, ok = QInputDialog.getItem(
            self, 
            "选择输入色彩空间", 
            "请选择图像的输入色彩空间:", 
            available_spaces, 
            available_spaces.index(self.input_color_space) if self.input_color_space in available_spaces else 0, 
            False
        )
        
        if ok and color_space:
            try:
                self.input_color_space = color_space
                

                
                # 更新状态栏
                self.statusBar().showMessage(f"已设置输入色彩空间: {color_space}")
                
                # 如果已经有图像，重新处理
                if self.current_image:
                    self._reload_with_color_space()
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置色彩空间失败: {str(e)}")
    
    def _reload_with_icc(self):
        """使用新的ICC配置文件重新加载图像"""
        if not self.current_image:
            return
            
        try:
            # 重新生成代理
            self.current_proxy = self.image_manager.generate_proxy(self.current_image)
            
            # 应用ICC配置文件
            if self.input_icc_profile:
                self.current_proxy = self.color_space_manager.apply_icc_profile_to_image(
                    self.current_proxy, self.input_icc_profile
                )
            
            # 转换到工作色彩空间
            source_color_space = self.current_proxy.color_space
            self.current_proxy = self.color_space_manager.convert_to_working_space(
                self.current_proxy, source_color_space
            )
            
            # 重新生成小代理（如果需要）
            if self.current_params.small_proxy:
                try:
                    size_str = self.current_params.proxy_size
                    if 'x' in size_str:
                        size = int(size_str.split('x')[0])
                    else:
                        size = int(size_str)
                    proxy_size = (size, size)
                except:
                    proxy_size = (256, 256)
            else:
                proxy_size = (512, 512)
            
            self.current_proxy = self.image_manager.generate_proxy(self.current_proxy, proxy_size)
            
            # 更新预览
            self._update_preview()
            
            # 自动适应窗口大小
            self.preview_widget.fit_to_window()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新处理图像失败: {str(e)}")
    
    def _reload_with_color_space(self):
        """使用新的色彩空间重新加载图像"""
        if not self.current_image:
            return
        
        try:
            # 重新生成代理
            self.current_proxy = self.image_manager.generate_proxy(self.current_image)
            
            # 设置新的色彩空间
            self.current_proxy = self.color_space_manager.set_image_color_space(
                self.current_proxy, self.input_color_space
            )
            
            # 转换到工作色彩空间
            self.current_proxy = self.color_space_manager.convert_to_working_space(
                self.current_proxy
            )
            
            # 生成更小的代理图像用于实时预览
            if self.current_params.small_proxy:
                # 从proxy_size字符串解析大小
                try:
                    size_str = self.current_params.proxy_size
                    if 'x' in size_str:
                        size = int(size_str.split('x')[0])
                    else:
                        size = int(size_str)
                    proxy_size = (size, size)
                except:
                    proxy_size = (256, 256)  # 默认值
            else:
                proxy_size = (512, 512)
            
            self.current_proxy = self.image_manager.generate_proxy(self.current_proxy, proxy_size)
            
            # 重新处理预览
            self._update_preview()
            
            # 自动适应窗口大小
            self.preview_widget.fit_to_window()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新加载图像失败: {str(e)}")
    
    def _initialize_color_space_info(self):
        """初始化色彩空间信息"""
        try:
            # 验证默认色彩空间
            if self.color_space_manager.validate_color_space(self.input_color_space):
                self.statusBar().showMessage(f"已设置默认输入色彩空间: {self.input_color_space}")
            else:
                # 如果默认色彩空间无效，使用第一个可用的
                available_spaces = self.color_space_manager.get_available_color_spaces()
                if available_spaces:
                    self.input_color_space = available_spaces[0]
                    self.statusBar().showMessage(f"默认色彩空间无效，使用: {self.input_color_space}")
                else:
                    print("错误: 没有可用的色彩空间")
        except Exception as e:
            print(f"初始化色彩空间信息失败: {str(e)}")
    
    def _save_image(self):
        """保存图像"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 获取可用的色彩空间
        available_spaces = self.color_space_manager.get_available_color_spaces()
        
        # 打开保存设置对话框
        save_dialog = SaveImageDialog(self, available_spaces)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # 获取保存设置
        settings = save_dialog.get_settings()
        
        # 确定文件扩展名
        extension = ".tiff" if settings["format"] == "tiff" else ".jpg"
        filter_str = "TIFF文件 (*.tiff *.tif)" if settings["format"] == "tiff" else "JPEG文件 (*.jpg *.jpeg)"
        
        # 生成默认文件名：{原文件名}_CC_{色彩空间名}
        original_filename = Path(self.current_image.file_path).stem if self.current_image.file_path else "untitled"
        default_filename = f"{original_filename}_CC_{settings['color_space']}{extension}"
        
        # 获取上次保存的目录
        last_directory = enhanced_config_manager.get_directory("save_image")
        if last_directory:
            default_path = str(Path(last_directory) / default_filename)
        else:
            default_path = default_filename
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            default_path,
            filter_str
        )
        
        if file_path:
            # 保存当前目录
            enhanced_config_manager.set_directory("save_image", file_path)
            
            try:
                # 重要：将原图转换到工作色彩空间，保持与预览一致
                print(f"导出前的色彩空间转换:")
                print(f"  原始图像色彩空间: {self.current_image.color_space}")
                print(f"  输入色彩空间设置: {self.input_color_space}")
                
                # 先设置输入色彩空间
                working_image = self.color_space_manager.set_image_color_space(
                    self.current_image, self.input_color_space
                )
                # 转换到工作色彩空间（ACEScg）
                working_image = self.color_space_manager.convert_to_working_space(
                    working_image
                )
                print(f"  转换后工作色彩空间: {working_image.color_space}")
                
                # 应用调色参数到工作空间的图像（根据设置决定是否包含曲线）
                result_image = self.the_enlarger.apply_full_pipeline(
                    working_image, 
                    self.current_params,
                    include_curve=settings["include_curve"]
                )
                
                # 转换到输出色彩空间
                result_image = self.color_space_manager.convert_to_display_space(
                    result_image, settings["color_space"]
                )
                
                # 保存图像
                self.image_manager.save_image(
                    result_image, 
                    file_path, 
                    bit_depth=settings["bit_depth"],
                    quality=95
                )
                
                self.statusBar().showMessage(
                    f"图像已保存: {Path(file_path).name} "
                    f"({settings['bit_depth']}bit, {settings['color_space']})"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")
    
    def _reset_parameters(self):
        """重置调色参数"""
        # 创建一个新的默认参数对象
        self.current_params = ColorGradingParams()
        # 手动设置我们想要的非标准默认值
        self.current_params.density_gamma = 2.6
        self.current_params.correction_matrix_file = "Cineon_States_M_to_Print_Density"
        self.current_params.enable_correction_matrix = True
        
        # 设置默认曲线为Kodak Endura Paper
        self._load_default_curves()
        
        # 将重置后的参数应用到UI
        self.parameter_panel.current_params = self.current_params
        self.parameter_panel.update_ui_from_params()
        
        # 触发预览更新
        self._update_preview()
        self.statusBar().showMessage("参数已重置")
    
    def _load_default_curves(self):
        """加载默认曲线（Kodak Endura Paper）"""
        try:
            curve_file = Path("config/curves/Kodak_Endura_Paper.json")
            if curve_file.exists():
                with open(curve_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'curves' in data and 'RGB' in data['curves']:
                        # 设置RGB主曲线
                        self.current_params.curve_points = data['curves']['RGB']
                        self.current_params.enable_curve = True
                        
                        # 设置单通道曲线
                        if 'R' in data['curves']:
                            self.current_params.curve_points_r = data['curves']['R']
                            self.current_params.enable_curve_r = True
                        if 'G' in data['curves']:
                            self.current_params.curve_points_g = data['curves']['G']
                            self.current_params.enable_curve_g = True
                        if 'B' in data['curves']:
                            self.current_params.curve_points_b = data['curves']['B']
                            self.current_params.enable_curve_b = True
                        
                    else:
                        print("默认曲线文件格式不正确")
            else:
                print("默认曲线文件不存在")
        except Exception as e:
            print(f"加载默认曲线失败: {e}")
    
    def _toggle_original_view(self, checked: bool):
        """切换原始图像视图"""
        if checked:
            # 显示原始图像
            if self.current_proxy:
                self.preview_widget.set_image(self.current_proxy)
        else:
            # 显示调色后的图像
            self._update_preview()
    
    def _reset_view(self):
        """重置预览视图"""
        self.preview_widget.reset_view()
        self.statusBar().showMessage("视图已重置")
    

    
    def _estimate_film_type(self):
        """估算胶片类型"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "没有加载的图像")
            return
        
        try:
            film_type = self.grading_engine.estimate_film_type(self.current_image)
            QMessageBox.information(self, "胶片类型", f"估算的胶片类型: {film_type}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"估算胶片类型失败: {str(e)}")
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 DiVERE",
            "DiVERE - 专业胶片校色工具\n\n"
            "版本: 0.1.0\n"
            "基于ACEScg Linear工作流的数字化胶片后期处理\n\n"
            "© 2025 V7"
        )
    

    
    def _update_preview(self):
        """更新预览"""
        if not self.current_proxy:
            return
        
        try:
            import time
            t0 = time.time()
            # 使用统一的完整处理模式
            result_image = self.the_enlarger.apply_full_pipeline(
                self.current_proxy, self.current_params
            )
            t1 = time.time()
            # 转换到显示色彩空间
            result_image = self.color_space_manager.convert_to_display_space(
                result_image, "DisplayP3"
            )
            t2 = time.time()
            
            # 更新预览
            self.preview_widget.set_image(result_image)
            
            # 性能监控（细分阶段）
            print(
                f"预览耗时: 管线={(t1 - t0)*1000:.1f}ms, 显示色彩转换={(t2 - t1)*1000:.1f}ms, 总={(t2 - t0)*1000:.1f}ms"
            )
            
        except Exception as e:
            print(f"更新预览失败: {e}")

    def _open_config_manager(self):
        """打开配置管理器"""
        from divere.ui.config_manager_dialog import ConfigManagerDialog
        dialog = ConfigManagerDialog(self)
        dialog.exec()
        
    def _toggle_profiling(self, enabled: bool):
        """切换预览Profiling"""
        self.the_enlarger.set_profiling_enabled(enabled)
        self.color_space_manager.set_profiling_enabled(enabled)
        self.statusBar().showMessage("预览Profiling已开启" if enabled else "预览Profiling已关闭")
    
    def on_parameter_changed(self):
        """参数改变时的回调"""
        # 从参数面板获取最新参数
        self.current_params = self.parameter_panel.get_current_params()
        
        # 使用智能延迟机制
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        self.preview_timer.start()
    
    def get_current_params(self) -> ColorGradingParams:
        """获取当前调色参数"""
        return self.current_params
    
    def set_current_params(self, params: ColorGradingParams):
        """设置当前调色参数"""
        self.current_params = params
        
    def _on_image_rotated(self, direction):
        """处理图像旋转
        Args:
            direction: 旋转方向，1=左旋，-1=右旋
        """
        if self.current_image and self.current_proxy:
            # 旋转原始图像
            rotated_array = np.rot90(self.current_image.array, direction)
            self.current_image.array = np.ascontiguousarray(rotated_array)
            self.current_image.height, self.current_image.width = self.current_image.width, self.current_image.height
            
            # 旋转代理图像
            rotated_proxy = np.rot90(self.current_proxy.array, direction)
            self.current_proxy.array = np.ascontiguousarray(rotated_proxy)
            self.current_proxy.height, self.current_proxy.width = self.current_proxy.width, self.current_proxy.height
            
            # 重新处理预览
            self._update_preview()
            
            # 自动适应窗口大小
            self.preview_widget.fit_to_window() 