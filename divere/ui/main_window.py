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
    QSplitter, QLabel, QDockWidget, QDialog, QApplication
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtGui import QAction, QKeySequence
import numpy as np

from divere.core.image_manager import ImageManager
from divere.core.color_space import ColorSpaceManager
from divere.core.the_enlarger import TheEnlarger
from divere.core.lut_processor import LUTProcessor

from divere.core.data_types import (
    ImageData, ColorGradingParams, Preset, 
    ColorSpaceDefinition, MatrixDefinition
)
from divere.utils.enhanced_config_manager import enhanced_config_manager
from divere.utils.preset_manager import PresetManager, apply_preset_to_params

from .preview_widget import PreviewWidget
from .save_dialog import SaveImageDialog
from .parameter_panel import ParameterPanel
from .theme import apply_theme, current_theme


class _PreviewWorkerSignals(QObject):
    result = Signal(int, ImageData, tuple)
    error = Signal(int, str)
    finished = Signal(int)


class _PreviewWorker(QRunnable):
    def __init__(self, seq: int, image: ImageData, params: ColorGradingParams, the_enlarger, color_space_manager):
        super().__init__()
        self.seq = seq
        self.image = image
        self.params = params
        self.the_enlarger = the_enlarger
        self.color_space_manager = color_space_manager
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            import time
            t0 = time.time()
            result_image = self.the_enlarger.apply_full_pipeline(self.image, self.params)
            t1 = time.time()
            result_image = self.color_space_manager.convert_to_display_space(result_image, "DisplayP3")
            t2 = time.time()
            self.signals.result.emit(self.seq, result_image, (t0, t1, t2))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.signals.error.emit(self.seq, f"{e}\n{tb}")
        finally:
            self.signals.finished.emit(self.seq)
class MainWindow(QMainWindow):
    """主窗口"""
    preview_updated = Signal()
    
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
        self.setWindowTitle("DiVERE - 数字彩色放大机")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建界面
        self._create_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        # 主题：启动时应用上次选择
        try:
            app = QApplication.instance()
            saved_theme = enhanced_config_manager.get_ui_setting("theme", "dark")
            apply_theme(app, saved_theme)
        except Exception as _:
            pass
        
        # 初始化默认色彩空间
        self._initialize_color_space_info()
        
        # 实时预览更新（智能延迟机制 + 后台线程）
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(10)  # 10ms延迟，超快响应
        
        # 拖动状态跟踪
        self.is_dragging = False
        # 首次加载后在首帧预览到达时适应窗口
        self._fit_after_next_preview: bool = False
        
        # 预览后台线程池与任务调度
        self.thread_pool: QThreadPool = QThreadPool.globalInstance()
        # 限制为1，防止堆积；配合“忙碌/待处理”标志实现去抖
        try:
            self.thread_pool.setMaxThreadCount(1)
        except Exception:
            pass
        self._preview_busy: bool = False
        self._preview_pending: bool = False
        self._preview_seq_counter: int = 0

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

        # 加载预设
        load_preset_action = QAction("加载预设...", self)
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        # 保存预设
        save_preset_action = QAction("保存预设...", self)
        save_preset_action.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset_action)
        
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

        # 保存图像副本
        save_as_action = QAction("另存为...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_image_as)
        file_menu.addAction(save_as_action)
        
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

        # 主题切换
        view_menu.addSeparator()
        dark_action = QAction("暗黑模式", self)
        dark_action.setCheckable(True)
        try:
            dark_action.setChecked(current_theme(QApplication.instance()) == "dark")
        except Exception:
            dark_action.setChecked(True)
        dark_action.toggled.connect(self._toggle_dark_mode)
        view_menu.addAction(dark_action)
        
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
        toolbar.setObjectName("mainToolBar")
        self.addToolBar(toolbar)
        
        
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

    def _apply_theme_and_refresh(self, theme: str):
        try:
            app = QApplication.instance()
            apply_theme(app, theme)
            enhanced_config_manager.set_ui_setting("theme", theme)
            # 将主题传递给曲线编辑器的自绘颜色
            try:
                self.parameter_panel.curve_editor.curve_edit_widget.apply_palette(app.palette(), theme)
            except Exception:
                pass
        except Exception as e:
            print(f"应用主题失败: {e}")

    def _toggle_dark_mode(self, enabled: bool):
        theme = "dark" if enabled else "light"
        self._apply_theme_and_refresh(theme)
    
    def _open_image(self):
        """打开图像文件"""
        # 获取上次打开的目录
        last_directory = enhanced_config_manager.get_directory("open_image")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图像文件",
            last_directory,
            "图像文件 (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp *.fff)"
        )
        
        if file_path:
            # 保存当前目录
            enhanced_config_manager.set_directory("open_image", file_path)
            
            try:
                # 加载图像
                self.current_image = self.image_manager.load_image(file_path)
                
                # 生成代理（使用统一配置中的代理尺寸）
                self.current_proxy = self.image_manager.generate_proxy(
                    self.current_image,
                    self.the_enlarger.preview_config.get_proxy_size_tuple()
                )
                
                # 若为 .fff 文件，优先将输入色彩空间切为 AdobeRGB_Linear
                try:
                    if str(file_path).lower().endswith('.fff'):
                        self.input_color_space = 'AdobeRGB_Linear'
                        # 尝试同步参数面板下拉（若存在）
                        if hasattr(self, 'parameter_panel') and hasattr(self.parameter_panel, 'input_colorspace_combo'):
                            idx = self.parameter_panel.input_colorspace_combo.findText('AdobeRGB_Linear')
                            if idx >= 0:
                                self.parameter_panel.input_colorspace_combo.setCurrentIndex(idx)
                except Exception:
                    pass

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
                
                # 生成更小的代理图像用于实时预览（统一读取自PreviewConfig）
                proxy_size = self.the_enlarger.preview_config.get_proxy_size_tuple()
                self.current_proxy = self.image_manager.generate_proxy(self.current_proxy, proxy_size)
                print(f"生成实时预览代理: {self.current_proxy.width}x{self.current_proxy.height}")
                
                # 触发预览，并在首帧结果到达时适应窗口
                self._fit_after_next_preview = True
                self._update_preview()
                
                self.statusBar().showMessage(f"已加载图像: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
    
    def _load_preset(self):
        """加载预设文件并应用"""
        last_directory = enhanced_config_manager.get_directory("preset")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载预设", last_directory, "预设文件 (*.json)"
        )
        if not file_path:
            return

        enhanced_config_manager.set_directory("preset", str(Path(file_path).parent))

        try:
            preset = PresetManager.load_preset(file_path)
            if not preset:
                raise ValueError("加载预设返回空值")

            # 1. 应用色彩空间
            if preset.input_color_space and preset.input_color_space.name:
                cs_name = preset.input_color_space.name
                # 如果预设包含定义，则优先使用定义注册或更新色彩空间
                if preset.input_color_space.definition:
                    try:
                        primaries = np.array(preset.input_color_space.definition['primaries_xy'])
                        wp = np.array(preset.input_color_space.definition['white_point_xy'])
                        gamma = float(preset.input_color_space.definition.get('gamma', 1.0))
                        # 注册为自定义空间，确保可追溯
                        custom_name = f"{cs_name}_preset"
                        self.color_space_manager.register_custom_colorspace(
                            custom_name, primaries, white_point_xy=wp, gamma=gamma
                        )
                        self.input_color_space = custom_name
                    except Exception as e:
                        QMessageBox.warning(self, "色彩空间警告", f"无法从预设创建色彩空间 '{cs_name}': {e}。将尝试使用同名内置空间。")
                        self.input_color_space = cs_name
                else:
                    self.input_color_space = cs_name
            
            # 2. 更新参数面板的下拉框
            combo = self.parameter_panel.input_colorspace_combo
            index = combo.findText(self.input_color_space, Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                 # 如果找不到，可能是自定义的，尝试刷新列表
                self.parameter_panel.input_colorspace_combo.blockSignals(True)
                self.parameter_panel.input_colorspace_combo.clear()
                self.parameter_panel.input_colorspace_combo.addItems(self.color_space_manager.get_available_color_spaces())
                index = combo.findText(self.input_color_space, Qt.MatchFlag.MatchFixedString)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    QMessageBox.warning(self, "警告", f"预设中的色彩空间 '{self.input_color_space}' 当前不可用。")
                self.parameter_panel.input_colorspace_combo.blockSignals(False)


            # 3. 应用调色参数（部分应用）
            apply_preset_to_params(preset, self.parameter_panel.current_params)
            
            # 处理矩阵
            if preset.correction_matrix:
                matrix_def = preset.correction_matrix
                if matrix_def.name == "custom" and matrix_def.values:
                    self.parameter_panel.current_params.correction_matrix = np.array(matrix_def.values)
                    self.parameter_panel.current_params.correction_matrix_file = "custom"
                else:
                    self.parameter_panel.current_params.correction_matrix_file = matrix_def.name
                    self.parameter_panel.current_params.correction_matrix = None

            self.parameter_panel.update_ui_from_params()
            
            # 4. 如果有图像，重新加载并应用
            if self.current_image:
                self._reload_with_color_space()

            self.statusBar().showMessage(f"已加载预设: {preset.name}")

        except (IOError, ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, "加载预设失败", str(e))

    def _save_preset(self):
        """保存当前设置为预设文件"""
        if not self.current_image:
            QMessageBox.warning(self, "请先打开一张图片", "无法保存预设，因为需要基于当前状态创建。")
            return

        last_directory = enhanced_config_manager.get_directory("preset")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存预设", last_directory, "预设文件 (*.json)"
        )
        if not file_path:
            return

        # 确保文件名以.json结尾
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
        
        enhanced_config_manager.set_directory("preset", str(Path(file_path).parent))

        try:
            # 弹窗让用户输入预设名称
            from PySide6.QtWidgets import QInputDialog
            preset_name, ok = QInputDialog.getText(self, "预设名称", "请输入预设名称:")
            if not ok or not preset_name:
                preset_name = Path(file_path).stem

            # 创建Preset对象
            params = self.parameter_panel.get_current_params()
            
            # 构造Input Color Space Definition
            cs_def = None
            cs_info = self.color_space_manager.get_color_space_info(self.input_color_space)
            if cs_info:
                cs_def = ColorSpaceDefinition(
                    name=self.input_color_space,
                    definition={
                        "primaries_xy": np.array(cs_info["primaries"]).tolist(),
                        "white_point_xy": np.array(cs_info["white_point"]).tolist(),
                        "gamma": cs_info.get("gamma", 1.0)
                    }
                )

            # 构造Matrix Definition
            matrix_def = None
            matrix_name = params.correction_matrix_file
            matrix_values = None
            if matrix_name == "custom" and params.correction_matrix is not None:
                matrix_values = params.correction_matrix.tolist()
            else:
                matrix_data = self.the_enlarger._load_correction_matrix(matrix_name)
                if matrix_data and "matrix" in matrix_data:
                    matrix_values = matrix_data["matrix"]
            
            if matrix_name:
                matrix_def = MatrixDefinition(name=matrix_name, values=matrix_values)

            preset = Preset(
                name=preset_name,
                input_color_space=cs_def,
                correction_matrix=matrix_def,
                grading_params=params.to_dict()
            )

            # 保存预设
            PresetManager.save_preset(preset, file_path)
            self.statusBar().showMessage(f"预设已保存: {preset.name}")

        except (IOError, KeyError) as e:
            QMessageBox.critical(self, "保存预设失败", str(e))

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
            # 重新生成代理（使用统一配置中的代理尺寸）
            self.current_proxy = self.image_manager.generate_proxy(
                self.current_image,
                self.the_enlarger.preview_config.get_proxy_size_tuple()
            )
            
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
            proxy_size = self.the_enlarger.preview_config.get_proxy_size_tuple()
            
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
            # 重新生成代理（使用统一配置中的代理尺寸）
            self.current_proxy = self.image_manager.generate_proxy(
                self.current_image,
                self.the_enlarger.preview_config.get_proxy_size_tuple()
            )
            
            # 设置新的色彩空间
            self.current_proxy = self.color_space_manager.set_image_color_space(
                self.current_proxy, self.input_color_space
            )
            
            # 转换到工作色彩空间
            self.current_proxy = self.color_space_manager.convert_to_working_space(
                self.current_proxy
            )
            
            # 生成更小的代理图像用于实时预览（统一读取自PreviewConfig）
            proxy_size = self.the_enlarger.preview_config.get_proxy_size_tuple()
            
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
        
        self._execute_save(settings, force_dialog=True) # 强制弹出另存为
    
    def _save_image_as(self):
        """“另存为”图像"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        available_spaces = self.color_space_manager.get_available_color_spaces()
        save_dialog = SaveImageDialog(self, available_spaces)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        settings = save_dialog.get_settings()
        self._execute_save(settings, force_dialog=True)

    def _execute_save(self, settings: dict, force_dialog: bool = False):
        """执行保存操作"""
        file_path = self.current_image.file_path if self.current_image else None
        
        if force_dialog or not file_path:
            extension = ".tiff" if settings["format"] == "tiff" else ".jpg"
            filter_str = "TIFF文件 (*.tiff *.tif)" if settings["format"] == "tiff" else "JPEG文件 (*.jpg *.jpeg)"
            original_filename = Path(self.current_image.file_path).stem if self.current_image and self.current_image.file_path else "untitled"
            default_filename = f"{original_filename}_CC_{settings['color_space']}{extension}"
            
            last_directory = enhanced_config_manager.get_directory("save_image")
            default_path = str(Path(last_directory) / default_filename) if last_directory else default_filename
            
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", default_path, filter_str)
        
        if not file_path:
            return
            
        enhanced_config_manager.set_directory("save_image", str(Path(file_path).parent))
            
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
            # 导出必须使用全精度（禁用低精度LUT）+ 分块并行
            result_image = self.the_enlarger.apply_full_pipeline(
                working_image,
                self.current_params,
                include_curve=settings["include_curve"],
                for_export=True
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
            "DiVERE - 数字彩色放大机\n\n"
            "版本: 0.1.0\n"
            "基于ACEScg Linear工作流的数字化胶片后期处理\n\n"
            "© 2025 V7"
        )
    

    
    def _update_preview(self):
        """在后台线程中更新预览，保持UI线程响应。"""
        if not self.current_proxy:
            return
        if self._preview_busy:
            # 正在计算，标记一次待处理，稍后接力执行最新一次
            self._preview_pending = True
            return

        # 复制输入，避免在后台修改共享对象
        from copy import deepcopy
        try:
            proxy_copy = ImageData(
                array=self.current_proxy.array.copy() if self.current_proxy.array is not None else None,
                width=self.current_proxy.width,
                height=self.current_proxy.height,
                channels=self.current_proxy.channels,
                dtype=self.current_proxy.dtype,
                color_space=self.current_proxy.color_space,
                icc_profile=self.current_proxy.icc_profile,
                metadata=deepcopy(self.current_proxy.metadata),
                file_path=self.current_proxy.file_path,
                is_proxy=self.current_proxy.is_proxy,
                proxy_scale=self.current_proxy.proxy_scale,
            )
            params_copy = deepcopy(self.current_params)
        except Exception as e:
            print(f"预览准备失败: {e}")
            return

        self._preview_busy = True
        self._preview_seq_counter += 1
        seq = int(self._preview_seq_counter)

        worker = _PreviewWorker(
            seq=seq,
            image=proxy_copy,
            params=params_copy,
            the_enlarger=self.the_enlarger,
            color_space_manager=self.color_space_manager,
        )
        worker.signals.result.connect(self._on_preview_result)
        worker.signals.error.connect(self._on_preview_error)
        worker.signals.finished.connect(self._on_preview_finished)
        self.thread_pool.start(worker)

    def _on_preview_result(self, seq: int, result_image: ImageData, timings: tuple[float, float, float]):
        # 应用最新结果
        t0, t1, t2 = timings
        self.preview_widget.set_image(result_image)
        # 若标记为首帧需要适应窗口，则在真正有图像时触发一次
        if getattr(self, "_fit_after_next_preview", False):
            try:
                self.preview_widget.fit_to_window()
            finally:
                self._fit_after_next_preview = False
        print(f"预览耗时: 管线={(t1 - t0)*1000:.1f}ms, 显示色彩转换={(t2 - t1)*1000:.1f}ms, 总={(t2 - t0)*1000:.1f}ms")
        # 发出预览已更新信号
        self.preview_updated.emit()

    def _on_preview_error(self, seq: int, message: str):
        print(f"更新预览失败(seq={seq}): {message}")

    def _on_preview_finished(self, seq: int):
        self._preview_busy = False
        if self._preview_pending:
            self._preview_pending = False
            # 立刻触发最新一次请求
            self.preview_timer.start(0)

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
            # 在旋转前捕获预览锚点
            self.preview_widget.prepare_rotate(direction)
            # 旋转原始图像
            rotated_array = np.rot90(self.current_image.array, direction)
            self.current_image.array = np.ascontiguousarray(rotated_array)
            self.current_image.height, self.current_image.width = self.current_image.width, self.current_image.height
            
            # 旋转代理图像
            rotated_proxy = np.rot90(self.current_proxy.array, direction)
            self.current_proxy.array = np.ascontiguousarray(rotated_proxy)
            self.current_proxy.height, self.current_proxy.width = self.current_proxy.width, self.current_proxy.height
            
            # 重新处理预览（结果回到UI时会自动应用旋转锚点）
            self._update_preview()
            
            # 自动适应窗口大小
            # 不再强制适应窗口，避免缩放变化导致“旋转会缩放”的现象
