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
    QSplitter, QLabel, QDockWidget, QDialog, QApplication, QPushButton
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtGui import QAction, QKeySequence
import numpy as np

from divere.core.app_context import ApplicationContext
from divere.core.data_types import ImageData, ColorGradingParams, Preset, InputTransformationDefinition, MatrixDefinition, CurveDefinition
from divere.utils.enhanced_config_manager import enhanced_config_manager
from divere.utils.preset_manager import PresetManager, apply_preset_to_params
from divere.utils.auto_preset_manager import AutoPresetManager
from divere.utils.spectral_sharpening import run as run_spectral_sharpening

from .preview_widget import PreviewWidget
from .save_dialog import SaveImageDialog
from .parameter_panel import ParameterPanel
from .theme import apply_theme, current_theme


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化核心组件
        self.context = ApplicationContext(self)
        
        # 当前状态
        # self.current_image: Optional[ImageData] = None # 已迁移
        # self.current_proxy: Optional[ImageData] = None # 已迁移
        # self.current_params = ColorGradingParams() # 已迁移
        # self.input_color_space: str = "CCFLGeneric_Linear"  # 已迁移
        # self.current_orientation: int = 0 # 已迁移

        # 设置窗口
        self.setWindowTitle("DiVERE - 数字彩色放大机")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建界面
        self._create_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        self._connect_context_signals()
        self._connect_panel_signals()

        # 主题：启动时应用上次选择
        try:
            app = QApplication.instance()
            saved_theme = enhanced_config_manager.get_ui_setting("theme", "dark")
            apply_theme(app, saved_theme)
        except Exception as _:
            pass
        
        # 初始化默认色彩空间 - 逻辑迁移到 Context
        # self._initialize_color_space_info()
        
        # 实时预览更新 - 逻辑迁移到 Context
        # self.preview_timer = QTimer()
        # self.preview_timer.timeout.connect(self._update_preview)
        # self.preview_timer.setSingleShot(True)
        # self.preview_timer.setInterval(10)  # 10ms延迟，超快响应
        
        # 拖动状态跟踪
        self.is_dragging = False
        # 首次加载后在首帧预览到达时适应窗口
        self._fit_after_next_preview: bool = False
        
        # 预览后台线程池 - 逻辑迁移到 Context
        # self.thread_pool: QThreadPool = QThreadPool.globalInstance()
        # 限制为1，防止堆积；配合“忙碌/待处理”标志实现去抖
        try:
            self.context.thread_pool.setMaxThreadCount(1)
        except Exception:
            pass
        self._preview_busy: bool = False
        self._preview_pending: bool = False
        self._preview_seq_counter: int = 0

        # 最后，初始化参数面板的默认值
        self.parameter_panel.initialize_defaults(self.context.get_current_params())
        
        # 自动加载测试图像（可选）
        # self._load_demo_image()
        
    def _apply_crop_and_rotation_for_export(self, src_image: ImageData, rect_norm: Optional[tuple], orientation_deg: int) -> ImageData:
        """按导出标准链路应用裁剪与旋转：先裁剪再旋转。"""
        try:
            out = src_image
            # 裁剪
            if rect_norm and out and out.array is not None:
                x, y, w, h = rect_norm
                H, W = out.height, out.width
                x0 = int(round(x * W)); y0 = int(round(y * H))
                x1 = int(round((x + w) * W)); y1 = int(round((y + h) * H))
                x0 = max(0, min(W - 1, x0)); x1 = max(x0 + 1, min(W, x1))
                y0 = max(0, min(H - 1, y0)); y1 = max(y0 + 1, min(H, y1))
                cropped = out.array[y0:y1, x0:x1, :].copy()
                out = out.copy_with_new_array(cropped)
            # 旋转（逆时针）
            deg = int(orientation_deg) % 360
            if deg != 0 and out and out.array is not None:
                k = (deg // 90) % 4
                if k:
                    out = out.copy_with_new_array(np.rot90(out.array, k=int(k)))
            return out
        except Exception:
            return src_image

    def _convert_to_grayscale_if_bw_mode(self, image: ImageData) -> ImageData:
        """Convert image to grayscale if current film type is B&W mode."""
        try:
            # Check if current film type is monochrome
            current_film_type = self.context.get_current_film_type()
            if not self.context.film_type_controller.is_monochrome_type(current_film_type):
                return image  # Not B&W mode, return unchanged
            
            if image.array is None or image.array.ndim != 3 or image.array.shape[2] != 3:
                return image  # Not RGB format, return unchanged
            
            # Convert RGB to grayscale using ITU-R BT.709 weights
            # Same formula as used in preview_widget.py
            luminance = (0.2126 * image.array[:, :, 0] + 
                        0.7152 * image.array[:, :, 1] + 
                        0.0722 * image.array[:, :, 2])
            
            # Create single-channel grayscale array
            grayscale_array = luminance[:, :, np.newaxis]
            
            # Return new ImageData with grayscale array
            return image.copy_with_new_array(grayscale_array)
            
        except Exception as e:
            print(f"Grayscale conversion failed: {e}")
            return image  # Return original on error
    
    def _connect_context_signals(self):
        """连接 ApplicationContext 的信号到UI槽函数"""
        self.context.preview_updated.connect(self._on_preview_updated)
        self.context.status_message_changed.connect(self.statusBar().showMessage)
        self.context.image_loaded.connect(self._on_image_loaded)
        self.context.autosave_requested.connect(self._on_autosave_requested)
        try:
            self.context.preview_updated.connect(lambda _: self._update_apply_contactsheet_enabled())
        except Exception:
            pass
        try:
            self.preview_widget.request_focus_contactsheet.connect(self._on_request_focus_contactsheet)
        except Exception:
            pass

    def _connect_panel_signals(self):
        """连接ParameterPanel的信号"""
        self.parameter_panel.auto_color_requested.connect(self._on_auto_color_requested)
        self.parameter_panel.auto_color_iterative_requested.connect(self._on_auto_color_iterative_requested)
        self.parameter_panel.ccm_optimize_requested.connect(self._on_ccm_optimize_requested)
        self.parameter_panel.save_custom_colorspace_requested.connect(self._on_save_custom_colorspace_requested)
        self.parameter_panel.toggle_color_checker_requested.connect(self.preview_widget.toggle_color_checker)
        # 色卡变换信号连接
        self.parameter_panel.cc_flip_horizontal_requested.connect(self.preview_widget.flip_colorchecker_horizontal)
        self.parameter_panel.cc_flip_vertical_requested.connect(self.preview_widget.flip_colorchecker_vertical)
        self.parameter_panel.cc_rotate_left_requested.connect(self.preview_widget.rotate_colorchecker_left)
        self.parameter_panel.cc_rotate_right_requested.connect(self.preview_widget.rotate_colorchecker_right)
        # 当 UCS 三角拖动结束：注册/切换到一个临时 custom 输入空间，触发代理重建与预览
        self.parameter_panel.custom_primaries_changed.connect(self._on_custom_primaries_changed)
        # LUT导出信号
        self.parameter_panel.lut_export_requested.connect(self._on_lut_export_requested)
        # 预览裁剪交互
        self.preview_widget.crop_committed.connect(self._on_crop_committed)
        # 单张裁剪（不创建正式crop项）
        try:
            self.preview_widget.single_crop_committed.connect(self._on_single_crop_committed)
        except Exception:
            pass
        self.preview_widget.crop_updated.connect(self._on_crop_updated)
        self.preview_widget.request_focus_crop.connect(self._on_request_focus_crop)
        self.preview_widget.request_restore_crop.connect(self._on_request_restore_crop)
        # 裁剪选择条信号
        try:
            self.preview_widget.request_switch_profile.connect(self._on_request_switch_profile)
            self.preview_widget.request_new_crop.connect(self._on_request_new_crop)
            self.preview_widget.request_delete_crop.connect(self._on_request_delete_crop)
        except Exception:
            pass
        # Context → UI：裁剪改变后刷新 overlay
        try:
            self.context.crop_changed.connect(self.preview_widget.set_crop_overlay)
        except Exception:
            pass

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
        self.parameter_panel = ParameterPanel(self.context)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        self.parameter_panel.input_colorspace_changed.connect(self.on_input_colorspace_changed)
        self.parameter_panel.film_type_changed.connect(self.on_film_type_changed)
        
        # Connect ApplicationContext signals
        self.context.film_type_changed.connect(self.on_context_film_type_changed)
        parameter_dock = QDockWidget("调色参数", self)
        parameter_dock.setWidget(self.parameter_panel)
        parameter_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, parameter_dock)
        
        # 中央预览区域
        self.preview_widget = PreviewWidget(self.context)
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
        load_preset_action = QAction("导入预设...", self)
        load_preset_action.setToolTip("从文件导入预设并应用到当前图像")
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        # 保存预设
        save_preset_action = QAction("导出预设...", self)
        save_preset_action.setToolTip("将当前参数导出为预设文件")
        save_preset_action.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset_action)
        
        file_menu.addSeparator()
        
        # 选择输入色彩变换
        colorspace_action = QAction("设置输入色彩变换", self)
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
        
        tools_menu.addSeparator()
        
        # 文件分类规则管理器
        file_classification_action = QAction("文件分类规则管理器", self)
        file_classification_action.setToolTip("管理文件分类规则和默认预设文件")
        file_classification_action.triggered.connect(self._open_file_classification_manager)
        tools_menu.addAction(file_classification_action)

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
        
        # LUT数学一致性验证
        lut_test_action = QAction("测试LUT数学一致性", self)
        lut_test_action.triggered.connect(self._test_lut_chain_consistency)
        tools_menu.addAction(lut_test_action)
        
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
        # 沿用接触印像设置（只在聚焦裁剪时可用）
        apply_contactsheet_action = QAction("沿用接触印像设置", self)
        apply_contactsheet_action.setToolTip("将接触印像的调色参数复制到当前裁剪")
        apply_contactsheet_action.triggered.connect(self._on_apply_contactsheet_to_crop)
        toolbar.addAction(apply_contactsheet_action)
        self._apply_contactsheet_action = apply_contactsheet_action
        # 初始禁用，进入聚焦模式后启用
        try:
            self._apply_contactsheet_action.setEnabled(False)
        except Exception:
            pass
        

    
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
            # 保存当前目录（写入父目录）
            enhanced_config_manager.set_directory("open_image", file_path)
            self.context.load_image(file_path)

    def _on_image_loaded(self):
        self._fit_after_next_preview = True
        # 图像加载后刷新裁剪选择条（基于 Context 内部列表）
        try:
            crops = getattr(self.context, '_crops', [])
            active_id = getattr(self.context, '_active_crop_id', None)
            self.preview_widget.refresh_crop_selector(crops, active_id)
        except Exception:
            pass

    def _on_autosave_requested(self):
        """处理来自Context的自动保存请求"""
        current_image = self.context.get_current_image()
        if not current_image or not current_image.file_path:
            return

        # 根据是否存在裁剪，决定保存 single 还是 contactsheet
        crops = self.context.get_all_crops()
        if not crops:
            # 无裁剪：保存为 v3 single
            preset = self.context.export_single_preset()
            self.context.auto_preset_manager.save_preset_for_image(current_image.file_path, preset)
        else:
            # 有裁剪：保存为 v3 contactsheet
            bundle = self.context.export_preset_bundle()
            self.context.auto_preset_manager.save_bundle_for_image(current_image.file_path, bundle)
        
        preset_file_path = self.context.auto_preset_manager.get_current_preset_file_path()
        if preset_file_path:
            self.statusBar().showMessage(f"参数已自动保存到: {preset_file_path.name}")

    def _on_request_switch_profile(self, kind: str, crop_id: object):
        """处理切换Profile请求"""
        try:
            if kind == 'contactsheet':
                # 切换到原图模式
                self.context.switch_to_contactsheet()
                self.context.restore_crop_preview()
                is_focused = False
                # 恢复原图需要等预览更新完成再适应窗口
                self._fit_after_next_preview = True
            elif kind == 'crop' and isinstance(crop_id, str):
                # 一次性切换到裁剪并聚焦，避免闪烁
                self.context.switch_to_crop_focused(crop_id)
                is_focused = True
                # 聚焦需要等预览更新完成再适应窗口
                self._fit_after_next_preview = True
            else:
                is_focused = False
                
            # 刷新选择条状态
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused)
            # 同步参数面板
            self.parameter_panel.initialize_defaults(self.context.get_current_params())
            # 更新工具可见性/可用性
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"切换Profile失败: {e}")

    def _on_request_new_crop(self):
        """响应新增裁剪请求
        - 若已有 >=1 个裁剪：智能新增（复制相同大小并布局），不进入鼠标框选
        - 若没有裁剪：保持现有逻辑，由预览进入手动框选
        """
        try:
            crops = self.context.get_all_crops()
            if isinstance(crops, list) and len(crops) >= 1:
                # 调用智能新增：复制尺寸、按长宽比布局
                new_id = self.context.smart_add_crop()
                if new_id:
                    # 切回原图显示所有裁剪（不聚焦），并显示编号按钮
                    self.context.switch_to_contactsheet()
                    try:
                        self.preview_widget._hide_single_crop_selector = False
                    except Exception:
                        pass
                    self.preview_widget.refresh_crop_selector(
                        self.context.get_all_crops(),
                        self.context.get_active_crop_id(),
                        is_focused=False
                    )
                    # 同步参数面板
                    self.parameter_panel.initialize_defaults(self.context.get_current_params())
                return
        except Exception as e:
            print(f"智能新增裁剪失败: {e}")
        # 无裁剪时，保持原逻辑：预览组件会进入手动框选
        return

    def _on_request_delete_crop(self, crop_id: str):
        try:
            self.context.delete_crop(crop_id)
            # 刷新选择条
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            # 更新工具可用性
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"删除裁剪失败: {e}")

    def _on_apply_contactsheet_to_crop(self):
        try:
            self.context.apply_contactsheet_to_active_crop()
            # 同步参数面板
            self.parameter_panel.initialize_defaults(self.context.get_current_params())
        except Exception as e:
            print(f"沿用接触印像设置失败: {e}")

    def _on_auto_color_requested(self):
        self.context.run_auto_color_correction(self.preview_widget.get_current_image_data)

    def _on_auto_color_iterative_requested(self):
        self.context.run_iterative_auto_color(self.preview_widget.get_current_image_data)

    # _apply_preset logic is now in ApplicationContext
    # def _apply_preset(self, preset: Preset): ...

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

            # 0. 检查raw_file是否匹配
            current_image = self.context.get_current_image()
            if preset.raw_file and current_image:
                current_filename = Path(current_image.file_path).name
                if preset.raw_file != current_filename:
                    from PySide6.QtWidgets import QMessageBox
                    reply = QMessageBox.question(self, "预设警告", 
                        f"预设应用于 '{preset.raw_file}',\n"
                        f"当前图像是 '{current_filename}'.\n\n"
                        "确定要应用吗？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        return

            self.context.load_preset(preset)
            
            # 如果有图像，触发预览更新
            if current_image:
                self.context._prepare_proxy()
                self.context._trigger_preview_update()

        except (IOError, ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, "加载预设失败", str(e))

    def _create_preset_from_current_state(self, name: str) -> Preset:
        """从当前应用状态创建Preset对象"""
        params = self.context.get_current_params()
        
        # 构造 InputTransformationDefinition (名称+数值冗余)
        cs_name = self.context.get_input_color_space()
        cs_def = None
        definition = self.context.color_space_manager.get_color_space_definition(cs_name)
        if definition:
            cs_def = InputTransformationDefinition(name=cs_name, definition=definition)
        
        # 构造 MatrixDefinition
        matrix_display_name = self.parameter_panel.matrix_combo.currentText()
        matrix_values = None
        if params.density_matrix is not None:
            matrix_values = params.density_matrix.tolist()

        matrix_def = None
        if matrix_display_name:
            clean_matrix_name = matrix_display_name.replace("preset: ", "")
            # 如果UI显示是“自定义”，则在预设中记录为 "custom"
            if clean_matrix_name == "自定义":
                clean_matrix_name = "custom"
            matrix_def = MatrixDefinition(name=clean_matrix_name, values=matrix_values)

        # 构造Curve Definition
        curve_def = None
        curve_display_name = self.parameter_panel.curve_editor.curve_combo.currentText()
        curve_points = params.curve_points
        
        if curve_display_name:
            clean_curve_name = curve_display_name.replace("preset: ", "")
            if clean_curve_name == "自定义":
                clean_curve_name = "custom"
            curve_def = CurveDefinition(name=clean_curve_name, points=curve_points)

        # 构造文件名、裁切和方向
        current_image = self.context.get_current_image()
        raw_file = Path(current_image.file_path).name if current_image else None
        # 裁切：优先使用当前激活的 crop；否则使用 contactsheet 的单裁剪（BWC）
        crop_rect = self.context.get_active_crop() or self.context.get_contactsheet_crop_rect()

        # 获取当前crop实例（包含独立orientation）
        crop_instance = self.context.get_active_crop_instance()
        
        return Preset(
            name=name,
            # Metadata
            raw_file=raw_file,
            orientation=self.context.get_current_orientation(),  # 全局orientation
            crop=crop_rect,  # 向后兼容（single/原图裁剪）
            film_type=self.parameter_panel.get_current_film_type(),  # 胶片类型
            # 新的多裁剪结构（包含crop的独立orientation）
            crops=(
                [crop_instance.to_dict()]
                if crop_instance is not None else None
            ),
            active_crop_id=(crop_instance.id if crop_instance is not None else None),
            # Input Transformation
            input_transformation=cs_def,
            # Grading Parameters
            grading_params=params.to_dict(),
            density_matrix=matrix_def,
            density_curve=curve_def,
        )

    def _save_preset(self):
        """保存当前设置为预设文件"""
        if not self.context.get_current_image():
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
            preset = self._create_preset_from_current_state(preset_name)

            # 保存预设
            PresetManager.save_preset(preset, file_path)
            self.statusBar().showMessage(f"预设已保存: {preset.name}")

        except (IOError, KeyError) as e:
            QMessageBox.critical(self, "保存预设失败", str(e))

    def _select_input_color_space(self):
        """选择输入色彩变换"""
        from PySide6.QtWidgets import QInputDialog
        
        # 获取可用的色彩空间列表
        available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        # 显示选择对话框
        color_space, ok = QInputDialog.getItem(
            self, 
            "选择输入色彩变换", 
            "请选择图像的输入色彩变换:", 
            available_spaces, 
            available_spaces.index(self.context.get_input_color_space()) if self.context.get_input_color_space() in available_spaces else 0, 
            False
        )
        
        if ok and color_space:
            try:
                self.context.set_input_color_space(color_space)
                # 更新状态栏
                self.statusBar().showMessage(f"已设置输入色彩变换: {color_space}")
                # 如果已经有图像，重新处理
                if self.context.get_current_image():
                    self.context._reload_with_color_space()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置色彩空间失败: {str(e)}")
    
    def _reload_with_icc(self):
        """使用新的ICC配置文件重新加载图像"""
        if not self.context.get_current_image():
            return
            
        try:
            # 重新生成代理（使用统一配置中的代理尺寸）
            self.context.current_proxy = self.context.image_manager.generate_proxy(
                self.context.current_image,
                self.context.the_enlarger.preview_config.get_proxy_size_tuple()
            )
            
            # 应用ICC配置文件
            if self.context.input_icc_profile:
                self.context.current_proxy = self.context.color_space_manager.apply_icc_profile_to_image(
                    self.context.current_proxy, self.context.input_icc_profile
                )
            
            # 转换到工作色彩空间
            source_color_space = self.context.current_proxy.color_space
            self.context.current_proxy = self.context.color_space_manager.convert_to_working_space(
                self.context.current_proxy, source_color_space
            )
            
            # 重新生成小代理（如果需要）
            proxy_size = self.context.the_enlarger.preview_config.get_proxy_size_tuple()
            
            self.context.current_proxy = self.context.image_manager.generate_proxy(self.context.current_proxy, proxy_size)
            
            # 更新预览
            self.context._trigger_preview_update()
            
            # 自动适应窗口大小
            self.preview_widget.fit_to_window()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新处理图像失败: {str(e)}")
    
    def _reload_with_color_space(self):
        """使用新的色彩空间重新加载图像"""
        if not self.context.get_current_image():
            return
        
        try:
            # 重新生成代理（使用统一配置中的代理尺寸）
            self.context.current_proxy = self.context.image_manager.generate_proxy(
                self.context.current_image,
                self.context.the_enlarger.preview_config.get_proxy_size_tuple()
            )
            
            # 设置新的色彩空间
            self.context.current_proxy = self.context.color_space_manager.set_image_color_space(
                self.context.current_proxy, self.context.get_input_color_space()
            )
            
            # 转换到工作色彩空间
            self.context.current_proxy = self.context.color_space_manager.convert_to_working_space(
                self.context.current_proxy
            )
            
            # 生成更小的代理图像用于实时预览（统一读取自PreviewConfig）
            proxy_size = self.context.the_enlarger.preview_config.get_proxy_size_tuple()
            
            self.context.current_proxy = self.context.image_manager.generate_proxy(self.context.current_proxy, proxy_size)
            
            # 重新处理预览
            self.context._trigger_preview_update()
            
            # 自动适应窗口大小
            self.preview_widget.fit_to_window()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新加载图像失败: {str(e)}")

    # ===== Spectral Sharpening Hooks =====
    def _on_ccm_optimize_requested(self):
        """根据色卡执行光谱锐化优化（后台），更新输入色彩空间与参数。"""
        current_image = self.context.get_current_image()
        if not (current_image and current_image.array is not None):
            QMessageBox.warning(self, "提示", "请先打开一张图片")
            return
        # 获取当前输入空间 gamma（若取不到，退化为1.0）
        cs_name = self.context.get_input_color_space()
        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
        input_gamma = float(cs_info.get("gamma", 1.0))

        # 取色卡角点
        cc_corners = getattr(self.preview_widget, 'cc_corners', None)
        if not cc_corners or len(cc_corners) != 4:
            QMessageBox.information(self, "提示", "请在预览中启用色卡选择器并设置四角点")
            return

        # 取当前密度校正矩阵
        params = self.context.get_current_params()
        use_mat = bool(params.enable_density_matrix)
        corr_mat = params.density_matrix if (use_mat and params.density_matrix is not None) else None

        self.statusBar().showMessage("正在根据色卡优化光谱锐化参数...（CMA-ES）")

        # 在UI线程直接调用会卡顿。这里用QRunnable封装，复用全局线程池。
        class _CCMWorker(QRunnable):
            def __init__(self, image_array, corners, gamma, use_mat, mat):
                super().__init__()
                self.image_array = image_array
                self.corners = corners
                self.gamma = gamma
                self.use_mat = use_mat
                self.mat = mat
                self.result = None
                self.error = None
            @Slot()
            def run(self):
                try:
                    self.result = run_spectral_sharpening(
                        self.image_array,
                        self.corners,
                        self.gamma,
                        self.use_mat,
                        self.mat,
                        optimizer_max_iter=120,
                        optimizer_tolerance=1e-6,
                    )
                except Exception as e:
                    import traceback
                    self.error = f"{e}\n{traceback.format_exc()}"

        worker = _CCMWorker(current_image.array, cc_corners, input_gamma, use_mat, corr_mat)

        def _on_done():
            try:
                if worker.error:
                    QMessageBox.critical(self, "优化失败", worker.error)
                    self.statusBar().showMessage("光谱锐化优化失败")
                    return
                res = worker.result or {}
                params_dict = res.get('parameters', {})
                primaries_xy = np.asarray(params_dict.get('primaries_xy'), dtype=float)
                if primaries_xy is None or primaries_xy.shape != (3, 2):
                    QMessageBox.warning(self, "结果无效", "未获得有效的基色坐标")
                    self.statusBar().showMessage("光谱锐化优化完成但结果无效")
                    return

                # 注册并切换到自定义输入色彩空间
                base_name = cs_name.replace("_custom", "").replace("_preset", "")
                custom_name = f"{base_name}_custom"
                self.context.color_space_manager.register_custom_colorspace(custom_name, primaries_xy, None, gamma=1.0)
                # 使用专用入口以便重建代理
                self.context.set_input_color_space(custom_name)

                # 应用其他参数更新
                new_params = self.context.get_current_params().copy()
                # 密度参数与RB对数增益
                new_params.density_gamma = float(params_dict.get('gamma', new_params.density_gamma))
                new_params.density_dmax = float(params_dict.get('dmax', new_params.density_dmax))
                r_gain = float(params_dict.get('r_gain', new_params.rgb_gains[0]))
                b_gain = float(params_dict.get('b_gain', new_params.rgb_gains[2]))
                new_params.rgb_gains = (r_gain, new_params.rgb_gains[1], b_gain)

                self.context.update_params(new_params)
                self.statusBar().showMessage(
                    f"光谱锐化完成：RMSE={float(res.get('rmse', 0.0)):.4f}, 已应用到自定义输入色彩变换"
                )
            finally:
                pass

        # 轮询检测任务完成（简化型）。
        def _poll():
            if getattr(worker, 'result', None) is not None or getattr(worker, 'error', None) is not None:
                _on_done(); return
            QTimer.singleShot(150, _poll)

        self.context.thread_pool.start(worker)
        QTimer.singleShot(150, _poll)

    def _on_save_custom_colorspace_requested(self, primaries_dict: dict):
        """保存 UCS 三角的基色坐标为输入色彩变换 JSON（用户目录）。"""
        try:
            # primaries_dict: {'R': (x,y), 'G': (x,y), 'B': (x,y)}
            name_base = self.context.get_input_color_space().replace("_custom", "").replace("_preset", "")
            save_name = f"{name_base}_custom"
            data = {
                "name": save_name,
                "primaries": {
                    "R": [float(primaries_dict['R'][0]), float(primaries_dict['R'][1])],
                    "G": [float(primaries_dict['G'][0]), float(primaries_dict['G'][1])],
                    "B": [float(primaries_dict['B'][0]), float(primaries_dict['B'][1])],
                },
                # 采用D65与 gamma=1.0（扫描线性）
                "white_point": [0.3127, 0.3290],
                "gamma": 1.0,
            }
            ok = enhanced_config_manager.save_user_config("colorspace", save_name, data)
            if ok:
                self.statusBar().showMessage(f"已保存输入色彩变换到用户目录: {save_name}.json")
            else:
                QMessageBox.warning(self, "保存失败", "无法保存到用户配置目录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存输入色彩变换失败: {str(e)}")
    
    def _initialize_color_space_info(self):
        """初始化色彩空间信息"""
        try:
            # 验证默认色彩空间
            if self.context.color_space_manager.validate_color_space(self.context.get_input_color_space()):
                self.statusBar().showMessage(f"已设置默认输入色彩变换: {self.context.get_input_color_space()}")
            else:
                # 如果默认色彩空间无效，使用第一个可用的
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
                if available_spaces:
                    self.context.set_input_color_space(available_spaces[0])
                    self.statusBar().showMessage(f"默认色彩空间无效，使用: {self.context.get_input_color_space()}")
                else:
                    print("错误: 没有可用的色彩空间")
        except Exception as e:
            print(f"初始化色彩空间信息失败: {str(e)}")
    
    def _save_image(self):
        """保存图像"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 检测当前是否为B&W模式
        current_film_type = self.context.get_current_film_type()
        is_bw_mode = self.context.film_type_controller.is_monochrome_type(current_film_type)
        
        # 根据B&W模式获取合适的色彩空间列表
        if is_bw_mode:
            available_spaces = self.context.color_space_manager.get_grayscale_colorspaces()
            # 如果没有灰度色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        else:
            available_spaces = self.context.color_space_manager.get_color_colorspaces()
            # 如果没有彩色色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        # 打开保存设置对话框
        save_dialog = SaveImageDialog(self, available_spaces, is_bw_mode)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # 获取保存设置
        settings = save_dialog.get_settings()
        
        self._execute_save(settings, force_dialog=True) # 强制弹出另存为
    
    def _save_image_as(self):
        """“另存为”图像"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 检测当前是否为B&W模式
        current_film_type = self.context.get_current_film_type()
        is_bw_mode = self.context.film_type_controller.is_monochrome_type(current_film_type)
        
        # 根据B&W模式获取合适的色彩空间列表
        if is_bw_mode:
            available_spaces = self.context.color_space_manager.get_grayscale_colorspaces()
            # 如果没有灰度色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        else:
            available_spaces = self.context.color_space_manager.get_color_colorspaces()
            # 如果没有彩色色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        save_dialog = SaveImageDialog(self, available_spaces, is_bw_mode)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        settings = save_dialog.get_settings()
        self._execute_save(settings, force_dialog=True)

    def _execute_save(self, settings: dict, force_dialog: bool = False):
        """执行保存操作"""
        current_image = self.context.get_current_image()
        file_path = current_image.file_path if current_image else None
        
        if force_dialog or not file_path:
            extension = ".tiff" if settings["format"] == "tiff" else ".jpg"
            filter_str = "TIFF文件 (*.tiff *.tif)" if settings["format"] == "tiff" else "JPEG文件 (*.jpg *.jpeg)"
            original_filename = Path(current_image.file_path).stem if current_image and current_image.file_path else "untitled"

            # 模式判断：single / contactsheet(single crop) / contactsheet(all)
            save_mode = settings.get("save_mode", "single")
            base_dir = str(Path(current_image.file_path).parent) if current_image and current_image.file_path else ""

            # 计算编号/命名
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            has_contactsheet_single = (self.context.get_contactsheet_crop_rect() is not None and (not crops or active_id is None))

            def _default_name_single():
                # single 模式：CC-原文件名
                return f"CC-{original_filename}{extension}"

            def _default_name_contactsheet_all():
                # contactsheet“保存所有”：基名（没有编号），后续批量时加 -[两位数]
                return f"CC-{original_filename}{extension}"

            def _default_name_contactsheet_single():
                # contactsheet 的接触印像：固定中文后缀
                if active_id and crops:
                    # 若为正式裁剪聚焦，仍按编号命名
                    for i, c in enumerate(crops, start=1):
                        if getattr(c, 'id', None) == active_id:
                            return f"CC-{original_filename}-{i:02d}{extension}"
                # 非正式单裁剪：接触印像
                return f"CC-{original_filename}-接触印像{extension}"

            if save_mode == 'all':
                # 仅选择“目录”和“基名”，不真正返回 file_path（批量保存时逐个拼接）
                base_choice = QFileDialog.getExistingDirectory(self, "选择保存目录", base_dir)
                if not base_choice:
                    return
                # 询问基名（可选），默认 CC-原文件名
                default_basename = f"CC-{original_filename}"
                from PySide6.QtWidgets import QInputDialog
                basename, ok = QInputDialog.getText(self, "保存所有", "文件基名（将自动加 -[编号] 与扩展名）:", text=default_basename)
                if not ok or not basename:
                    basename = default_basename
                # 执行批量保存
                self._execute_batch_save(settings, base_choice, basename, extension)
                return
            else:
                # 保存单张
                if crops and active_id:
                    default_filename = _default_name_contactsheet_single()
                elif has_contactsheet_single:
                    default_filename = _default_name_contactsheet_single()
                else:
                    default_filename = _default_name_single()

                default_path = str(Path(base_dir) / default_filename) if base_dir else default_filename
                file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", default_path, filter_str)
        
        if not file_path:
            return
            
        enhanced_config_manager.set_directory("save_image", str(Path(file_path).parent))
            
        try:
            # 根据保存模式确定裁剪与方向
            save_mode = settings.get("save_mode", "single")
            crop_instance = self.context.get_active_crop_instance()
            rect_norm = None
            orientation = self.context.get_current_orientation()
            if save_mode == 'single':
                if crop_instance is not None:
                    rect_norm = crop_instance.rect_norm
                    orientation = crop_instance.orientation
                elif self.context.get_contactsheet_crop_rect() is not None:
                    rect_norm = self.context.get_contactsheet_crop_rect()
                    orientation = self.context.get_current_orientation()
            # 应用裁剪与旋转
            final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)

            # 重要：将原图转换到工作色彩空间，保持与预览一致
            print(f"导出前的色彩空间转换:")
            print(f"  原始图像色彩空间: {final_image.color_space}")
            print(f"  输入色彩变换设置: {self.context.get_input_color_space()}")
            
            # 先设置输入色彩变换
            working_image = self.context.color_space_manager.set_image_color_space(
                final_image, self.context.get_input_color_space()
            )
            # 前置IDT Gamma（导出走高精度pow）
            try:
                cs_name = self.context.get_input_color_space()
                cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                idt_gamma = float(cs_info.get("gamma", 1.0))
            except Exception:
                idt_gamma = 1.0
            if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                    working_image.array, idt_gamma, use_optimization=False
                )
                working_image = working_image.copy_with_new_array(arr)
            # 转到工作色彩空间（ACEScg），跳过逆伽马
            working_image = self.context.color_space_manager.convert_to_working_space(
                working_image, skip_gamma_inverse=True
            )
            print(f"  转换后工作色彩空间: {working_image.color_space}")
            
            # 导出模式：提升为float64精度，确保全程高精度计算
            if working_image.array is not None:
                working_image.array = working_image.array.astype(np.float64)
                working_image.dtype = np.float64
            
            # 应用调色参数到工作空间的图像（根据设置决定是否包含曲线）
            # 导出必须使用全精度（禁用低精度LUT）+ 分块并行
            result_image = self.context.the_enlarger.apply_full_pipeline(
                working_image,
                self.context.get_current_params(),
                include_curve=settings["include_curve"],
                for_export=True
            )
            
            # 转换到输出色彩空间
            result_image = self.context.color_space_manager.convert_to_display_space(
                result_image, settings["color_space"]
            )
            
            # Convert to grayscale for B&W film types
            result_image = self._convert_to_grayscale_if_bw_mode(result_image)
            
            # 根据扩展名与设置计算"有效位深"
            ext = str(Path(file_path).suffix).lower()
            requested_bit_depth = int(settings.get("bit_depth", 8))
            if ext in [".jpg", ".jpeg"]:
                effective_bit_depth = 8
            elif ext in [".png", ".tif", ".tiff"]:
                effective_bit_depth = 16 if requested_bit_depth == 16 else 8
            else:
                effective_bit_depth = requested_bit_depth

            # 保存图像
            self.context.image_manager.save_image(
                result_image,
                file_path,
                bit_depth=effective_bit_depth,
                quality=95,
                export_color_space=settings.get("color_space")
            )
            
            self.statusBar().showMessage(
                f"图像已保存: {Path(file_path).name} "
                f"({effective_bit_depth}bit, {settings['color_space']})"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")

    def _execute_batch_save(self, settings: dict, target_dir: str, basename: str, extension: str):
        """执行批量保存（保存所有裁剪）
        - 命名：{basename}-[两位编号]{extension}
        - 顺序：按当前 crops 列表顺序
        - 若没有任何裁剪但存在 contactsheet 单裁剪，视为一张（编号01）
        """
        try:
            crops = self.context.get_all_crops()
            if crops:
                for i, crop in enumerate(crops, start=1):
                    # 切到该裁剪 Profile（不聚焦，避免视图闪烁），并以该裁剪的 orientation 导出
                    self.context.switch_to_crop(crop.id)
                    # 处理图像（复用 _execute_save 的核心管道，但不弹对话框）
                    filename = f"{basename}-{i:02d}{extension}"
                    file_path = str(Path(target_dir) / filename)
                    # 复用单张保存流程：构造一个“强制路径”，跳过另存弹窗
                    tmp_settings = dict(settings)
                    # 临时将 force_dialog 置 False 并直接走保存
                    # 下面直接复制 _execute_save 后半段的处理流程：
                    current_image = self.context.get_current_image()
                    crop_instance = self.context.get_active_crop_instance()
                    rect_norm = crop_instance.rect_norm if crop_instance is not None else None
                    orientation = crop_instance.orientation if crop_instance is not None else self.context.get_current_orientation()
                    final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)
                    working_image = self.context.color_space_manager.set_image_color_space(
                        final_image, self.context.get_input_color_space()
                    )
                    # 前置IDT Gamma
                    try:
                        cs_name = self.context.get_input_color_space()
                        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                        idt_gamma = float(cs_info.get("gamma", 1.0))
                    except Exception:
                        idt_gamma = 1.0
                    if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                        arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                            working_image.array, idt_gamma, use_optimization=False
                        )
                        working_image = working_image.copy_with_new_array(arr)
                    working_image = self.context.color_space_manager.convert_to_working_space(
                        working_image, skip_gamma_inverse=True
                    )
                    result_image = self.context.the_enlarger.apply_full_pipeline(
                        working_image,
                        self.context.get_current_params(),
                        include_curve=settings["include_curve"],
                        for_export=True
                    )
                    result_image = self.context.color_space_manager.convert_to_display_space(
                        result_image, settings["color_space"]
                    )
                    # Convert to grayscale for B&W film types
                    result_image = self._convert_to_grayscale_if_bw_mode(result_image)
                    # 有效位深
                    ext = extension.lower()
                    requested_bit_depth = int(settings.get("bit_depth", 8))
                    if ext in [".jpg", ".jpeg"]:
                        effective_bit_depth = 8
                    elif ext in [".png", ".tif", ".tiff"]:
                        effective_bit_depth = 16 if requested_bit_depth == 16 else 8
                    else:
                        effective_bit_depth = requested_bit_depth
                    # 保存
                    self.context.image_manager.save_image(
                        result_image,
                        file_path,
                        bit_depth=effective_bit_depth,
                        quality=95,
                        export_color_space=settings.get("color_space")
                    )
                self.statusBar().showMessage(f"已保存所有裁剪到: {target_dir}")
            else:
                # 无正式裁剪：若存在 contactsheet 单裁剪，视为一张（编号01）
                if self.context.get_contactsheet_crop_rect() is not None:
                    filename = f"{basename}-01{extension}"
                    file_path = str(Path(target_dir) / filename)
                    # 直接复用 _execute_save：构造一次弹窗路径
                    # 为保持简单，这里复用保存单张路径
                    # 保存图像（复制单张保存处理）：
                    current_image = self.context.get_current_image()
                    rect_norm = self.context.get_contactsheet_crop_rect()
                    orientation = self.context.get_current_orientation()
                    final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)
                    working_image = self.context.color_space_manager.set_image_color_space(
                        final_image, self.context.get_input_color_space()
                    )
                    # 前置IDT Gamma
                    try:
                        cs_name = self.context.get_input_color_space()
                        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                        idt_gamma = float(cs_info.get("gamma", 1.0))
                    except Exception:
                        idt_gamma = 1.0
                    if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                        arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                            working_image.array, idt_gamma, use_optimization=False
                        )
                        working_image = working_image.copy_with_new_array(arr)
                    working_image = self.context.color_space_manager.convert_to_working_space(
                        working_image, skip_gamma_inverse=True
                    )
                    result_image = self.context.the_enlarger.apply_full_pipeline(
                        working_image,
                        self.context.get_current_params(),
                        include_curve=settings["include_curve"],
                        for_export=True
                    )
                    result_image = self.context.color_space_manager.convert_to_display_space(
                        result_image, settings["color_space"]
                    )
                    # Convert to grayscale for B&W film types
                    result_image = self._convert_to_grayscale_if_bw_mode(result_image)
                    # 有效位深
                    ext = extension.lower()
                    requested_bit_depth = int(settings.get("bit_depth", 8))
                    if ext in [".jpg", ".jpeg"]:
                        effective_bit_depth = 8
                    elif ext in [".png", ".tif", ".tiff"]:
                        effective_bit_depth = 16 if requested_bit_depth == 16 else 8
                    else:
                        effective_bit_depth = requested_bit_depth
                    # 保存
                    self.context.image_manager.save_image(
                        result_image,
                        file_path,
                        bit_depth=effective_bit_depth,
                        quality=95,
                        export_color_space=settings.get("color_space")
                    )
                    self.statusBar().showMessage(f"已保存: {Path(file_path).name}")
                else:
                    self.statusBar().showMessage("没有需要保存的裁剪")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存所有失败: {e}")
    
    def _reset_parameters(self):
        """重置调色参数"""
        self.context.reset_params()
    
    def _load_default_curves(self):
        """加载默认曲线（Kodak Endura Paper）"""
        # 逻辑已迁移或将在Context中重新实现
        pass
    
    def _toggle_original_view(self, checked: bool):
        """切换原始图像视图"""
        if checked:
            # 显示原始图像
            if self.context.get_current_image():
                # TODO: 需要从Context获取原始代理图像
                # self.preview_widget.set_image(self.current_proxy)
                pass
        else:
            # 显示调色后的图像
            self.context._trigger_preview_update()
    
    def _reset_view(self):
        """重置预览视图"""
        self.preview_widget.reset_view()
        self.statusBar().showMessage("视图已重置")
    

    
    def _estimate_film_type(self):
        """估算胶片类型"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "警告", "没有加载的图像")
            return
        
        try:
            # self.grading_engine is not defined in this file.
            # Assuming it's meant to be self.the_enlarger.grading_engine
            # or that it should be initialized if not already.
            # For now, commenting out as it's not defined.
            # film_type = self.grading_engine.estimate_film_type(self.current_image)
            # QMessageBox.information(self, "胶片类型", f"估算的胶片类型: {film_type}")
            pass # Placeholder for actual estimation logic
        except Exception as e:
            QMessageBox.critical(self, "错误", f"估算胶片类型失败: {str(e)}")
    
    def _open_file_classification_manager(self):
        """打开文件分类规则管理器"""
        try:
            from divere.standalone_tools.launcher import launch_file_classification_manager
            self.file_classification_manager = launch_file_classification_manager(self)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件分类规则管理器: {str(e)}")
    
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
        """此方法现在由Context的信号触发，或直接调用Context的方法"""
        self.context._trigger_preview_update()

    def _on_preview_updated(self, result_image: ImageData):
        self.preview_widget.set_image(result_image)
        if self._fit_after_next_preview:
            try:
                # 确保图像设置完成后再适应窗口
                from PySide6.QtCore import QTimer
                QTimer.singleShot(10, self.preview_widget.fit_to_window)
            finally:
                self._fit_after_next_preview = False
        # 更新工具可用性
        try:
            self._update_apply_contactsheet_enabled()
        except Exception:
            pass

    def _update_apply_contactsheet_enabled(self):
        """仅在 contact sheet 模式下、进入单张 crop 聚焦时才显示该按钮。"""
        try:
            focused = bool(getattr(self.context, '_crop_focused', False))
            kind = self.context.get_current_profile_kind()
            # 只有当 kind == 'crop' 且聚焦时，显示“沿用接触印像设置”
            visible = (kind == 'crop' and focused)
            try:
                self._apply_contactsheet_action.setVisible(visible)
            except Exception:
                pass
        except Exception:
            pass

    def _open_config_manager(self):
        """打开配置管理器"""
        from divere.ui.config_manager_dialog import ConfigManagerDialog
        dialog = ConfigManagerDialog(self)
        dialog.exec()

    # ===== 裁剪：UI协调槽 =====
    def _on_crop_committed(self, rect_norm: tuple):
        """处理新建裁剪"""
        try:
            # 不论当前 Profile，点击“+”后的裁剪都视为“新增 crop”
            orientation = self.context.get_current_orientation()
            crop_id = self.context.add_crop(rect_norm, orientation)
            if crop_id:
                # 切换到该 crop 的 profile（不自动聚焦）
                self.context.switch_to_crop(crop_id)
                # 刷新裁剪选择条（强制显示编号）
                try:
                    self.preview_widget._hide_single_crop_selector = False
                except Exception:
                    pass
                crops = self.context.get_all_crops()
                active_id = self.context.get_active_crop_id()
                self.preview_widget.refresh_crop_selector(crops, active_id)
                # 更新参数面板
                self.parameter_panel.on_context_params_changed(self.context.get_current_params())
        except Exception as e:
            print(f"创建裁剪失败: {e}")
    
    def _on_single_crop_committed(self, rect_norm: tuple):
        """处理单张裁剪：不创建正式crop项，仅在 contactsheet 上记录裁剪并显示 overlay。"""
        try:
            # 设置 contactsheet 裁剪（新方法）
            if hasattr(self.context, 'set_contactsheet_crop'):
                self.context.set_contactsheet_crop(rect_norm)
            else:
                # 回退：直接通过 crop_changed 信号驱动 overlay
                self.preview_widget.set_crop_overlay(rect_norm)
            # 刷新选择条：当仅存在 single 裁剪时隐藏编号
            try:
                self.preview_widget._hide_single_crop_selector = True
                crops = self.context.get_all_crops()
                active_id = self.context.get_active_crop_id()
                self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            except Exception:
                pass
        except Exception as e:
            print(f"创建单张裁剪失败: {e}")
    
    def _on_crop_updated(self, crop_id_or_rect, rect_norm=None):
        """处理更新现有裁剪"""
        try:
            # 兼容两种调用方式
            if isinstance(crop_id_or_rect, str):
                # 新格式: (crop_id, rect_norm)
                crop_id = crop_id_or_rect
                if rect_norm:
                    # 更新指定裁剪
                    for crop in self.context._crops:
                        if crop.id == crop_id:
                            crop.rect_norm = rect_norm
                            break
                    self.context._autosave_timer.start()
            else:
                # 旧格式: (rect_norm)
                self.context.update_active_crop(crop_id_or_rect)
        except Exception as e:
            print(f"更新裁剪失败: {e}")

    def _on_request_focus_crop(self, crop_id=None):
        """处理聚焦裁剪请求"""
        try:
            if crop_id:
                # 一次性切到指定裁剪并聚焦，避免先显示原图再聚焦的闪烁
                self.context.switch_to_crop_focused(crop_id)
            else:
                self.context.focus_on_active_crop()
            # 刷新UI
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=True)
            # 进入聚焦后等预览更新完成再适应窗口
            self._fit_after_next_preview = True
            # 更新工具可用性
            try:
                self._update_apply_contactsheet_enabled()
            except Exception:
                pass
        except Exception as e:
            print(f"聚焦裁剪失败: {e}")

    def _on_request_restore_crop(self):
        try:
            self.context.restore_crop_preview()
            # 恢复到原图模式后，刷新选择条与显示状态
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            # 恢复到原图需要等预览更新完成再适应窗口
            self._fit_after_next_preview = True
            # 更新工具可见性/可用性
            self._update_apply_contactsheet_enabled()
        except Exception:
            pass

    def _on_request_focus_contactsheet(self):
        """进入接触印像/单张裁剪聚焦。"""
        try:
            # 若 Context 尚未记录 contactsheet 裁剪，但元数据里有 overlay，则先回写一份
            try:
                img = self.preview_widget.get_current_image_data()
                if img and img.metadata:
                    rect = img.metadata.get('crop_overlay')
                    if rect and getattr(self.context, '_contactsheet_crop_rect', None) is None:
                        if hasattr(self.context, 'set_contactsheet_crop'):
                            self.context.set_contactsheet_crop(tuple(rect))
            except Exception:
                pass

            self.context.focus_on_contactsheet_crop()
            # 刷新UI
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=True)
            self._fit_after_next_preview = True
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"接触印像聚焦失败: {e}")

    def _on_custom_primaries_changed(self, primaries_xy: dict):
        """当用户在 UCS 三角拖动完成后，基于 primaries_xy 注册并切换到临时输入空间。
        遵循单向数据流：通过 Context 的 set_input_color_space 触发代理重建与预览更新。
        """
        try:
            # 规范输入为 (3,2) 数组顺序 R,G,B
            arr = np.array([primaries_xy['R'], primaries_xy['G'], primaries_xy['B']], dtype=float)
            base_name = self.context.get_input_color_space().replace("_custom", "").replace("_preset", "")
            temp_name = f"{base_name}_custom"
            # 注册/覆盖临时空间（gamma=1.0，白点D65）
            self.context.color_space_manager.register_custom_colorspace(temp_name, arr, None, gamma=1.0)
            # 切换输入色彩变换（Context 内部会重建代理并刷新预览）
            self.context.set_input_color_space(temp_name)
            # 不修改其他调色参数，仅切换输入空间
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "提示", f"应用临时基色失败: {e}")
            except Exception:
                pass
        
    def _toggle_profiling(self, enabled: bool):
        """切换预览Profiling"""
        self.context.the_enlarger.set_profiling_enabled(enabled)
        self.context.color_space_manager.set_profiling_enabled(enabled)
        self.statusBar().showMessage("预览Profiling已开启" if enabled else "预览Profiling已关闭")
    
    def on_parameter_changed(self):
        """参数改变时的回调"""
        new_params = self.parameter_panel.get_current_params()
        self.context.update_params(new_params)
    
    def on_input_colorspace_changed(self, space_name: str):
        """输入色彩空间改变时的回调 - 需要特殊处理以重建代理"""
        self.context.set_input_color_space(space_name)
    
    def on_film_type_changed(self, film_type: str):
        """胶片类型改变时的回调"""
        # Check if transitioning from color to B&W (data loss warning)
        old_film_type = self.context.get_current_film_type()
        old_is_mono = self.context.film_type_controller.is_monochrome_type(old_film_type)
        new_is_mono = self.context.film_type_controller.is_monochrome_type(film_type)
        
        # Show warning dialog if transitioning from color to B&W
        if not old_is_mono and new_is_mono:
            if not self._confirm_color_to_bw_transition():
                # User cancelled - revert film type selection in UI
                self.parameter_panel.set_film_type(old_film_type)
                return
        
        # Proceed with film type change
        self.context.set_current_film_type(film_type)

    def on_context_film_type_changed(self, film_type: str):
        """ApplicationContext胶片类型改变时的回调 - 同步UI"""
        # Update film type dropdown
        self.parameter_panel.set_film_type(film_type)
        
        # Apply neutralization for B&W film types immediately
        # This prevents the preview flash issue when loading B&W presets
        self._apply_bw_neutralization_if_needed(film_type)
    
    def _apply_bw_neutralization_if_needed(self, film_type: str):
        """Apply neutral values for B&W film types"""
        # Check if this is a B&W film type
        if not self.context.film_type_controller.is_monochrome_type(film_type):
            return
        
        # Get current parameters
        params = self.context.get_current_params()
        
        # Set neutral values for B&W mode
        # 1. Set IDT color transformation to identity
        params.input_color_space_name = "Identity"  # This should result in identity transform
        
        # 2. Set RGB gains to (0.0, 0.0, 0.0)
        params.rgb_gains = (0.0, 0.0, 0.0)
        
        # 3. Set density matrix to identity
        params.density_matrix = np.eye(3)
        params.density_matrix_name = "Identity"
        
        # 4. Set RGB curves to Ilford Multigrade 2 (proper B&W curve)
        # Load the curve from the curve manager if available, otherwise use linear fallback
        try:
            # Get the curve from configuration
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            curve_files = enhanced_config_manager.get_config_files("curves")
            ilford_curve = None
            
            for curve_file in curve_files:
                if "Ilford MGFB 2" in str(curve_file) or "Ilford_MGFB_2" in str(curve_file):
                    curve_data = enhanced_config_manager.load_config_file(curve_file)
                    if curve_data and "curves" in curve_data:
                        # Use RGB curve if available
                        if "RGB" in curve_data["curves"]:
                            ilford_curve = curve_data["curves"]["RGB"]
                        break
            
            # Apply Ilford curve if found, otherwise use linear
            if ilford_curve:
                params.curve_points = ilford_curve
                params.curve_points_r = curve_data["curves"].get("R", [(0.0, 0.0), (1.0, 1.0)])
                params.curve_points_g = curve_data["curves"].get("G", [(0.0, 0.0), (1.0, 1.0)])
                params.curve_points_b = curve_data["curves"].get("B", [(0.0, 0.0), (1.0, 1.0)])
                params.density_curve_name = "Ilford MGFB 2"
            else:
                # Fallback to linear
                linear_curve = [(0.0, 0.0), (1.0, 1.0)]
                params.curve_points_r = linear_curve
                params.curve_points_g = linear_curve
                params.curve_points_b = linear_curve
                params.curve_points = linear_curve
                params.density_curve_name = "linear"
                
        except Exception as e:
            print(f"Warning: Could not load Ilford Multigrade 2 curve, using linear: {e}")
            # Fallback to linear
            linear_curve = [(0.0, 0.0), (1.0, 1.0)]
            params.curve_points_r = linear_curve
            params.curve_points_g = linear_curve
            params.curve_points_b = linear_curve
            params.curve_points = linear_curve
            params.density_curve_name = "linear"
        
        # Apply the neutralized parameters
        self.context.update_params(params)
        
        # Trigger auto-save after B&W override
        self.context.autosave_requested.emit()

    def _confirm_color_to_bw_transition(self) -> bool:
        """Confirm color to B&W transition with user"""
        from PySide6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 
            "胶片类型变更确认", 
            "切换到黑白模式将覆盖当前的彩色校正设置：\n"
            "• IDT色彩变换将设为单位矩阵\n"
            "• 密度矩阵将设为单位矩阵\n" 
            "• RGB增益将设为 (0, 0, 0)\n"
            "• RGB曲线将设为 Ilford Multigrade 2\n\n"
            "当前的彩色校正信息将丢失。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        return reply == QMessageBox.StandardButton.Yes

    def get_current_params(self) -> ColorGradingParams:
        """获取当前调色参数"""
        return self.context.get_current_params()
    
    def set_current_params(self, params: ColorGradingParams):
        """设置当前调色参数"""
        self.context.update_params(params)
    
    def _on_image_rotated(self, direction):
        """处理图像旋转：委托给 ApplicationContext 维护朝向与预览更新"""
        try:
            self.context.rotate(int(direction))
        except Exception:
            pass
    
    def _on_lut_export_requested(self, lut_type: str, file_path: str, size: int):
        """处理LUT导出请求"""
        try:
            # 获取当前参数
            current_params = self.context.get_current_params()
            
            if lut_type == "input_transform":
                success = self._export_input_transform_lut(current_params, file_path, size)
            elif lut_type == "color_correction":
                success = self._export_color_correction_lut(current_params, file_path, size)
            elif lut_type == "density_curve":
                success = self._export_density_curve_lut(current_params, file_path, size)
            else:
                success = False
                
            if success:
                self.statusBar().showMessage(f"{lut_type} LUT已导出到: {file_path}")
            else:
                self.statusBar().showMessage(f"{lut_type} LUT导出失败")
                
        except Exception as e:
            print(f"LUT导出失败: {e}")
            self.statusBar().showMessage(f"LUT导出失败: {e}")
    
    def _export_input_transform_lut(self, params, file_path: str, size: int) -> bool:
        """导出输入设备转换LUT (3D) - 包含IDT Gamma + 色彩空间转换"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 获取当前IDT Gamma值
            try:
                cs_name = self.context.get_input_color_space()
                cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                idt_gamma = float(cs_info.get("gamma", 1.0))
            except Exception:
                idt_gamma = 1.0
            
            # 创建专门的输入设备转换配置
            idt_config = {
                "idt_gamma": idt_gamma,
                "context": self.context,
                "input_colorspace_name": params.input_color_space_name
            }
            
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_input_device_transform_lut(
                idt_config, file_path, size
            )
            
        except Exception as e:
            print(f"导出输入转换LUT失败: {e}")
            return False
    
    def _export_color_correction_lut(self, params, file_path: str, size: int) -> bool:
        """导出反相校色LUT (3D, 不含密度曲线)"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 创建不含密度曲线的参数副本
            color_params = params.copy()
            color_params.enable_density_curve = False
            color_params.curve_points = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_r = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_g = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_b = [(0.0, 0.0), (1.0, 1.0)]
            
            # 管线配置
            pipeline_config = {
                "params": color_params,
                "context": self.context,
                "the_enlarger": self.context.the_enlarger
            }
            
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_pipeline_lut(
                pipeline_config, file_path, "3D", size
            )
            
        except Exception as e:
            print(f"导出反相校色LUT失败: {e}")
            return False
    
    def _export_density_curve_lut(self, params, file_path: str, size: int) -> bool:
        """导出密度曲线LUT (1D) - 在密度空间应用曲线"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 提取曲线数据
            curves = {
                'R': params.curve_points_r or [(0.0, 0.0), (1.0, 1.0)],
                'G': params.curve_points_g or [(0.0, 0.0), (1.0, 1.0)],
                'B': params.curve_points_b or [(0.0, 0.0), (1.0, 1.0)]
            }
            
            # 如果有RGB通用曲线，使用它
            if params.curve_points and params.curve_points != [(0.0, 0.0), (1.0, 1.0)]:
                curves['R'] = curves['G'] = curves['B'] = params.curve_points
            
            # 使用密度曲线专用方法（包含屏幕反光补偿）
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_density_curve_lut(
                curves, file_path, size, params.screen_glare_compensation
            )
            
        except Exception as e:
            print(f"导出密度曲线LUT失败: {e}")
            return False
    
    def _test_lut_chain_consistency(self) -> None:
        """测试三个LUT串联的数学一致性（调试功能）"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            params = self.context.get_current_params()
            lut_interface = DiVERELUTInterface()
            
            # 验证数学一致性（使用小样本快速测试）
            stats = lut_interface.validate_lut_chain_consistency(
                self.context, params, "", "", "", test_samples=50
            )
            
            if 'error' in stats:
                print(f"LUT链验证出错: {stats['error']}")
                return
                
            print("=== LUT链数学一致性验证结果 ===")
            print(f"测试样本数: {stats['samples_tested']}")
            print(f"最大绝对误差: {stats['max_abs_error']:.6f}")
            print(f"平均绝对误差: {stats['mean_abs_error']:.6f}")
            print(f"最大相对误差: {stats['max_rel_error']:.6f}")
            print(f"平均相对误差: {stats['mean_rel_error']:.6f}")
            print(f"RMSE: {stats['rmse']:.6f}")
            
            # 简单判断数学一致性
            if stats['max_abs_error'] < 1e-3 and stats['mean_abs_error'] < 1e-4:
                print("✅ 数学一致性良好！")
                self.statusBar().showMessage("LUT链数学一致性验证通过")
            else:
                print("⚠️ 检测到数学不一致，可能需要进一步调试")
                self.statusBar().showMessage("LUT链存在数学不一致问题")
                
        except Exception as e:
            print(f"LUT一致性测试失败: {e}")
            import traceback
            traceback.print_exc()
        
# 移除 Worker 相关类定义
# class _PreviewWorkerSignals(QObject): ...
# class _PreviewWorker(QRunnable): ...
