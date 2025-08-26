"""
参数面板
包含所有调色参数的控件
"""

from typing import Optional, Tuple
import numpy as np
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QComboBox,
    QGroupBox, QPushButton, QCheckBox, QTabWidget,
    QScrollArea, QMessageBox, QInputDialog, QFileDialog
)
from PySide6.QtCore import Qt, Signal

from divere.core.data_types import ColorGradingParams, Preset
from divere.core.app_context import ApplicationContext
from divere.ui.curve_editor_widget import CurveEditorWidget
from divere.ui.ucs_triangle_widget import UcsTriangleWidget
from divere.core.color_space import xy_to_uv, uv_to_xy
from divere.utils.enhanced_config_manager import enhanced_config_manager


class ParameterPanel(QWidget):
    """参数面板 (重构版)"""
    
    parameter_changed = Signal()
    auto_color_requested = Signal()
    auto_color_iterative_requested = Signal()
    input_colorspace_changed = Signal(str)

    # Signals for complex actions requiring coordination
    ccm_optimize_requested = Signal()
    save_custom_colorspace_requested = Signal(dict)
    toggle_color_checker_requested = Signal(bool)
    # 色卡变换信号
    cc_flip_horizontal_requested = Signal()
    cc_flip_vertical_requested = Signal()
    cc_rotate_left_requested = Signal()
    cc_rotate_right_requested = Signal()
    # 新增：基色(primaries)改变（拖动结束时触发，负担轻）
    custom_primaries_changed = Signal(dict)
    # LUT导出信号
    lut_export_requested = Signal(str, str, int)  # (lut_type, file_path, size)
    
    def __init__(self, context: ApplicationContext):
        super().__init__()
        self.context = context
        self.current_params = self.context.get_current_params().copy()
        
        self._is_updating_ui = False
        self.context.params_changed.connect(self.on_context_params_changed)
        
        self._create_ui()
        self._connect_signals()
        
    def on_context_params_changed(self, params: ColorGradingParams):
        """当Context中的参数改变时，更新UI"""
        self.current_params = params.copy()
        self.update_ui_from_params()

    def initialize_defaults(self, initial_params: ColorGradingParams):
        """由主窗口调用，在加载图像后设置并应用默认参数"""
        self.current_params = initial_params.copy()
        self.update_ui_from_params()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        tab_widget = QTabWidget()
        tab_widget.addTab(self._create_basic_tab(), "输入色彩科学")
        tab_widget.addTab(self._create_density_tab(), "密度与矩阵")
        tab_widget.addTab(self._create_rgb_tab(), "RGB曝光")
        tab_widget.addTab(self._create_curve_tab(), "密度曲线")
        tab_widget.addTab(self._create_debug_tab(), "管线控制")
        
        content_layout.addWidget(tab_widget)
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

    def _create_basic_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        colorspace_group = QGroupBox("输入色彩变换")
        colorspace_layout = QGridLayout(colorspace_group)
        # IDT Gamma（在下拉菜单上方）
        self.idt_gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.idt_gamma_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.idt_gamma_slider, self.idt_gamma_spinbox, 50, 280, 0.5, 2.8, 0.01)
        colorspace_layout.addWidget(QLabel("IDT Gamma:"), 0, 0)
        colorspace_layout.addWidget(self.idt_gamma_slider, 0, 1)
        colorspace_layout.addWidget(self.idt_gamma_spinbox, 0, 2)
        self.input_colorspace_combo = QComboBox()
        spaces = self.context.color_space_manager.get_available_color_spaces()
        for space in spaces:
            self.input_colorspace_combo.addItem(space, space)
        colorspace_layout.addWidget(QLabel("色彩空间:"), 1, 0)
        colorspace_layout.addWidget(self.input_colorspace_combo, 1, 1, 1, 2)
        layout.addWidget(colorspace_group)
        
        # --- Spectral Sharpening Section ---
        self.enable_scanner_spectral_checkbox = QCheckBox("扫描仪光谱锐化")
        layout.addWidget(self.enable_scanner_spectral_checkbox)

        self.ucs_widget = UcsTriangleWidget()
        self.ucs_widget.setVisible(False)
        layout.addWidget(self.ucs_widget)

        # 色卡选择器和变换按钮的水平布局
        cc_selector_layout = QHBoxLayout()
        self.cc_selector_checkbox = QCheckBox("色卡选择器")
        self.cc_selector_checkbox.setVisible(False)
        cc_selector_layout.addWidget(self.cc_selector_checkbox)
        
        # 色卡变换按钮
        self.cc_flip_h_button = QPushButton("↔")
        self.cc_flip_h_button.setToolTip("水平翻转色卡选择器")
        self.cc_flip_h_button.setFixedWidth(30)
        self.cc_flip_h_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_flip_h_button)
        
        self.cc_flip_v_button = QPushButton("↕")
        self.cc_flip_v_button.setToolTip("竖直翻转色卡选择器")
        self.cc_flip_v_button.setFixedWidth(30)
        self.cc_flip_v_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_flip_v_button)
        
        self.cc_rotate_l_button = QPushButton("↶")
        self.cc_rotate_l_button.setToolTip("左旋转色卡选择器")
        self.cc_rotate_l_button.setFixedWidth(30)
        self.cc_rotate_l_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_rotate_l_button)
        
        self.cc_rotate_r_button = QPushButton("↷")
        self.cc_rotate_r_button.setToolTip("右旋转色卡选择器")
        self.cc_rotate_r_button.setFixedWidth(30)
        self.cc_rotate_r_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_rotate_r_button)
        
        cc_selector_layout.addStretch()  # 推到左边
        layout.addLayout(cc_selector_layout)

        self.ccm_optimize_button = QPushButton("根据色卡计算光谱锐化转换")
        self.ccm_optimize_button.setToolTip("从色卡选择器读取24个颜色并优化参数")
        self.ccm_optimize_button.setVisible(False)
        self.ccm_optimize_button.setEnabled(False)
        layout.addWidget(self.ccm_optimize_button)

        self.save_input_colorspace_button = QPushButton("保存输入色彩变换结果")
        self.save_input_colorspace_button.setToolTip("将当前UCS三角形对应的基色与白点保存为JSON文件")
        self.save_input_colorspace_button.setVisible(False)
        layout.addWidget(self.save_input_colorspace_button)

        layout.addStretch()
        return widget

    def _create_density_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        inversion_group = QGroupBox("密度反相")
        inversion_layout = QGridLayout(inversion_group)
        self.density_gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_gamma_spinbox = QDoubleSpinBox()
        self.density_dmax_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_dmax_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.density_gamma_slider, self.density_gamma_spinbox, 50, 400, 0.5, 4.0, 0.01)
        self._setup_slider_spinbox(self.density_dmax_slider, self.density_dmax_spinbox, 0, 480, 0.0, 4.8, 0.01)
        inversion_layout.addWidget(QLabel("密度反差:"), 0, 0)
        inversion_layout.addWidget(self.density_gamma_slider, 0, 1)
        inversion_layout.addWidget(self.density_gamma_spinbox, 0, 2)
        inversion_layout.addWidget(QLabel("最大密度:"), 1, 0)
        inversion_layout.addWidget(self.density_dmax_slider, 1, 1)
        inversion_layout.addWidget(self.density_dmax_spinbox, 1, 2)
        layout.addWidget(inversion_group)

        matrix_group = QGroupBox("密度校正矩阵")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_editor_widgets = []
        matrix_grid = QGridLayout()
        for i in range(3):
            row = []
            for j in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-10.0, 10.0); spinbox.setSingleStep(0.01); spinbox.setDecimals(4); spinbox.setFixedWidth(80)
                matrix_grid.addWidget(spinbox, i, j)
                row.append(spinbox)
            self.matrix_editor_widgets.append(row)
        
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("预设:"))
        self.matrix_combo = QComboBox()
        self.matrix_combo.addItem("自定义", "custom")
        available = self.context.the_enlarger.pipeline_processor.get_available_matrices()
        for matrix_id in available:
            data = self.context.the_enlarger.pipeline_processor.get_matrix_data(matrix_id)
            if data: self.matrix_combo.addItem(data.get("name", matrix_id), matrix_id)
        combo_layout.addWidget(self.matrix_combo)
        matrix_layout.addLayout(combo_layout)
        matrix_layout.addLayout(matrix_grid)
        layout.addWidget(matrix_group)
        layout.addStretch()
        return widget

    def _create_rgb_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        rgb_group = QGroupBox("RGB曝光调整")
        rgb_layout = QGridLayout(rgb_group)
        self.red_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.red_gain_spinbox = QDoubleSpinBox()
        self.green_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.green_gain_spinbox = QDoubleSpinBox()
        self.blue_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.blue_gain_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.red_gain_slider, self.red_gain_spinbox, -300, 300, -3.0, 3.0, 0.01)
        self._setup_slider_spinbox(self.green_gain_slider, self.green_gain_spinbox, -300, 300, -3.0, 3.0, 0.01)
        self._setup_slider_spinbox(self.blue_gain_slider, self.blue_gain_spinbox, -300, 300, -3.0, 3.0, 0.01)
        rgb_layout.addWidget(QLabel("R:"), 0, 0); rgb_layout.addWidget(self.red_gain_slider, 0, 1); rgb_layout.addWidget(self.red_gain_spinbox, 0, 2)
        rgb_layout.addWidget(QLabel("G:"), 1, 0); rgb_layout.addWidget(self.green_gain_slider, 1, 1); rgb_layout.addWidget(self.green_gain_spinbox, 1, 2)
        rgb_layout.addWidget(QLabel("B:"), 2, 0); rgb_layout.addWidget(self.blue_gain_slider, 2, 1); rgb_layout.addWidget(self.blue_gain_spinbox, 2, 2)
        self.auto_color_single_button = QPushButton("AI自动校色（单次）")
        self.auto_color_multi_button = QPushButton("AI自动校色（多次）")
        rgb_layout.addWidget(self.auto_color_single_button, 3, 1)
        rgb_layout.addWidget(self.auto_color_multi_button, 3, 2)
        layout.addWidget(rgb_group)
        layout.addStretch()
        return widget

    def _create_curve_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.curve_editor = CurveEditorWidget()
        layout.addWidget(self.curve_editor)
        return widget

    def _create_debug_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        pipeline_group = QGroupBox("管道步骤控制")
        pipeline_layout = QVBoxLayout(pipeline_group)
        self.enable_density_inversion_checkbox = QCheckBox("启用密度反相")
        self.enable_density_matrix_checkbox = QCheckBox("启用密度矩阵")
        self.enable_rgb_gains_checkbox = QCheckBox("启用RGB增益")
        self.enable_density_curve_checkbox = QCheckBox("启用密度曲线")
        pipeline_layout.addWidget(self.enable_density_inversion_checkbox)
        pipeline_layout.addWidget(self.enable_density_matrix_checkbox)
        pipeline_layout.addWidget(self.enable_rgb_gains_checkbox)
        pipeline_layout.addWidget(self.enable_density_curve_checkbox)
        layout.addWidget(pipeline_group)
        
        # LUT导出组
        lut_group = QGroupBox("LUT导出")
        lut_layout = QVBoxLayout(lut_group)
        
        # 输入设备转换LUT (3D)
        input_lut_layout = QHBoxLayout()
        input_lut_layout.addWidget(QLabel("输入设备转换LUT (3D):"))
        input_lut_layout.addStretch()
        self.input_lut_size_combo = QComboBox()
        self.input_lut_size_combo.addItems(["16", "32", "64", "128"])
        self.input_lut_size_combo.setCurrentText("64")
        input_lut_layout.addWidget(self.input_lut_size_combo)
        self.export_input_lut_btn = QPushButton("导出")
        self.export_input_lut_btn.clicked.connect(self._on_export_input_lut)
        input_lut_layout.addWidget(self.export_input_lut_btn)
        lut_layout.addLayout(input_lut_layout)
        
        # 反相校色LUT (3D, 不含密度曲线)
        color_lut_layout = QHBoxLayout()
        color_lut_layout.addWidget(QLabel("反相校色LUT (3D):"))
        color_lut_layout.addStretch()
        self.color_lut_size_combo = QComboBox()
        self.color_lut_size_combo.addItems(["16", "32", "64", "128"])
        self.color_lut_size_combo.setCurrentText("64")
        color_lut_layout.addWidget(self.color_lut_size_combo)
        self.export_color_lut_btn = QPushButton("导出")
        self.export_color_lut_btn.clicked.connect(self._on_export_color_lut)
        color_lut_layout.addWidget(self.export_color_lut_btn)
        lut_layout.addLayout(color_lut_layout)
        
        # 密度曲线LUT (1D)
        curve_lut_layout = QHBoxLayout()
        curve_lut_layout.addWidget(QLabel("密度曲线LUT (1D):"))
        curve_lut_layout.addStretch()
        self.curve_lut_size_combo = QComboBox()
        self.curve_lut_size_combo.addItems(["2048", "4096", "8192", "16384", "32768", "65536"])
        self.curve_lut_size_combo.setCurrentText("4096")
        curve_lut_layout.addWidget(self.curve_lut_size_combo)
        self.export_curve_lut_btn = QPushButton("导出")
        self.export_curve_lut_btn.clicked.connect(self._on_export_curve_lut)
        curve_lut_layout.addWidget(self.export_curve_lut_btn)
        lut_layout.addLayout(curve_lut_layout)
        
        layout.addWidget(lut_group)
        layout.addStretch()
        return widget
    
    def _setup_slider_spinbox(self, slider, spinbox, s_min, s_max, sp_min, sp_max, sp_step):
        slider.setRange(s_min, s_max)
        spinbox.setRange(sp_min, sp_max)
        spinbox.setSingleStep(sp_step)
        spinbox.setDecimals(2)

    def _connect_signals(self):
        self.input_colorspace_combo.currentTextChanged.connect(self._on_input_colorspace_changed)
        # IDT Gamma联动
        self.idt_gamma_slider.valueChanged.connect(self._on_idt_gamma_slider_changed)
        self.idt_gamma_spinbox.valueChanged.connect(self._on_idt_gamma_spinbox_changed)
        
        self.density_gamma_slider.valueChanged.connect(self._on_gamma_slider_changed)
        self.density_gamma_spinbox.valueChanged.connect(self._on_gamma_spinbox_changed)
        self.density_dmax_slider.valueChanged.connect(self._on_dmax_slider_changed)
        self.density_dmax_spinbox.valueChanged.connect(self._on_dmax_spinbox_changed)
        
        self.red_gain_slider.valueChanged.connect(self._on_red_gain_slider_changed)
        self.red_gain_spinbox.valueChanged.connect(self._on_red_gain_spinbox_changed)
        self.green_gain_slider.valueChanged.connect(self._on_green_gain_slider_changed)
        self.green_gain_spinbox.valueChanged.connect(self._on_green_gain_spinbox_changed)
        self.blue_gain_slider.valueChanged.connect(self._on_blue_gain_slider_changed)
        self.blue_gain_spinbox.valueChanged.connect(self._on_blue_gain_spinbox_changed)

        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_combo_changed)
        for i in range(3):
            for j in range(3):
                # 统一由专用槽处理：必要时自动勾选“启用密度矩阵”，并触发参数变更
                self.matrix_editor_widgets[i][j].valueChanged.connect(self._on_matrix_value_changed)
        
        # 曲线编辑器发出 (curve_name, points)，使用专用槽以丢弃参数并统一触发
        self.curve_editor.curve_changed.connect(self._on_curve_changed)
        
        for checkbox in [self.enable_density_inversion_checkbox, self.enable_density_matrix_checkbox, 
                         self.enable_rgb_gains_checkbox, self.enable_density_curve_checkbox]:
            checkbox.toggled.connect(self._on_debug_step_changed)

        self.auto_color_single_button.clicked.connect(self.auto_color_requested.emit)
        self.auto_color_multi_button.clicked.connect(self.auto_color_iterative_requested.emit)

        # Spectral sharpening signals
        self.enable_scanner_spectral_checkbox.toggled.connect(self._on_scanner_spectral_toggled)
        # dragFinished(dict) → 触发参数变更，同时发出 primaries 改变
        self.ucs_widget.dragFinished.connect(lambda coords: self._on_ucs_drag_finished(coords))
        self.ucs_widget.resetPointRequested.connect(self._on_reset_point)
        self.cc_selector_checkbox.toggled.connect(self._on_cc_selector_toggled)
        self.cc_flip_h_button.clicked.connect(self._on_cc_flip_horizontal)
        self.cc_flip_v_button.clicked.connect(self._on_cc_flip_vertical)
        self.cc_rotate_l_button.clicked.connect(self._on_cc_rotate_left)
        self.cc_rotate_r_button.clicked.connect(self._on_cc_rotate_right)
        self.ccm_optimize_button.clicked.connect(self.ccm_optimize_requested.emit)
        self.save_input_colorspace_button.clicked.connect(self._on_save_input_colorspace_clicked)

    def update_ui_from_params(self):
        self._is_updating_ui = True
        try:
            params = self.current_params
            self._sync_combo_box(self.input_colorspace_combo, params.input_color_space_name)
            # 读取当前输入空间的gamma
            try:
                info = self.context.color_space_manager.get_color_space_info(params.input_color_space_name) or {}
                g = float(info.get('gamma', 1.0))
            except Exception:
                g = 1.0
            self.idt_gamma_slider.setValue(int(g * 100))
            self.idt_gamma_spinbox.setValue(g)
            
            self.density_gamma_slider.setValue(int(params.density_gamma * 100))
            self.density_gamma_spinbox.setValue(params.density_gamma)
            self.density_dmax_slider.setValue(int(params.density_dmax * 100))
            self.density_dmax_spinbox.setValue(params.density_dmax)
            
            self.red_gain_slider.setValue(int(params.rgb_gains[0] * 100))
            self.red_gain_spinbox.setValue(params.rgb_gains[0])
            self.green_gain_slider.setValue(int(params.rgb_gains[1] * 100))
            self.green_gain_spinbox.setValue(params.rgb_gains[1])
            self.blue_gain_slider.setValue(int(params.rgb_gains[2] * 100))
            self.blue_gain_spinbox.setValue(params.rgb_gains[2])
            
            matrix = params.density_matrix if params.density_matrix is not None else np.eye(3)
            for i in range(3):
                for j in range(3):
                    self.matrix_editor_widgets[i][j].setValue(matrix[i,j])
            self._sync_combo_box(self.matrix_combo, params.density_matrix_name)

            curves = {'RGB': params.curve_points, 'R': params.curve_points_r, 'G': params.curve_points_g, 'B': params.curve_points_b}
            # 避免在拖动过程中反复重置内部曲线与选择状态：当曲线内容未变化时跳过写回
            try:
                current_curves = self.curve_editor.get_all_curves()
                if not self._curves_equal(current_curves, curves):
                    self.curve_editor.set_all_curves(curves)
            except Exception:
                self.curve_editor.set_all_curves(curves)
            self._sync_combo_box(self.curve_editor.curve_combo, params.density_curve_name)
            
            self.enable_density_inversion_checkbox.setChecked(params.enable_density_inversion)
            self.enable_density_matrix_checkbox.setChecked(params.enable_density_matrix)
            self.enable_rgb_gains_checkbox.setChecked(params.enable_rgb_gains)
            self.enable_density_curve_checkbox.setChecked(params.enable_density_curve)
        finally:
            self._is_updating_ui = False

    def _curves_equal(self, a: dict, b: dict) -> bool:
        try:
            keys = ('RGB','R','G','B')
            for k in keys:
                pa = a.get(k, []) if isinstance(a, dict) else []
                pb = b.get(k, []) if isinstance(b, dict) else []
                if len(pa) != len(pb):
                    return False
                for (xa, ya), (xb, yb) in zip(pa, pb):
                    if abs(float(xa) - float(xb)) > 1e-6 or abs(float(ya) - float(yb)) > 1e-6:
                        return False
            return True
        except Exception:
            return False

    def _sync_combo_box(self, combo: QComboBox, name: str):
        for i in range(combo.count()):
            if combo.itemData(i) == name or combo.itemText(i).strip('*') == name:
                combo.setCurrentIndex(i)
                return
        
        display_name = f"*{name}"
        combo.insertItem(0, display_name, name)
        combo.setCurrentIndex(0)

    def get_current_params(self) -> ColorGradingParams:
        params = ColorGradingParams()
        params.input_color_space_name = self.input_colorspace_combo.currentData() or self.input_colorspace_combo.currentText().strip('*')
        params.density_gamma = self.density_gamma_spinbox.value()
        params.density_dmax = self.density_dmax_spinbox.value()
        params.rgb_gains = (self.red_gain_spinbox.value(), self.green_gain_spinbox.value(), self.blue_gain_spinbox.value())
        
        matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = self.matrix_editor_widgets[i][j].value()
        params.density_matrix = matrix
        params.density_matrix_name = self.matrix_combo.currentData() or self.matrix_combo.currentText().strip('*')
        
        all_curves = self.curve_editor.get_all_curves()
        params.curve_points = all_curves.get('RGB', [])
        params.curve_points_r = all_curves.get('R', [])
        params.curve_points_g = all_curves.get('G', [])
        params.curve_points_b = all_curves.get('B', [])
        params.density_curve_name = self.curve_editor.curve_combo.currentData() or self.curve_editor.curve_combo.currentText().strip('*')
        
        params.enable_density_inversion = self.enable_density_inversion_checkbox.isChecked()
        params.enable_density_matrix = self.enable_density_matrix_checkbox.isChecked()
        params.enable_rgb_gains = self.enable_rgb_gains_checkbox.isChecked()
        params.enable_density_curve = self.enable_density_curve_checkbox.isChecked()
        return params

    # --- Internal sync slots ---
    def _on_gamma_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.density_gamma_spinbox.blockSignals(True)
        self.density_gamma_spinbox.setValue(value / 100.0)
        self.density_gamma_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_gamma_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.density_gamma_slider.blockSignals(True)
        self.density_gamma_slider.setValue(int(value * 100))
        self.density_gamma_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_dmax_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.density_dmax_spinbox.blockSignals(True)
        self.density_dmax_spinbox.setValue(value / 100.0)
        self.density_dmax_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_dmax_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.density_dmax_slider.blockSignals(True)
        self.density_dmax_slider.setValue(int(value * 100))
        self.density_dmax_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_red_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.red_gain_spinbox.blockSignals(True)
        self.red_gain_spinbox.setValue(value / 100.0)
        self.red_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_red_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.red_gain_slider.blockSignals(True)
        self.red_gain_slider.setValue(int(value * 100))
        self.red_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_green_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.green_gain_spinbox.blockSignals(True)
        self.green_gain_spinbox.setValue(value / 100.0)
        self.green_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_green_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.green_gain_slider.blockSignals(True)
        self.green_gain_slider.setValue(int(value * 100))
        self.green_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_blue_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.blue_gain_spinbox.blockSignals(True)
        self.blue_gain_spinbox.setValue(value / 100.0)
        self.blue_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_blue_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.blue_gain_slider.blockSignals(True)
        self.blue_gain_slider.setValue(int(value * 100))
        self.blue_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    # --- Action slots ---
    def _on_input_colorspace_changed(self, space_name: str):
        """当输入色彩空间改变时，更新UCS Diagram和IDT Gamma"""
        try:
            # 移除星号标记
            clean_name = space_name.strip('*')
            
            # 更新IDT Gamma
            cs_info = self.context.color_space_manager.get_color_space_info(clean_name) or {}
            gamma = float(cs_info.get("gamma", 1.0))
            
            # 更新UI（避免触发信号循环）
            self._is_updating_ui = True
            self.idt_gamma_slider.setValue(int(gamma * 100))
            self.idt_gamma_spinbox.setValue(gamma)
            self._is_updating_ui = False
            
            # 更新UCS Diagram以反映新的色彩空间基色
            if 'primaries' in cs_info:
                primaries = cs_info['primaries']
                # 转换xy坐标到uv坐标
                coords_uv = {}
                for i, key in enumerate(['R', 'G', 'B']):
                    if i < len(primaries):
                        x, y = primaries[i]
                        u, v = xy_to_uv(x, y)
                        coords_uv[key] = (u, v)
                
                # 更新UCS Diagram
                if len(coords_uv) == 3:
                    self.ucs_widget.set_uv_coordinates(coords_uv)
                    
        except Exception as e:
            print(f"更新色彩空间失败: {e}")
            pass
        
        # 发出色彩空间特定变更信号，触发专用处理路径
        clean_name = space_name.strip('*')
        self.input_colorspace_changed.emit(clean_name)

    def _on_matrix_combo_changed(self, index: int):
        if self._is_updating_ui: return
        matrix_id = self.matrix_combo.itemData(index)
        if matrix_id and matrix_id != "custom":
            matrix = self.context.the_enlarger.pipeline_processor.get_density_matrix_array(matrix_id)
            if matrix is not None:
                self._is_updating_ui = True
                try:
                    for i in range(3):
                        for j in range(3):
                            self.matrix_editor_widgets[i][j].setValue(float(matrix[i, j]))
                    # 选择了预设矩阵，默认自动启用
                    self.enable_density_matrix_checkbox.setChecked(True)
                finally:
                    self._is_updating_ui = False
        self.parameter_changed.emit()

    # --- IDT Gamma slots ---
    def _on_idt_gamma_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.idt_gamma_spinbox.blockSignals(True)
        self.idt_gamma_spinbox.setValue(value / 100.0)
        self.idt_gamma_spinbox.blockSignals(False)
        self._apply_idt_gamma_to_colorspace()

    def _on_idt_gamma_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.idt_gamma_slider.blockSignals(True)
        self.idt_gamma_slider.setValue(int(value * 100))
        self.idt_gamma_slider.blockSignals(False)
        self._apply_idt_gamma_to_colorspace()

    def _apply_idt_gamma_to_colorspace(self):
        """将UI中的IDT Gamma写入当前输入色彩空间的内存定义，并重建代理。"""
        try:
            g = float(self.idt_gamma_spinbox.value())
            g = max(0.5, min(2.8, g))
            space_name = self.input_colorspace_combo.currentData() or self.input_colorspace_combo.currentText().strip('*')
            self.context.color_space_manager.update_color_space_gamma(space_name, g)
            # 触发Context按当前色彩空间重建proxy（内部会skip逆伽马，并应用前置幂次）
            self.context._prepare_proxy(); self.context._trigger_preview_update()
        except Exception:
            pass
    
    def _on_scanner_spectral_toggled(self, checked: bool):
        self.ucs_widget.setVisible(checked)
        self.cc_selector_checkbox.setVisible(checked)
        self.cc_flip_h_button.setVisible(checked)
        self.cc_flip_v_button.setVisible(checked)
        self.cc_rotate_l_button.setVisible(checked)
        self.cc_rotate_r_button.setVisible(checked)
        self.ccm_optimize_button.setVisible(checked)
        self.save_input_colorspace_button.setVisible(checked)
        if checked:
            self._on_reset_point('R')
            self._on_reset_point('G')
            self._on_reset_point('B')
            
    def _apply_ucs_coords(self):
        if self._is_updating_ui: return
        self.parameter_changed.emit()

    def _on_ucs_drag_finished(self, coords_uv: dict):
        """UCS拖动结束：发出无参参数变更，并将 primaries (xy) 以字典形式广播给上层。"""
        try:
            if not isinstance(coords_uv, dict):
                coords_uv = self.ucs_widget.get_uv_coordinates()
            # 转换到 xy
            primaries_xy = {}
            for key in ("R", "G", "B"):
                if key in coords_uv:
                    u, v = coords_uv[key]
                    x, y = uv_to_xy(u, v)
                    primaries_xy[key] = (float(x), float(y))
            # 发出两类信号：UI参数变更 + 基色更新
            self.parameter_changed.emit()
            if len(primaries_xy) == 3:
                self.custom_primaries_changed.emit(primaries_xy)
        except Exception:
            # 即使转换失败，也保持无参的参数变更触发
            self.parameter_changed.emit()

    def _on_reset_point(self, key: str):
        space = self.input_colorspace_combo.currentText().strip('*')
        info = self.context.color_space_manager.get_color_space_info(space)
        if info and 'primaries' in info:
            prim = np.array(info['primaries'], dtype=float)
            idx = {'R': 0, 'G': 1, 'B': 2}.get(key, 0)
            u, v = xy_to_uv(prim[idx, 0], prim[idx, 1])
            self.ucs_widget.set_uv_coordinates({key: (u, v)})

    def _on_cc_selector_toggled(self, checked: bool):
        self.ccm_optimize_button.setEnabled(checked)
        self.toggle_color_checker_requested.emit(checked)

    def _on_save_input_colorspace_clicked(self):
        coords_uv = self.ucs_widget.get_uv_coordinates()
        if not all(k in coords_uv for k in ("R", "G", "B")):
            QMessageBox.warning(self, "警告", "没有可保存的基色坐标。")
            return
            
        primaries = {k: uv_to_xy(*v) for k, v in coords_uv.items()}
        self.save_custom_colorspace_requested.emit(primaries)

    def _on_cc_flip_horizontal(self):
        """水平翻转色卡选择器"""
        self.cc_flip_horizontal_requested.emit()
    
    def _on_cc_flip_vertical(self):
        """竖直翻转色卡选择器"""
        self.cc_flip_vertical_requested.emit()
    
    def _on_cc_rotate_left(self):
        """左旋转色卡选择器"""
        self.cc_rotate_left_requested.emit()
    
    def _on_cc_rotate_right(self):
        """右旋转色卡选择器"""
        self.cc_rotate_right_requested.emit()

    def _on_curve_changed(self, curve_name, points):
        if self._is_updating_ui: return
        self.parameter_changed.emit()

    def _on_matrix_value_changed(self, *args):
        """矩阵单元格改动：必要时自动勾选启用，并触发参数变更。"""
        if self._is_updating_ui:
            return
        if not self.enable_density_matrix_checkbox.isChecked():
            # 用户开始编辑矩阵时，自动启用矩阵
            self.enable_density_matrix_checkbox.setChecked(True)
        self.parameter_changed.emit()
        
    def _on_debug_step_changed(self):
        if self._is_updating_ui: return
        self.parameter_changed.emit()
        
    def _on_auto_color_single_clicked(self):
        if self._is_updating_ui: return
        self.auto_color_requested.emit()

    def _on_auto_color_correct_clicked(self):
        if self._is_updating_ui: return
        self.auto_color_iterative_requested.emit()
    
    def _on_export_input_lut(self):
        """导出输入设备转换LUT"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出输入设备转换LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.input_lut_size_combo.currentText())
            self.lut_export_requested.emit("input_transform", file_path, size)
    
    def _on_export_color_lut(self):
        """导出反相校色LUT（不含密度曲线）"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出反相校色LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.color_lut_size_combo.currentText())
            self.lut_export_requested.emit("color_correction", file_path, size)
    
    def _on_export_curve_lut(self):
        """导出密度曲线LUT"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出密度曲线LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.curve_lut_size_combo.currentText())
            self.lut_export_requested.emit("density_curve", file_path, size)
