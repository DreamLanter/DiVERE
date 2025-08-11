"""
参数面板
包含所有调色参数的控件
"""

from typing import Optional
import numpy as np
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QPushButton, QCheckBox, QTabWidget,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer

from divere.core.data_types import ColorGradingParams
from divere.ui.curve_editor_widget import CurveEditorWidget
from divere.utils.enhanced_config_manager import enhanced_config_manager


class ParameterPanel(QWidget):
    """参数面板 (重构版)"""
    
    parameter_changed = Signal()
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_params = ColorGradingParams()
        self._is_updating_ui = False
        
        # 自动校色迭代状态
        self._auto_color_iteration = 0
        self._auto_color_max_iterations = 3
        self._auto_color_total_gains = np.zeros(3)

        self._create_ui()
        self._connect_signals()
        
    def initialize_defaults(self):
        """由主窗口调用，在加载图像后设置并应用默认参数"""
        self._sync_ui_defaults_to_params()
        self.current_params.density_gamma = 2.6
        self.current_params.correction_matrix_file = "Cineon_States_M_to_Print_Density"
        self.current_params.enable_correction_matrix = True
        self.current_params.enable_density_inversion = True
        self.current_params.enable_rgb_gains = True
        self.current_params.enable_density_curve = True
        
        # 设置默认曲线为Kodak Endura Paper
        self._load_default_curves()
        
        self.update_ui_from_params()
        self.parameter_changed.emit()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        tab_widget = QTabWidget()
        tab_widget.addTab(self._create_basic_tab(), "输入色彩管理")
        tab_widget.addTab(self._create_density_tab(), "密度与矩阵")
        tab_widget.addTab(self._create_rgb_tab(), "RGB曝光")
        tab_widget.addTab(self._create_curve_tab(), "密度曲线")
        tab_widget.addTab(self._create_debug_tab(), "管线控制及LUT")
        
        content_layout.addWidget(tab_widget)
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

    def _create_basic_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        colorspace_group = QGroupBox("输入色彩空间")
        colorspace_layout = QGridLayout(colorspace_group)
        self.input_colorspace_combo = QComboBox()
        if hasattr(self.main_window, 'color_space_manager'):
            spaces = self.main_window.color_space_manager.get_available_color_spaces()
            self.input_colorspace_combo.addItems(spaces)
            default = self.main_window.color_space_manager.get_default_color_space()
            if default in spaces:
                self.input_colorspace_combo.setCurrentText(default)
        colorspace_layout.addWidget(QLabel("色彩空间:"), 0, 0)
        colorspace_layout.addWidget(self.input_colorspace_combo, 0, 1)
        layout.addWidget(colorspace_group)
        
        # 添加说明文字框
        note_group = QGroupBox("重要说明")
        note_layout = QVBoxLayout(note_group)
        note_text = QLabel("Note: 当前版本的DiVERE用的色彩管理是石器时代手搓版，暂时没有读取照片ICC的能力，照片进来后会直接进行色彩空间的基色变换，这意味着要求扫描件数据的gamma=1。推荐的实践：用vuescan软件搭配平板扫描做gamma=1的tiff文件，并且不做任何额外的色彩管理。\n Epson、X5推荐用AdobeRGB_Linear\n Nikon扫描推荐用Film_KodakRGB_Linear\n 翻拍由于与Status M相差太大，色彩会很难看（正在尝试用colorchecker辨识密度矩阵的功能）")
        note_text.setWordWrap(True)  # 启用自动换行
        note_text.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 8px; background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; }")
        note_layout.addWidget(note_text)
        layout.addWidget(note_group)
        
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
        self._setup_slider_spinbox(self.density_gamma_slider, self.density_gamma_spinbox, 50, 400, 0.5, 4.0, 0.01, 260)
        self._setup_slider_spinbox(self.density_dmax_slider, self.density_dmax_spinbox, 0, 480, 0.0, 4.8, 0.01, 200)
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
        available = self.main_window.the_enlarger.get_available_matrices()
        for matrix_id in available:
            data = self.main_window.the_enlarger._load_correction_matrix(matrix_id)
            if data: self.matrix_combo.addItem(data.get("name", matrix_id), matrix_id)
        combo_layout.addWidget(self.matrix_combo)
        matrix_layout.addLayout(combo_layout)
        matrix_layout.addLayout(matrix_grid)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        reset_button = QPushButton("重置为单位矩阵")
        reset_button.clicked.connect(self._reset_matrix_to_identity)
        button_layout.addWidget(reset_button)
        
        save_button = QPushButton("保存矩阵")
        save_button.clicked.connect(self._save_matrix)
        button_layout.addWidget(save_button)
        
        matrix_layout.addLayout(button_layout)
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
        self._setup_slider_spinbox(self.red_gain_slider, self.red_gain_spinbox, -300, 300, -3.0, 3.0, 0.01, 0)
        self._setup_slider_spinbox(self.green_gain_slider, self.green_gain_spinbox, -300, 300, -3.0, 3.0, 0.01, 0)
        self._setup_slider_spinbox(self.blue_gain_slider, self.blue_gain_spinbox, -300, 300, -3.0, 3.0, 0.01, 0)
        rgb_layout.addWidget(QLabel("R:"), 0, 0); rgb_layout.addWidget(self.red_gain_slider, 0, 1); rgb_layout.addWidget(self.red_gain_spinbox, 0, 2)
        rgb_layout.addWidget(QLabel("G:"), 1, 0); rgb_layout.addWidget(self.green_gain_slider, 1, 1); rgb_layout.addWidget(self.green_gain_spinbox, 1, 2)
        rgb_layout.addWidget(QLabel("B:"), 2, 0); rgb_layout.addWidget(self.blue_gain_slider, 2, 1); rgb_layout.addWidget(self.blue_gain_spinbox, 2, 2)
        # AI自动校色按钮：单次与多次
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
        self.enable_correction_matrix_checkbox = QCheckBox("启用校正矩阵")
        self.enable_rgb_gains_checkbox = QCheckBox("启用RGB增益")
        self.enable_density_curve_checkbox = QCheckBox("启用密度曲线")
        pipeline_layout.addWidget(self.enable_density_inversion_checkbox)
        pipeline_layout.addWidget(self.enable_correction_matrix_checkbox)
        pipeline_layout.addWidget(self.enable_rgb_gains_checkbox)
        pipeline_layout.addWidget(self.enable_density_curve_checkbox)
        layout.addWidget(pipeline_group)
        
        # 添加LUT导出功能
        lut_group = QGroupBox("LUT导出")
        lut_layout = QVBoxLayout(lut_group)
        lut_layout.setSpacing(8)  # 设置垂直间距

        # 使用网格布局来确保对齐
        lut_grid = QGridLayout()
        lut_grid.setColumnStretch(0, 1)  # 按钮列可拉伸
        lut_grid.setColumnStretch(1, 0)  # 标签列固定宽度
        lut_grid.setColumnStretch(2, 0)  # 下拉框列固定宽度
        
        # 输入色彩管理LUT导出
        self.export_input_cc_lut_button = QPushButton("导出输入色彩管理LUT")
        self.export_input_cc_lut_button.setToolTip("生成包含输入色彩空间转换的3D LUT文件，将程序内置的输入色彩管理过程完全相同地应用于LUT")
        self.export_input_cc_lut_button.setMinimumHeight(30)  # 设置最小高度
        lut_grid.addWidget(self.export_input_cc_lut_button, 0, 0)
        
        # 输入色彩管理LUT size选择
        lut_size_label1 = QLabel("LUT Size:")
        lut_size_label1.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lut_grid.addWidget(lut_size_label1, 0, 1)
        
        self.input_cc_lut_size_combo = QComboBox()
        self.input_cc_lut_size_combo.addItems(["8", "16", "32", "48", "64", "96", "128"])
        self.input_cc_lut_size_combo.setCurrentText("64")  # 默认64
        self.input_cc_lut_size_combo.setFixedWidth(70)
        self.input_cc_lut_size_combo.setMinimumHeight(30)
        lut_grid.addWidget(self.input_cc_lut_size_combo, 0, 2)

        # 3D LUT导出
        self.export_3dlut_button = QPushButton("导出3D LUT (应用所有使能功能)")
        self.export_3dlut_button.setToolTip("生成包含所有使能调色功能的3D LUT文件，不包含输入色彩空间转换")
        self.export_3dlut_button.setMinimumHeight(30)  # 设置最小高度
        lut_grid.addWidget(self.export_3dlut_button, 1, 0)
        
        # 3D LUT size选择
        lut_size_label2 = QLabel("LUT Size:")
        lut_size_label2.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lut_grid.addWidget(lut_size_label2, 1, 1)
        
        self.lut_3d_size_combo = QComboBox()
        self.lut_3d_size_combo.addItems(["8", "16", "32", "48", "64", "96", "128"])
        self.lut_3d_size_combo.setCurrentText("64")  # 默认64
        self.lut_3d_size_combo.setFixedWidth(70)
        self.lut_3d_size_combo.setMinimumHeight(30)
        lut_grid.addWidget(self.lut_3d_size_combo, 1, 2)

        # 密度曲线1D LUT导出
        self.export_density_curve_1dlut_button = QPushButton("导出密度曲线1D LUT")
        self.export_density_curve_1dlut_button.setToolTip("生成包含RGB和单通道密度曲线的1D LUT文件")
        self.export_density_curve_1dlut_button.setMinimumHeight(30)  # 设置最小高度
        lut_grid.addWidget(self.export_density_curve_1dlut_button, 2, 0)
        
        # 密度曲线1D LUT size选择
        lut_size_label3 = QLabel("LUT Size:")
        lut_size_label3.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lut_grid.addWidget(lut_size_label3, 2, 1)
        
        self.density_curve_1dlut_size_combo = QComboBox()
        self.density_curve_1dlut_size_combo.addItems(["256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        self.density_curve_1dlut_size_combo.setCurrentText("65536")  # 默认65536
        self.density_curve_1dlut_size_combo.setFixedWidth(70)
        self.density_curve_1dlut_size_combo.setMinimumHeight(30)
        lut_grid.addWidget(self.density_curve_1dlut_size_combo, 2, 2)
        
        lut_layout.addLayout(lut_grid)
        
        layout.addWidget(lut_group)
        layout.addStretch()
        return widget
    
    def _setup_slider_spinbox(self, slider, spinbox, s_min, s_max, sp_min, sp_max, sp_step, s_default):
        slider.setRange(s_min, s_max); slider.setValue(s_default)
        spinbox.setRange(sp_min, sp_max); spinbox.setSingleStep(sp_step)
        # 根据步进值设置小数位数
        if sp_step == 0.01:
            spinbox.setDecimals(2)
        elif sp_step == 0.001:
            spinbox.setDecimals(3)
        else:
            spinbox.setDecimals(1)
        default_val = float(s_default)
        if sp_step < 1:
            default_val /= 100.0
        spinbox.setValue(default_val)

    def _connect_signals(self):
        self.input_colorspace_combo.currentTextChanged.connect(self._on_input_colorspace_changed)
        self.density_gamma_slider.valueChanged.connect(self._on_density_gamma_changed)
        self.density_gamma_spinbox.valueChanged.connect(self._on_density_gamma_changed)
        self.density_dmax_slider.valueChanged.connect(self._on_density_dmax_changed)
        self.density_dmax_spinbox.valueChanged.connect(self._on_density_dmax_changed)
        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_combo_changed)
        for i in range(3):
            for j in range(3): self.matrix_editor_widgets[i][j].valueChanged.connect(self._on_matrix_editor_changed)
        self.red_gain_slider.valueChanged.connect(self._on_red_gain_changed)
        self.red_gain_spinbox.valueChanged.connect(self._on_red_gain_changed)
        self.green_gain_slider.valueChanged.connect(self._on_green_gain_changed)
        self.green_gain_spinbox.valueChanged.connect(self._on_green_gain_changed)
        self.blue_gain_slider.valueChanged.connect(self._on_blue_gain_changed)
        self.blue_gain_spinbox.valueChanged.connect(self._on_blue_gain_changed)
        # 绑定AI自动校色按钮
        self.auto_color_single_button.clicked.connect(self._on_auto_color_single_clicked)
        self.auto_color_multi_button.clicked.connect(self._on_auto_color_correct_clicked)
        self.curve_editor.curve_changed.connect(self._on_curve_changed)

        self.enable_density_inversion_checkbox.toggled.connect(self._on_debug_step_changed)
        self.enable_correction_matrix_checkbox.toggled.connect(self._on_debug_step_changed)
        self.enable_rgb_gains_checkbox.toggled.connect(self._on_debug_step_changed)
        self.enable_density_curve_checkbox.toggled.connect(self._on_debug_step_changed)
        self.export_3dlut_button.clicked.connect(self._on_export_3dlut_clicked)
        self.export_input_cc_lut_button.clicked.connect(self._on_export_input_cc_lut_clicked)
        self.export_density_curve_1dlut_button.clicked.connect(self._on_export_density_curve_1dlut_clicked)

    def update_ui_from_params(self):
        self._is_updating_ui = True
        try:
            params = self.current_params
            self.density_gamma_slider.setValue(int(float(params.density_gamma) * 100))
            self.density_gamma_spinbox.setValue(float(params.density_gamma))
            self.density_dmax_slider.setValue(int(float(params.density_dmax) * 100))
            self.density_dmax_spinbox.setValue(float(params.density_dmax))
            
            matrix_id = params.correction_matrix_file if params.correction_matrix_file else ""
            index = self.matrix_combo.findData(matrix_id)
            self.matrix_combo.setCurrentIndex(index if index >= 0 else 0)
            
            matrix_to_display = np.eye(3)
            if matrix_id == "custom" and params.correction_matrix is not None:
                matrix_to_display = params.correction_matrix
            elif matrix_id:
                data = self.main_window.the_enlarger._load_correction_matrix(matrix_id)
                if data and "matrix" in data: matrix_to_display = np.array(data["matrix"])
            for i in range(3):
                for j in range(3): self.matrix_editor_widgets[i][j].setValue(float(matrix_to_display[i, j]))

            self.red_gain_slider.setValue(int(float(params.rgb_gains[0]) * 100))
            self.red_gain_spinbox.setValue(float(params.rgb_gains[0]))
            self.green_gain_slider.setValue(int(float(params.rgb_gains[1]) * 100))
            self.green_gain_spinbox.setValue(float(params.rgb_gains[1]))
            self.blue_gain_slider.setValue(int(float(params.rgb_gains[2]) * 100))
            self.blue_gain_spinbox.setValue(float(params.rgb_gains[2]))

            # 设置所有通道的曲线
            curves = {
                'RGB': params.curve_points,
                'R': getattr(params, 'curve_points_r', [(0.0, 0.0), (1.0, 1.0)]),
                'G': getattr(params, 'curve_points_g', [(0.0, 0.0), (1.0, 1.0)]),
                'B': getattr(params, 'curve_points_b', [(0.0, 0.0), (1.0, 1.0)])
            }
            self.curve_editor.set_all_curves(curves)
            
            if hasattr(self.curve_editor, 'curve_edit_widget'):
                self.curve_editor.curve_edit_widget.set_dmax(params.density_dmax)
                self.curve_editor.curve_edit_widget.set_gamma(params.density_gamma)
            


            self.enable_density_inversion_checkbox.setChecked(params.enable_density_inversion)
            self.enable_correction_matrix_checkbox.setChecked(params.enable_correction_matrix)
            self.enable_rgb_gains_checkbox.setChecked(params.enable_rgb_gains)
            self.enable_density_curve_checkbox.setChecked(params.enable_density_curve)
        finally:
            self._is_updating_ui = False

    def _sync_ui_defaults_to_params(self):
        self.current_params.density_gamma = self.density_gamma_spinbox.value()
        self.current_params.density_dmax = self.density_dmax_spinbox.value()
        self.current_params.rgb_gains = (self.red_gain_spinbox.value(), self.green_gain_spinbox.value(), self.blue_gain_spinbox.value())
        
        # 获取所有通道的曲线
        all_curves = self.curve_editor.get_all_curves()
        self.current_params.curve_points = all_curves.get('RGB', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_r = all_curves.get('R', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_g = all_curves.get('G', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_b = all_curves.get('B', [(0.0, 0.0), (1.0, 1.0)])

    def _on_input_colorspace_changed(self, space):
        if self._is_updating_ui: return
        self.main_window.input_color_space = space
        if self.main_window.current_image: self.main_window._reload_with_color_space()

    def _on_density_gamma_changed(self):
        if self._is_updating_ui: return
        val = self.density_gamma_slider.value() / 100.0 if self.sender() == self.density_gamma_slider else self.density_gamma_spinbox.value()
        self.current_params.density_gamma = float(val)
        self.update_ui_from_params()
        if hasattr(self, 'curve_editor'):
            self.curve_editor.curve_edit_widget.set_gamma(self.current_params.density_gamma)
        self.parameter_changed.emit()

    def _on_density_dmax_changed(self):
        if self._is_updating_ui: return
        val = self.density_dmax_slider.value() / 100.0 if self.sender() == self.density_dmax_slider else self.density_dmax_spinbox.value()
        self.current_params.density_dmax = float(val)
        self.update_ui_from_params()
        if hasattr(self, 'curve_editor'):
            self.curve_editor.curve_edit_widget.set_dmax(self.current_params.density_dmax)
        self.parameter_changed.emit()

    def _on_matrix_combo_changed(self):
        if self._is_updating_ui: return
        matrix_id = self.matrix_combo.currentData()
        self.current_params.correction_matrix_file = matrix_id
        self.current_params.enable_correction_matrix = bool(matrix_id)
        if matrix_id != "custom": self.current_params.correction_matrix = None
        self.update_ui_from_params(); self.parameter_changed.emit()

    def _on_matrix_editor_changed(self):
        if self._is_updating_ui: return
        matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3): matrix[i, j] = self.matrix_editor_widgets[i][j].value()
        self.current_params.correction_matrix = matrix
        self.current_params.correction_matrix_file = "custom"
        self.current_params.enable_correction_matrix = True
        
        # 直接设置下拉菜单为"自定义"，避免update_ui_from_params的复杂逻辑
        self._is_updating_ui = True
        self.matrix_combo.setCurrentIndex(0)  # "自定义"是第一个项
        self._is_updating_ui = False
        
        self.parameter_changed.emit()
    
    def _reset_matrix_to_identity(self):
        if self._is_updating_ui: return
        self.current_params.correction_matrix = np.eye(3, dtype=np.float32)
        self.current_params.correction_matrix_file = "custom"
        self.current_params.enable_correction_matrix = True
        
        # 直接设置下拉菜单为"自定义"，避免update_ui_from_params的复杂逻辑
        self._is_updating_ui = True
        self.matrix_combo.setCurrentIndex(0)  # "自定义"是第一个项
        self._is_updating_ui = False
        
        self.parameter_changed.emit()

    def _on_red_gain_changed(self):
        if self._is_updating_ui: return
        val = self.red_gain_slider.value() / 100.0 if self.sender() == self.red_gain_slider else self.red_gain_spinbox.value()
        self.current_params.rgb_gains = (float(val), self.current_params.rgb_gains[1], self.current_params.rgb_gains[2])
        self.update_ui_from_params(); self.parameter_changed.emit()

    def _on_green_gain_changed(self):
        if self._is_updating_ui: return
        val = self.green_gain_slider.value() / 100.0 if self.sender() == self.green_gain_slider else self.green_gain_spinbox.value()
        self.current_params.rgb_gains = (self.current_params.rgb_gains[0], float(val), self.current_params.rgb_gains[2])
        self.update_ui_from_params(); self.parameter_changed.emit()

    def _on_blue_gain_changed(self):
        if self._is_updating_ui: return
        val = self.blue_gain_slider.value() / 100.0 if self.sender() == self.blue_gain_slider else self.blue_gain_spinbox.value()
        self.current_params.rgb_gains = (self.current_params.rgb_gains[0], self.current_params.rgb_gains[1], float(val))
        self.update_ui_from_params(); self.parameter_changed.emit()

    def _on_curve_changed(self, curve_name: str, points: list):
        if self._is_updating_ui: return
        
        # 获取所有通道的曲线
        all_curves = self.curve_editor.get_all_curves()
        
        # 更新参数中的所有曲线
        self.current_params.curve_points = all_curves.get('RGB', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_r = all_curves.get('R', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_g = all_curves.get('G', [(0.0, 0.0), (1.0, 1.0)])
        self.current_params.curve_points_b = all_curves.get('B', [(0.0, 0.0), (1.0, 1.0)])
        
        # 设置enable标志
        self.current_params.enable_curve = len(self.current_params.curve_points) >= 2
        self.current_params.enable_curve_r = len(self.current_params.curve_points_r) >= 2 and self.current_params.curve_points_r != [(0.0, 0.0), (1.0, 1.0)]
        self.current_params.enable_curve_g = len(self.current_params.curve_points_g) >= 2 and self.current_params.curve_points_g != [(0.0, 0.0), (1.0, 1.0)]
        self.current_params.enable_curve_b = len(self.current_params.curve_points_b) >= 2 and self.current_params.curve_points_b != [(0.0, 0.0), (1.0, 1.0)]
        
        self.parameter_changed.emit()


    
    def _on_debug_step_changed(self):
        if self._is_updating_ui: return
        self.current_params.enable_density_inversion = self.enable_density_inversion_checkbox.isChecked()
        self.current_params.enable_correction_matrix = self.enable_correction_matrix_checkbox.isChecked()
        self.current_params.enable_rgb_gains = self.enable_rgb_gains_checkbox.isChecked()
        self.current_params.enable_density_curve = self.enable_density_curve_checkbox.isChecked()
        self.parameter_changed.emit()

    def _on_auto_color_single_clicked(self):
        """AI自动校色（单次）"""
        if self._is_updating_ui: return
        preview_image = self.main_window.preview_widget.get_current_image_data()
        if preview_image is None or preview_image.array is None:
            print("自动校色失败：没有可用的预览图像。")
            return
        # 初始化单次迭代状态
        self._auto_color_iteration = 0
        self._auto_color_max_iterations = 1
        self._auto_color_total_gains = np.zeros(3)
        # 执行一次
        self._perform_auto_color_iteration()

    def _on_auto_color_correct_clicked(self):
        """处理自动校色按钮点击事件，使用迭代次数逻辑"""
        if self._is_updating_ui: return
        
        preview_image = self.main_window.preview_widget.get_current_image_data()
        if preview_image is None or preview_image.array is None:
            print("自动校色失败：没有可用的预览图像。")
            return
            
        print("开始自动校色（基于迭代次数）...")
        
        # 初始化迭代状态
        self._auto_color_iteration = 0
        self._auto_color_max_iterations = 10  # 最大迭代次数
        self._auto_color_total_gains = np.zeros(3)
        
        # 开始第一次迭代
        self._perform_auto_color_iteration()

    def _perform_auto_color_iteration(self):
        """执行单次自动校色迭代，基于迭代次数"""
        # 检查最大迭代次数限制
        if self._auto_color_iteration >= self._auto_color_max_iterations:
            print("自动校色达到最大迭代次数限制，停止迭代。")
            print(f"最终累计增益: R={self._auto_color_total_gains[0]:.3f}, G={self._auto_color_total_gains[1]:.3f}, B={self._auto_color_total_gains[2]:.3f}")
            
            # 更新参数和UI
            current_rgb_gains = np.array(self.current_params.rgb_gains)
            new_rgb_gains = np.clip(current_rgb_gains + self._auto_color_total_gains, -1.0, 1.0)
            self.current_params.rgb_gains = tuple(new_rgb_gains)
            self.update_ui_from_params()
            self.parameter_changed.emit()
            return
            
        # 重新读取当前预览图像（这很关键！）
        current_preview = self.main_window.preview_widget.get_current_image_data()
        if current_preview is None or current_preview.array is None:
            print(f"迭代 {self._auto_color_iteration + 1}: 无法获取预览图像")
            
            # 即使无法获取预览图像，也要更新UI（使用累计的增益）
            if self._auto_color_total_gains.any():
                current_rgb_gains = np.array(self.current_params.rgb_gains)
                new_rgb_gains = np.clip(current_rgb_gains + self._auto_color_total_gains, -2.0, 2.0)
                self.current_params.rgb_gains = tuple(new_rgb_gains)
                self.update_ui_from_params()
                self.parameter_changed.emit()
            return
            
        # 计算当前图像的增益和光源RGB
        result = self.main_window.the_enlarger.calculate_auto_gain_learning_based(current_preview)
        current_gains = np.array(result[:3])  # 前三个是增益
        current_illuminant = np.array(result[3:])  # 后三个是光源RGB
        self._auto_color_iteration += 1
        
        print(f"  迭代 {self._auto_color_iteration}: 计算增益 = ({current_gains[0]:.3f}, {current_gains[1]:.3f}, {current_gains[2]:.3f})")
        print(f"  当前光源RGB = ({current_illuminant[0]:.3f}, {current_illuminant[1]:.3f}, {current_illuminant[2]:.3f})")
        
        # 累加增益
        self._auto_color_total_gains += current_gains
        
        # 应用当前增益到参数
        current_rgb_gains = np.array(self.current_params.rgb_gains)
        new_rgb_gains = np.clip(current_rgb_gains + current_gains, -2.0, 2.0)
        
        # 检查是否达到最大迭代次数
        if self._auto_color_iteration >= self._auto_color_max_iterations:
            print(f"自动校色完成！共执行 {self._auto_color_iteration} 次迭代")
            print(f"最终累计增益: R={self._auto_color_total_gains[0]:.3f}, G={self._auto_color_total_gains[1]:.3f}, B={self._auto_color_total_gains[2]:.3f}")
            print(f"最终RGB增益: R={new_rgb_gains[0]:.3f}, G={new_rgb_gains[1]:.3f}, B={new_rgb_gains[2]:.3f}")
            
            # 更新参数和UI
            self.current_params.rgb_gains = tuple(new_rgb_gains)
            self.update_ui_from_params()
            self.parameter_changed.emit()
            return
        
        # 如果增益变化很小（接近收敛），也提前结束（打印提示）
        if np.allclose(current_rgb_gains, new_rgb_gains, atol=1e-6):
            self.current_params.rgb_gains = tuple(new_rgb_gains)
            self.update_ui_from_params()
            self.parameter_changed.emit()
            print("自动校色提前收敛，停止迭代。")
            print(f"最终累计增益: R={self._auto_color_total_gains[0]:.3f}, G={self._auto_color_total_gains[1]:.3f}, B={self._auto_color_total_gains[2]:.3f}")
            return
        
        # 继续迭代
        self.current_params.rgb_gains = tuple(new_rgb_gains)
        self.update_ui_from_params()
        self.parameter_changed.emit()  # 触发预览更新
        
        # 等待一小段时间让预览更新完成，然后进行下一次迭代
        QTimer.singleShot(50, self._perform_auto_color_iteration)

    def get_current_params(self) -> ColorGradingParams:
        return self.current_params
    
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
                        
                        print(f"已加载默认曲线: Kodak Endura Paper")
                    else:
                        print("默认曲线文件格式不正确")
            else:
                print("默认曲线文件不存在")
        except Exception as e:
            print(f"加载默认曲线失败: {e}")

    def _on_export_3dlut_clicked(self):
        """处理3D LUT导出按钮点击事件"""
        if self._is_updating_ui: return
        
        try:
            # 获取当前参数
            current_params = self.get_current_params()
            
            # 检查是否有任何功能被启用
            enabled_features = []
            if current_params.enable_density_inversion:
                enabled_features.append("密度反相")
            if current_params.enable_correction_matrix:
                enabled_features.append("校正矩阵")
            if current_params.enable_rgb_gains:
                enabled_features.append("RGB增益")
            if current_params.enable_density_curve:
                enabled_features.append("密度曲线")
            
            if not enabled_features:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", "没有启用任何调色功能，无法生成有意义的LUT。")
                return
            
            # 创建用于LUT生成的参数副本，确保不包含输入色彩空间转换
            lut_params = current_params.copy()
            
            # 生成LUT
            from divere.utils.lut_generator.core import LUTManager
            
            def transform_function(input_rgb):
                """LUT变换函数，应用所有使能的调色功能"""
                # 创建虚拟图像数据
                from divere.core.data_types import ImageData
                import numpy as np
                
                # 确保输入是2D数组 (N, 3)
                if input_rgb.ndim == 1:
                    input_rgb = input_rgb.reshape(1, 3)
                
                # 创建虚拟图像
                virtual_image = ImageData(
                    array=input_rgb.reshape(-1, 1, 3),
                    width=1,
                    height=input_rgb.shape[0],
                    channels=3,
                    dtype=np.float32,
                    color_space="ACEScg",  # 使用ACEScg作为工作色彩空间
                    file_path="",
                    is_proxy=True,
                    proxy_scale=1.0
                )
                
                # 应用调色管道（不包含输入色彩空间转换）
                result = self.main_window.the_enlarger.apply_full_pipeline(virtual_image, lut_params)
                
                # 返回结果
                return result.array.reshape(-1, 3)
            
            # 获取选择的LUT size
            lut_size = int(self.lut_3d_size_combo.currentText())
            
            # 生成3D LUT
            lut_manager = LUTManager()
            lut_info = lut_manager.generate_3d_lut(
                transform_function, 
                size=lut_size, 
                title=f"DiVERE 3D LUT - {', '.join(enabled_features)} ({lut_size}x{lut_size}x{lut_size})"
            )
            
            # 选择保存路径
            from PySide6.QtWidgets import QFileDialog
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 获取上次保存LUT的目录
            last_directory = enhanced_config_manager.get_directory("save_lut")
            default_filename = f"DiVERE_3D_{lut_size}_{', '.join(enabled_features)}_{timestamp}.cube"
            if last_directory:
                default_path = str(Path(last_directory) / default_filename)
            else:
                default_path = default_filename
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存3D LUT文件",
                default_path,
                "CUBE Files (*.cube);;All Files (*)"
            )
            
            if file_path:
                # 保存当前目录
                enhanced_config_manager.set_directory("save_lut", file_path)
                
                # 保存LUT
                success = lut_manager.save_lut(lut_info, file_path)
                
                if success:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self, 
                        "成功", 
                        f"3D LUT已成功导出到:\n{file_path}\n\n包含功能: {', '.join(enabled_features)}"
                    )
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "错误", "保存LUT文件失败。")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"导出3D LUT时发生错误:\n{str(e)}")
            print(f"导出3D LUT错误: {e}")
            import traceback
            traceback.print_exc()

    def _on_export_input_cc_lut_clicked(self):
        """处理导出输入色彩管理LUT按钮点击事件"""
        if self._is_updating_ui: return
        
        try:
            # 获取当前输入色彩空间
            current_input_space = self.input_colorspace_combo.currentText()
            if not current_input_space:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", "请先选择输入色彩空间。")
                return
            
            # 获取工作色彩空间（通常是ACEScg）
            working_space = "ACEScg"  # 默认工作色彩空间
            
            # 检查色彩空间转换是否有效
            if not self.main_window.color_space_manager.validate_color_space(current_input_space):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", f"输入色彩空间 '{current_input_space}' 无效。")
                return
            
            if not self.main_window.color_space_manager.validate_color_space(working_space):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", f"工作色彩空间 '{working_space}' 无效。")
                return
            
            # 生成LUT
            from divere.utils.lut_generator.core import LUTManager
            
            def transform_function(input_rgb):
                """LUT变换函数，应用输入色彩空间转换"""
                # 创建虚拟图像数据
                from divere.core.data_types import ImageData
                import numpy as np
                
                # 确保输入是2D数组 (N, 3)
                if input_rgb.ndim == 1:
                    input_rgb = input_rgb.reshape(1, 3)
                
                # 创建虚拟图像，使用输入色彩空间
                virtual_image = ImageData(
                    array=input_rgb.reshape(-1, 1, 3),
                    width=1,
                    height=input_rgb.shape[0],
                    channels=3,
                    dtype=np.float32,
                    color_space=current_input_space,  # 使用当前选择的输入色彩空间
                    file_path="",
                    is_proxy=True,
                    proxy_scale=1.0
                )
                
                # 应用输入色彩空间转换（与_reload_with_color_space中的逻辑完全相同）
                # 1. 设置图像色彩空间
                converted_image = self.main_window.color_space_manager.set_image_color_space(
                    virtual_image, current_input_space
                )
                
                # 2. 转换到工作色彩空间
                converted_image = self.main_window.color_space_manager.convert_to_working_space(
                    converted_image
                )
                
                # 返回结果
                return converted_image.array.reshape(-1, 3)
            
            # 获取选择的LUT size
            lut_size = int(self.input_cc_lut_size_combo.currentText())
            
            # 生成3D LUT
            lut_manager = LUTManager()
            lut_info = lut_manager.generate_3d_lut(
                transform_function, 
                size=lut_size, 
                title=f"DiVERE 输入色彩管理LUT - {current_input_space} to {working_space} ({lut_size}x{lut_size}x{lut_size})"
            )
            
            # 选择保存路径
            from PySide6.QtWidgets import QFileDialog
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 获取上次保存LUT的目录
            last_directory = enhanced_config_manager.get_directory("save_lut")
            default_filename = f"DiVERE_3D_{lut_size}x{lut_size}x{lut_size}_InputCC_{current_input_space}_to_{working_space}_{timestamp}.cube"
            if last_directory:
                default_path = str(Path(last_directory) / default_filename)
            else:
                default_path = default_filename
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存输入色彩管理LUT文件",
                default_path,
                "CUBE Files (*.cube);;All Files (*)"
            )
            
            if file_path:
                # 保存当前目录
                enhanced_config_manager.set_directory("save_lut", file_path)
                
                # 保存LUT
                success = lut_manager.save_lut(lut_info, file_path)
                
                if success:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self, 
                        "成功", 
                        f"输入色彩管理LUT已成功导出到:\n{file_path}\n\n"
                        f"转换: {current_input_space} → {working_space}\n"
                        f"此LUT包含完整的输入色彩空间转换过程，与程序内置的色彩管理完全相同。"
                    )
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "错误", "保存LUT文件失败。")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"导出输入色彩管理LUT时发生错误:\n{str(e)}")
            print(f"导出输入色彩管理LUT错误: {e}")
            import traceback
            traceback.print_exc()

    def _on_export_density_curve_1dlut_clicked(self):
        """处理导出密度曲线1D LUT按钮点击事件"""
        if self._is_updating_ui: return
        
        try:
            # 获取当前参数
            current_params = self.get_current_params()
            
            # 检查是否有密度曲线存在（不依赖enable_density_curve checkbox）
            available_curves = []
            if current_params.curve_points and len(current_params.curve_points) >= 2:
                available_curves.append("RGB曲线")
            if current_params.curve_points_r and len(current_params.curve_points_r) >= 2:
                available_curves.append("R曲线")
            if current_params.curve_points_g and len(current_params.curve_points_g) >= 2:
                available_curves.append("G曲线")
            if current_params.curve_points_b and len(current_params.curve_points_b) >= 2:
                available_curves.append("B曲线")
            
            if not available_curves:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "警告", "没有定义任何密度曲线，无法生成1D LUT。")
                return
            
            # 获取选择的LUT size
            lut_size = int(self.density_curve_1dlut_size_combo.currentText())
            
            # 生成密度空间的1D LUT
            def generate_density_curve_1dlut():
                """生成密度空间的1D LUT"""
                import numpy as np
                
                # 步骤1: 生成线性数据
                input_linear = np.linspace(0, 1.0, lut_size)
                
                # 步骤2: 计算密度
                input_density = -np.log10(np.maximum(input_linear, 1e-10))
                
                # 初始化输出密度值
                output_density = input_density.copy()
                
                # 处理RGB主曲线
                if current_params.curve_points and len(current_params.curve_points) >= 2:
                    # 生成RGB曲线LUT
                    curve_samples = self.main_window.the_enlarger._generate_monotonic_curve(
                        current_params.curve_points, lut_size
                    )
                    rgb_lut = np.array([p[1] for p in curve_samples])
                    
                    # 将输入密度归一化到[0,1]用于查找曲线
                    # 注意：密度值需要反转，因为曲线是从暗到亮，而密度是从亮到暗
                    normalized_density = 1 - np.clip((input_density - 0) / (np.log10(65536) - 0), 0, 1)
                    
                    # 查找曲线值
                    lut_indices = np.clip(normalized_density * (lut_size - 1), 0, lut_size - 1).astype(int)
                    curve_output = rgb_lut[lut_indices]
                    
                    # 将曲线输出映射到输出密度范围
                    output_density_all = (1 - curve_output) * np.log10(65536)
                    
                    # 应用到所有通道
                    output_density = output_density_all
                
                # 处理单通道曲线
                channel_curves = [
                    (current_params.curve_points_r, 0),  # R通道
                    (current_params.curve_points_g, 1),  # G通道
                    (current_params.curve_points_b, 2)   # B通道
                ]
                
                # 准备输出数组
                final_output = np.zeros((lut_size, 3))
                
                for curve_points, channel_idx in channel_curves:
                    if curve_points and len(curve_points) >= 2:
                        # 生成单通道曲线LUT
                        curve_samples = self.main_window.the_enlarger._generate_monotonic_curve(
                            curve_points, lut_size
                        )
                        channel_lut = np.array([p[1] for p in curve_samples])
                        
                        # 将输入密度归一化到[0,1]用于查找曲线
                        normalized_density = 1 - np.clip((output_density - 0) / (np.log10(65536) - 0), 0, 1)
                        
                        # 查找曲线值
                        lut_indices = np.clip(normalized_density * (lut_size - 1), 0, lut_size - 1).astype(int)
                        curve_output = channel_lut[lut_indices]
                        
                        # 将曲线输出映射到输出密度范围
                        channel_output_density = (1 - curve_output) * np.log10(65536)
                        
                        # 将密度值转换回线性值
                        final_output[:, channel_idx] = np.power(10, -channel_output_density)
                    else:
                        # 如果没有单通道曲线，使用RGB曲线的结果或原始线性值
                        if current_params.curve_points and len(current_params.curve_points) >= 2:
                            # 使用RGB曲线的结果
                            final_output[:, channel_idx] = np.power(10, -output_density)
                        else:
                            # 使用原始线性值
                            final_output[:, channel_idx] = input_linear
                
                # 如果没有单通道曲线，但有RGB曲线，应用到所有通道
                if not any(curve_points and len(curve_points) >= 2 for curve_points, _ in channel_curves):
                    if current_params.curve_points and len(current_params.curve_points) >= 2:
                        final_output[:, :] = np.power(10, -output_density)[:, np.newaxis]
                    else:
                        final_output[:, :] = input_linear[:, np.newaxis]
                
                # 确保最终结果在合理范围内
                final_output = np.clip(final_output, 0.0, 1.0)
                
                return final_output
            
            # 生成1D LUT数据
            lut_data = generate_density_curve_1dlut()
            
            # 创建LUT信息字典
            lut_info = {
                'type': '1D',
                'size': lut_size,
                'data': lut_data,
                'title': f"DiVERE 密度曲线1D LUT - {', '.join(available_curves)} ({lut_size}点)",
                'curves': available_curves,
                'generator': None  # 我们直接生成数据，不使用LUT1DGenerator
            }
            
            # 选择保存路径
            from PySide6.QtWidgets import QFileDialog
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 获取上次保存LUT的目录
            last_directory = enhanced_config_manager.get_directory("save_lut")
            default_filename = f"DiVERE_1D_{lut_size}_{', '.join(available_curves)}_{timestamp}.cube"
            if last_directory:
                default_path = str(Path(last_directory) / default_filename)
            else:
                default_path = default_filename
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存密度曲线1D LUT文件",
                default_path,
                "CUBE Files (*.cube);;All Files (*)"
            )
            
            if file_path:
                # 保存当前目录
                enhanced_config_manager.set_directory("save_lut", file_path)
                
                # 直接保存为CUBE格式
                success = self._save_1dlut_as_cube(lut_data, file_path, lut_info['title'])
                
                if success:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self, 
                        "成功", 
                        f"密度曲线1D LUT已成功导出到:\n{file_path}\n\n"
                        f"包含曲线: {', '.join(available_curves)}\n"
                        f"LUT大小: {lut_size}点\n"
                        f"此1D LUT包含所有密度曲线，直接作用在密度空间上。"
                    )
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "错误", "保存LUT文件失败。")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"导出密度曲线1D LUT时发生错误:\n{str(e)}")
            print(f"导出密度曲线1D LUT错误: {e}")
            import traceback
            traceback.print_exc()

    def _save_1dlut_as_cube(self, lut_data, filepath: str, title: str) -> bool:
        """保存1D LUT为CUBE格式"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入CUBE文件头
                f.write(f"# {title}\n")
                f.write("# Generated by DiVERE\n")
                f.write(f"LUT_1D_SIZE {lut_data.shape[0]}\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
                f.write("\n")
                
                # 写入LUT数据
                for i in range(lut_data.shape[0]):
                    r, g, b = lut_data[i]
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
            
            return True
        except Exception as e:
            print(f"保存1D LUT失败: {e}")
            return False

    def _save_matrix(self):
        """保存当前矩阵到文件"""
        # 检查当前是否有自定义矩阵
        if (self.current_params.correction_matrix_file != "custom" or 
            self.current_params.correction_matrix is None):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                "警告", 
                "请先编辑矩阵或选择'自定义'模式，然后保存。"
            )
            return
        
        # 获取矩阵名称
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, 
            "保存矩阵", 
            "请输入矩阵名称:"
        )
        
        if not ok or not name:
            return
        
        # 获取描述
        description, ok = QInputDialog.getText(
            self, 
            "保存矩阵", 
            "请输入矩阵描述（可选）:"
        )
        
        if not ok:
            return
        
        # 准备矩阵数据
        matrix_data = {
            "name": name,
            "description": description,
            "film_type": "custom",
            "matrix_space": "density",  # 默认保存为密度空间矩阵
            "matrix": self.current_params.correction_matrix.tolist()
        }
        
        # 生成文件名（去除特殊字符）
        safe_filename = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        
        # 打开文件保存对话框
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        from ..utils.config_manager import config_manager
        
        # 获取上次保存矩阵的目录
        last_directory = enhanced_config_manager.get_directory("save_matrix")
        if not last_directory:
            last_directory = "config/matrices"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存矩阵文件",
            f"{last_directory}/{safe_filename}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                # 确保文件有.json扩展名
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # 保存当前目录
                enhanced_config_manager.set_directory("save_matrix", file_path)
                
                # 保存文件
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(matrix_data, f, indent=2, ensure_ascii=False)
                
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "成功", f"矩阵已保存到：\n{file_path}")
                
                # 重新加载矩阵列表
                self._refresh_matrix_combo()
                
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "错误", f"保存矩阵时出错：\n{str(e)}")
    
    def _refresh_matrix_combo(self):
        """刷新矩阵下拉列表"""
        # 保存当前选择
        current_data = self.matrix_combo.currentData()
        
        # 重新加载矩阵
        self.main_window.the_enlarger.reload_matrices()
        
        # 清空并重新填充
        self.matrix_combo.clear()
        self.matrix_combo.addItem("自定义", "custom")
        
        # 重新加载可用矩阵
        available = self.main_window.the_enlarger.get_available_matrices()
        for matrix_id in available:
            data = self.main_window.the_enlarger._load_correction_matrix(matrix_id)
            if data: 
                self.matrix_combo.addItem(data.get("name", matrix_id), matrix_id)
        
        # 恢复选择
        index = self.matrix_combo.findData(current_data)
        if index >= 0:
            self.matrix_combo.setCurrentIndex(index)
