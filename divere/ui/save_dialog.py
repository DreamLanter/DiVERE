"""
保存图像对话框
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QRadioButton, QComboBox, QCheckBox, QPushButton,
    QLabel, QGridLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt
from pathlib import Path


class SaveImageDialog(QDialog):
    """保存图像对话框"""
    
    def __init__(self, parent=None, color_spaces=None, is_bw_mode=False, color_space_manager=None):
        super().__init__(parent)
        self.setWindowTitle("保存图像设置")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._save_mode = 'single'  # 'single' | 'all'
        self._is_bw_mode = is_bw_mode
        self._color_space_manager = color_space_manager
        
        # 可用的色彩空间
        if color_spaces is None and color_space_manager:
            # 使用过滤后的regular色彩空间（有ICC文件的）
            self.color_spaces = color_space_manager.get_regular_color_spaces_with_icc()
        else:
            self.color_spaces = color_spaces or ["sRGB", "AdobeRGB", "ProPhotoRGB"]
        
        # 创建UI
        self._create_ui()
        
        # 设置默认值
        self._set_defaults()
        
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        
        # 文件格式选择
        format_group = QGroupBox("文件格式")
        format_layout = QVBoxLayout(format_group)
        
        self.tiff_16bit_radio = QRadioButton("16-bit TIFF (推荐)")
        self.jpeg_8bit_radio = QRadioButton("8-bit JPEG")
        
        format_layout.addWidget(self.tiff_16bit_radio)
        format_layout.addWidget(self.jpeg_8bit_radio)
        
        layout.addWidget(format_group)
        
        # 色彩空间选择
        colorspace_group = QGroupBox("输出色彩空间")
        colorspace_layout = QGridLayout(colorspace_group)
        
        colorspace_layout.addWidget(QLabel("色彩空间:"), 0, 0)
        self.colorspace_combo = QComboBox()
        self.colorspace_combo.addItems(self.color_spaces)
        colorspace_layout.addWidget(self.colorspace_combo, 0, 1)
        
        layout.addWidget(colorspace_group)
        
        # 处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout(options_group)
        
        self.include_curve_checkbox = QCheckBox("包含密度曲线调整")
        self.include_curve_checkbox.setChecked(True)
        options_layout.addWidget(self.include_curve_checkbox)
        
        layout.addWidget(options_group)
        
        # 按钮
        button_box = QDialogButtonBox()
        # 标准“保存单张”按钮
        ok_btn = button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("保存单张")
        # 自定义“保存所有”按钮
        save_all_btn = QPushButton("保存所有")
        button_box.addButton(save_all_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        # 取消
        cancel_btn = button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        
        def _on_ok():
            self._save_mode = 'single'
            self.accept()
        def _on_save_all():
            self._save_mode = 'all'
            self.accept()
        def _on_cancel():
            self.reject()
        ok_btn.clicked.connect(_on_ok)
        save_all_btn.clicked.connect(_on_save_all)
        cancel_btn.clicked.connect(_on_cancel)
        
        layout.addWidget(button_box)
        
        # 连接信号
        self.tiff_16bit_radio.toggled.connect(self._on_format_changed)
        self.jpeg_8bit_radio.toggled.connect(self._on_format_changed)
        
    def _set_defaults(self):
        """设置默认值"""
        self.tiff_16bit_radio.setChecked(True)
        self._on_format_changed()
        
    def _on_format_changed(self):
        """格式选择改变时更新默认色彩空间"""
        if self._is_bw_mode:
            # B&W mode: prioritize grayscale color spaces
            preferred = ["Gray_Gamma_2_2", "Gray Gamma 2.2", "Grayscale", "sRGB"]
        else:
            # Color mode: use existing logic
            if self.tiff_16bit_radio.isChecked():
                # 16-bit 默认使用 ACEScg_Linear（若不存在则退化到 ACEScg 或 DisplayP3/AdobeRGB）
                preferred = ["ACEScg_Linear", "ACEScg", "DisplayP3", "AdobeRGB", "sRGB"]
            else:
                # 8-bit JPEG 默认使用 DisplayP3（若不可用则退化到 sRGB/AdobeRGB）
                preferred = ["DisplayP3", "sRGB", "AdobeRGB"]
        
        for name in preferred:
            if name in self.color_spaces:
                self.colorspace_combo.setCurrentText(name)
                break
    
    def get_settings(self):
        """获取保存设置"""
        return {
            "format": "tiff" if self.tiff_16bit_radio.isChecked() else "jpeg",
            "bit_depth": 16 if self.tiff_16bit_radio.isChecked() else 8,
            "color_space": self.colorspace_combo.currentText(),
            "include_curve": self.include_curve_checkbox.isChecked(),
            "save_mode": self._save_mode
        }