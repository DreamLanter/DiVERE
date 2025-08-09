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
    
    def __init__(self, parent=None, color_spaces=None):
        super().__init__(parent)
        self.setWindowTitle("保存图像设置")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # 可用的色彩空间
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
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
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
        if self.tiff_16bit_radio.isChecked():
            # 16-bit TIFF 默认使用 DisplayP3
            if "DisplayP3" in self.color_spaces:
                self.colorspace_combo.setCurrentText("DisplayP3")
            elif "AdobeRGB" in self.color_spaces:
                self.colorspace_combo.setCurrentText("AdobeRGB")
        else:
            # 8-bit JPEG 默认使用 sRGB
            if "sRGB" in self.color_spaces:
                self.colorspace_combo.setCurrentText("sRGB")
    
    def get_settings(self):
        """获取保存设置"""
        return {
            "format": "tiff" if self.tiff_16bit_radio.isChecked() else "jpeg",
            "bit_depth": 16 if self.tiff_16bit_radio.isChecked() else 8,
            "color_space": self.colorspace_combo.currentText(),
            "include_curve": self.include_curve_checkbox.isChecked()
        }