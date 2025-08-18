#!/usr/bin/env python3
"""
配置管理对话框
提供用户友好的界面来管理配置文件
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QMessageBox, QFileDialog, QInputDialog, QGroupBox, QGridLayout,
    QSplitter, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon
from pathlib import Path
import json
import shutil

from divere.utils.enhanced_config_manager import enhanced_config_manager


class ConfigManagerDialog(QDialog):
    """配置管理对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置管理器")
        self.setGeometry(100, 100, 1000, 700)
        
        self._create_ui()
        self._load_configs()
        
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("DiVERE 配置管理器")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 用户配置目录信息
        config_dir_label = QLabel(f"用户配置目录: {enhanced_config_manager.get_user_config_dir_path()}")
        config_dir_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(config_dir_label)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 色彩空间配置标签页
        self._create_colorspace_tab()
        
        # 曲线配置标签页
        self._create_curves_tab()
        
        # 矩阵配置标签页
        self._create_matrices_tab()
        
        # 应用设置标签页
        self._create_app_settings_tab()
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        open_dir_btn = QPushButton("打开配置目录")
        open_dir_btn.clicked.connect(enhanced_config_manager.open_user_config_dir)
        button_layout.addWidget(open_dir_btn)
        
        backup_btn = QPushButton("备份配置")
        backup_btn.clicked.connect(self._backup_configs)
        button_layout.addWidget(backup_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def _create_colorspace_tab(self):
        """创建色彩空间配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 说明
        info_label = QLabel("色彩空间定义文件。用户配置优先于内置配置。")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：配置列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 按钮组
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("添加配置")
        add_btn.clicked.connect(self._add_colorspace_config)
        btn_layout.addWidget(add_btn)
        
        copy_btn = QPushButton("复制内置配置")
        copy_btn.clicked.connect(self._copy_colorspace_config)
        btn_layout.addWidget(copy_btn)
        
        delete_btn = QPushButton("删除配置")
        delete_btn.clicked.connect(self._delete_colorspace_config)
        btn_layout.addWidget(delete_btn)
        
        left_layout.addLayout(btn_layout)
        
        # 配置列表
        self.colorspace_list = QListWidget()
        self.colorspace_list.itemClicked.connect(self._on_colorspace_selected)
        left_layout.addWidget(self.colorspace_list)
        
        splitter.addWidget(left_widget)
        
        # 右侧：配置内容
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("配置内容:"))
        
        self.colorspace_editor = QTextEdit()
        self.colorspace_editor.setFont(QFont("Courier", 10))
        right_layout.addWidget(self.colorspace_editor)
        
        # 保存按钮
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self._save_colorspace_config)
        right_layout.addWidget(save_btn)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])
        
        self.tab_widget.addTab(tab, "色彩空间")
        
    def _create_curves_tab(self):
        """创建曲线配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 说明
        info_label = QLabel("密度打印曲线预设。支持RGB、R、G、B四个通道。")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：配置列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 按钮组
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("添加曲线")
        add_btn.clicked.connect(self._add_curve_config)
        btn_layout.addWidget(add_btn)
        
        copy_btn = QPushButton("复制内置曲线")
        copy_btn.clicked.connect(self._copy_curve_config)
        btn_layout.addWidget(copy_btn)
        
        delete_btn = QPushButton("删除曲线")
        delete_btn.clicked.connect(self._delete_curve_config)
        btn_layout.addWidget(delete_btn)
        
        left_layout.addLayout(btn_layout)
        
        # 配置列表
        self.curves_list = QListWidget()
        self.curves_list.itemClicked.connect(self._on_curve_selected)
        left_layout.addWidget(self.curves_list)
        
        splitter.addWidget(left_widget)
        
        # 右侧：配置内容
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("曲线配置:"))
        
        self.curves_editor = QTextEdit()
        self.curves_editor.setFont(QFont("Courier", 10))
        right_layout.addWidget(self.curves_editor)
        
        # 保存按钮
        save_btn = QPushButton("保存曲线")
        save_btn.clicked.connect(self._save_curve_config)
        right_layout.addWidget(save_btn)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])
        
        self.tab_widget.addTab(tab, "曲线")
        
    def _create_matrices_tab(self):
        """创建矩阵配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 说明
        info_label = QLabel("色彩校正矩阵。用于特定胶片类型的校正。")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：配置列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 按钮组
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("添加矩阵")
        add_btn.clicked.connect(self._add_matrix_config)
        btn_layout.addWidget(add_btn)
        
        copy_btn = QPushButton("复制内置矩阵")
        copy_btn.clicked.connect(self._copy_matrix_config)
        btn_layout.addWidget(copy_btn)
        
        delete_btn = QPushButton("删除矩阵")
        delete_btn.clicked.connect(self._delete_matrix_config)
        btn_layout.addWidget(delete_btn)
        
        left_layout.addLayout(btn_layout)
        
        # 配置列表
        self.matrices_list = QListWidget()
        self.matrices_list.itemClicked.connect(self._on_matrix_selected)
        left_layout.addWidget(self.matrices_list)
        
        splitter.addWidget(left_widget)
        
        # 右侧：配置内容
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("矩阵配置:"))
        
        self.matrices_editor = QTextEdit()
        self.matrices_editor.setFont(QFont("Courier", 10))
        right_layout.addWidget(self.matrices_editor)
        
        # 保存按钮
        save_btn = QPushButton("保存矩阵")
        save_btn.clicked.connect(self._save_matrix_config)
        right_layout.addWidget(save_btn)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])
        
        self.tab_widget.addTab(tab, "矩阵")
        
    def _create_app_settings_tab(self):
        """创建应用设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 说明
        info_label = QLabel("应用设置。包括界面设置、目录记忆等。")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 设置编辑器
        self.app_settings_editor = QTextEdit()
        self.app_settings_editor.setFont(QFont("Courier", 10))
        layout.addWidget(self.app_settings_editor)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        reload_btn = QPushButton("重新加载")
        reload_btn.clicked.connect(self._reload_app_settings)
        btn_layout.addWidget(reload_btn)
        
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self._save_app_settings)
        btn_layout.addWidget(save_btn)
        
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        self.tab_widget.addTab(tab, "应用设置")
        
    def _load_configs(self):
        """加载所有配置"""
        self._load_colorspace_configs()
        self._load_curves_configs()
        self._load_matrices_configs()
        self._load_app_settings()
        
    def _load_colorspace_configs(self):
        """加载色彩空间配置"""
        self.colorspace_list.clear()
        config_files = enhanced_config_manager.get_config_files("colorspace")
        
        for config_file in config_files:
            item = QListWidgetItem(config_file.stem)
            if config_file.parent == enhanced_config_manager.user_colorspace_dir:
                item.setText(f"📁 {config_file.stem} (用户)")
                item.setData(Qt.ItemDataRole.UserRole, "user")
            else:
                item.setText(f"📦 {config_file.stem} (内置)")
                item.setData(Qt.ItemDataRole.UserRole, "builtin")
            item.setData(Qt.ItemDataRole.UserRole + 1, config_file)
            self.colorspace_list.addItem(item)
            
    def _load_curves_configs(self):
        """加载曲线配置"""
        self.curves_list.clear()
        config_files = enhanced_config_manager.get_config_files("curves")
        
        for config_file in config_files:
            item = QListWidgetItem(config_file.stem)
            if config_file.parent == enhanced_config_manager.user_curves_dir:
                item.setText(f"📁 {config_file.stem} (用户)")
                item.setData(Qt.ItemDataRole.UserRole, "user")
            else:
                item.setText(f"📦 {config_file.stem} (内置)")
                item.setData(Qt.ItemDataRole.UserRole, "builtin")
            item.setData(Qt.ItemDataRole.UserRole + 1, config_file)
            self.curves_list.addItem(item)
            
    def _load_matrices_configs(self):
        """加载矩阵配置"""
        self.matrices_list.clear()
        config_files = enhanced_config_manager.get_config_files("matrices")
        
        for config_file in config_files:
            item = QListWidgetItem(config_file.stem)
            if config_file.parent == enhanced_config_manager.user_matrices_dir:
                item.setText(f"📁 {config_file.stem} (用户)")
                item.setData(Qt.ItemDataRole.UserRole, "user")
            else:
                item.setText(f"📦 {config_file.stem} (内置)")
                item.setData(Qt.ItemDataRole.UserRole, "builtin")
            item.setData(Qt.ItemDataRole.UserRole + 1, config_file)
            self.matrices_list.addItem(item)
            
    def _load_app_settings(self):
        """加载应用设置"""
        try:
            settings = enhanced_config_manager.app_settings
            self.app_settings_editor.setPlainText(json.dumps(settings, indent=2, ensure_ascii=False))
        except Exception as e:
            self.app_settings_editor.setPlainText(f"加载应用设置失败: {e}")
            
    def _on_colorspace_selected(self, item):
        """色彩空间配置被选中"""
        config_file = item.data(Qt.ItemDataRole.UserRole + 1)
        try:
            data = enhanced_config_manager.load_config_file(config_file)
            if data:
                self.colorspace_editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                self.colorspace_editor.setPlainText("无法加载配置文件")
        except Exception as e:
            self.colorspace_editor.setPlainText(f"加载配置失败: {e}")
            
    def _on_curve_selected(self, item):
        """曲线配置被选中"""
        config_file = item.data(Qt.ItemDataRole.UserRole + 1)
        try:
            data = enhanced_config_manager.load_config_file(config_file)
            if data:
                self.curves_editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                self.curves_editor.setPlainText("无法加载配置文件")
        except Exception as e:
            self.curves_editor.setPlainText(f"加载配置失败: {e}")
            
    def _on_matrix_selected(self, item):
        """矩阵配置被选中"""
        config_file = item.data(Qt.ItemDataRole.UserRole + 1)
        try:
            data = enhanced_config_manager.load_config_file(config_file)
            if data:
                self.matrices_editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                self.matrices_editor.setPlainText("无法加载配置文件")
        except Exception as e:
            self.matrices_editor.setPlainText(f"加载配置失败: {e}")
            
    def _add_colorspace_config(self):
        """添加色彩空间配置"""
        name, ok = QInputDialog.getText(self, "添加色彩空间", "请输入色彩变换名称:")
        if ok and name:
            # 创建默认配置
            default_config = {
                "name": name,
                "description": "用户自定义色彩空间",
                "primaries": {
                    "R": [0.64, 0.33],
                    "G": [0.30, 0.60],
                    "B": [0.15, 0.06]
                },
                "white_point": [0.3127, 0.3290],
                "gamma": 2.2
            }
            
            if enhanced_config_manager.save_user_config("colorspace", name, default_config):
                self._load_colorspace_configs()
                QMessageBox.information(self, "成功", f"已创建色彩空间配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "创建配置失败")
                
    def _add_curve_config(self):
        """添加曲线配置"""
        name, ok = QInputDialog.getText(self, "添加曲线", "请输入曲线名称:")
        if ok and name:
            # 创建默认配置
            default_config = {
                "name": name,
                "description": "用户自定义曲线",
                "curves": {
                    "RGB": [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]],
                    "R": [[0.0, 0.0], [1.0, 1.0]],
                    "G": [[0.0, 0.0], [1.0, 1.0]],
                    "B": [[0.0, 0.0], [1.0, 1.0]]
                }
            }
            
            if enhanced_config_manager.save_user_config("curves", name, default_config):
                self._load_curves_configs()
                QMessageBox.information(self, "成功", f"已创建曲线配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "创建配置失败")
                
    def _add_matrix_config(self):
        """添加矩阵配置"""
        name, ok = QInputDialog.getText(self, "添加矩阵", "请输入矩阵名称:")
        if ok and name:
            # 创建默认配置
            default_config = {
                "name": name,
                "description": "用户自定义矩阵",
                "matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]
            }
            
            if enhanced_config_manager.save_user_config("matrices", name, default_config):
                self._load_matrices_configs()
                QMessageBox.information(self, "成功", f"已创建矩阵配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "创建配置失败")
                
    def _copy_colorspace_config(self):
        """复制内置色彩空间配置"""
        current_item = self.colorspace_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_file = current_item.data(Qt.ItemDataRole.UserRole + 1)
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        
        if config_type == "user":
            QMessageBox.information(self, "提示", "用户配置无需复制")
            return
            
        name = config_file.stem
        if enhanced_config_manager.copy_default_to_user("colorspace", name):
            self._load_colorspace_configs()
            QMessageBox.information(self, "成功", f"已复制配置到用户目录: {name}")
        else:
            QMessageBox.critical(self, "错误", "复制配置失败")
            
    def _copy_curve_config(self):
        """复制内置曲线配置"""
        current_item = self.curves_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_file = current_item.data(Qt.ItemDataRole.UserRole + 1)
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        
        if config_type == "user":
            QMessageBox.information(self, "提示", "用户配置无需复制")
            return
            
        name = config_file.stem
        if enhanced_config_manager.copy_default_to_user("curves", name):
            self._load_curves_configs()
            QMessageBox.information(self, "成功", f"已复制配置到用户目录: {name}")
        else:
            QMessageBox.critical(self, "错误", "复制配置失败")
            
    def _copy_matrix_config(self):
        """复制内置矩阵配置"""
        current_item = self.matrices_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_file = current_item.data(Qt.ItemDataRole.UserRole + 1)
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        
        if config_type == "user":
            QMessageBox.information(self, "提示", "用户配置无需复制")
            return
            
        name = config_file.stem
        if enhanced_config_manager.copy_default_to_user("matrices", name):
            self._load_matrices_configs()
            QMessageBox.information(self, "成功", f"已复制配置到用户目录: {name}")
        else:
            QMessageBox.critical(self, "错误", "复制配置失败")
            
    def _delete_colorspace_config(self):
        """删除色彩空间配置"""
        current_item = self.colorspace_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法删除内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        reply = QMessageBox.question(self, "确认删除", f"确定要删除配置 '{name}' 吗？")
        
        if reply == QMessageBox.StandardButton.Yes:
            if enhanced_config_manager.delete_user_config("colorspace", name):
                self._load_colorspace_configs()
                self.colorspace_editor.clear()
                QMessageBox.information(self, "成功", f"已删除配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "删除配置失败")
                
    def _delete_curve_config(self):
        """删除曲线配置"""
        current_item = self.curves_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法删除内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        reply = QMessageBox.question(self, "确认删除", f"确定要删除配置 '{name}' 吗？")
        
        if reply == QMessageBox.StandardButton.Yes:
            if enhanced_config_manager.delete_user_config("curves", name):
                self._load_curves_configs()
                self.curves_editor.clear()
                QMessageBox.information(self, "成功", f"已删除配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "删除配置失败")
                
    def _delete_matrix_config(self):
        """删除矩阵配置"""
        current_item = self.matrices_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法删除内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        reply = QMessageBox.question(self, "确认删除", f"确定要删除配置 '{name}' 吗？")
        
        if reply == QMessageBox.StandardButton.Yes:
            if enhanced_config_manager.delete_user_config("matrices", name):
                self._load_matrices_configs()
                self.matrices_editor.clear()
                QMessageBox.information(self, "成功", f"已删除配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "删除配置失败")
                
    def _save_colorspace_config(self):
        """保存色彩空间配置"""
        current_item = self.colorspace_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法修改内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        
        try:
            content = self.colorspace_editor.toPlainText()
            data = json.loads(content)
            
            if enhanced_config_manager.save_user_config("colorspace", name, data):
                QMessageBox.information(self, "成功", f"已保存配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "保存配置失败")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "错误", f"JSON格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            
    def _save_curve_config(self):
        """保存曲线配置"""
        current_item = self.curves_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法修改内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        
        try:
            content = self.curves_editor.toPlainText()
            data = json.loads(content)
            
            if enhanced_config_manager.save_user_config("curves", name, data):
                QMessageBox.information(self, "成功", f"已保存配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "保存配置失败")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "错误", f"JSON格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            
    def _save_matrix_config(self):
        """保存矩阵配置"""
        current_item = self.matrices_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个配置")
            return
            
        config_type = current_item.data(Qt.ItemDataRole.UserRole)
        if config_type == "builtin":
            QMessageBox.warning(self, "警告", "无法修改内置配置")
            return
            
        name = current_item.text().split(" ")[1]  # 提取名称
        
        try:
            content = self.matrices_editor.toPlainText()
            data = json.loads(content)
            
            if enhanced_config_manager.save_user_config("matrices", name, data):
                QMessageBox.information(self, "成功", f"已保存配置: {name}")
            else:
                QMessageBox.critical(self, "错误", "保存配置失败")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "错误", f"JSON格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            
    def _reload_app_settings(self):
        """重新加载应用设置"""
        self._load_app_settings()
        QMessageBox.information(self, "成功", "已重新加载应用设置")
        
    def _save_app_settings(self):
        """保存应用设置"""
        try:
            content = self.app_settings_editor.toPlainText()
            data = json.loads(content)
            
            # 更新配置管理器中的设置
            enhanced_config_manager.app_settings = data
            enhanced_config_manager._save_app_settings(data)
            
            QMessageBox.information(self, "成功", "已保存应用设置")
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "错误", f"JSON格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            
    def _backup_configs(self):
        """备份配置"""
        if enhanced_config_manager.backup_user_configs():
            QMessageBox.information(self, "成功", "配置备份完成")
        else:
            QMessageBox.critical(self, "错误", "配置备份失败")
