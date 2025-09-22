"""
文件分类规则管理器
提供图形化界面来管理文件分类规则和默认预设文件
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget,
    QSplitter, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QIcon
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from divere.utils.path_manager import find_file, resolve_path


class FileClassificationManager(QMainWindow):
    """文件分类规则管理器主窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("文件分类规则管理器")
        self.setMinimumSize(1000, 700)
        
        # 数据
        self.rules_file_path = find_file("file_classification_rules.json", "config")
        if not self.rules_file_path:
            self.rules_file_path = "file_classification_rules.json"  # 回退到相对路径
        self.rules_data = self._load_rules()
        self.preset_files = self._scan_preset_files()
        
        # 当前编辑的规则索引
        self.current_rule_index = -1
        
        # 初始化UI
        self._init_ui()
        self._load_rules_to_table()
        
    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("文件分类规则管理器")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：规则列表和编辑
        left_widget = self._create_left_panel()
        splitter.addWidget(left_widget)
        
        # 右侧：预设文件管理
        right_widget = self._create_right_panel()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([600, 400])
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        save_btn = QPushButton("保存所有更改")
        save_btn.clicked.connect(self._save_all_changes)
        save_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        
        reload_btn = QPushButton("重新加载")
        reload_btn.clicked.connect(self._reload_rules)
        reload_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        
        bottom_layout.addWidget(save_btn)
        bottom_layout.addWidget(reload_btn)
        bottom_layout.addStretch()
        
        main_layout.addLayout(bottom_layout)
        
    def _create_left_panel(self) -> QWidget:
        """创建左侧面板：规则管理"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 规则列表
        rules_group = QGroupBox("分类规则列表")
        rules_layout = QVBoxLayout(rules_group)
        
        # 规则表格
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(3)
        self.rules_table.setHorizontalHeaderLabels(["规则名称", "条件数量", "默认预设"])
        self.rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.rules_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rules_table.itemSelectionChanged.connect(self._on_rule_selection_changed)
        rules_layout.addWidget(self.rules_table)
        
        # 规则操作按钮
        rules_btn_layout = QHBoxLayout()
        
        add_rule_btn = QPushButton("新建规则")
        add_rule_btn.clicked.connect(self._add_new_rule)
        
        delete_rule_btn = QPushButton("删除规则")
        delete_rule_btn.clicked.connect(self._delete_rule)
        
        rules_btn_layout.addWidget(add_rule_btn)
        rules_btn_layout.addWidget(delete_rule_btn)
        rules_btn_layout.addStretch()
        
        rules_layout.addLayout(rules_btn_layout)
        left_layout.addWidget(rules_group)
        
        # 规则编辑
        edit_group = QGroupBox("规则编辑")
        edit_layout = QVBoxLayout(edit_group)
        
        # 规则基本信息
        basic_layout = QGridLayout()
        
        basic_layout.addWidget(QLabel("规则名称:"), 0, 0)
        self.rule_name_edit = QLineEdit()
        basic_layout.addWidget(self.rule_name_edit, 0, 1)
        
        basic_layout.addWidget(QLabel("默认预设:"), 1, 0)
        self.default_preset_combo = QComboBox()
        self.default_preset_combo.addItems(self.preset_files)
        basic_layout.addWidget(self.default_preset_combo, 1, 1)
        
        edit_layout.addLayout(basic_layout)
        
        # 条件列表
        conditions_group = QGroupBox("匹配条件")
        conditions_layout = QVBoxLayout(conditions_group)
        
        # 条件表格
        self.conditions_table = QTableWidget()
        self.conditions_table.setColumnCount(4)
        self.conditions_table.setHorizontalHeaderLabels(["类型", "操作符", "值", "操作"])
        self.conditions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        conditions_layout.addWidget(self.conditions_table)
        
        # 条件操作按钮
        conditions_btn_layout = QHBoxLayout()
        
        add_condition_btn = QPushButton("添加条件")
        add_condition_btn.clicked.connect(self._add_condition)
        
        conditions_btn_layout.addWidget(add_condition_btn)
        conditions_btn_layout.addStretch()
        
        conditions_layout.addLayout(conditions_btn_layout)
        edit_layout.addWidget(conditions_group)
        
        left_layout.addWidget(edit_group)
        
        return left_widget
        
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板：预设文件管理"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 预设文件列表
        presets_group = QGroupBox("默认预设文件")
        presets_layout = QVBoxLayout(presets_group)
        
        # 预设文件表格
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(3)
        self.presets_table.setHorizontalHeaderLabels(["文件名", "类型", "操作"])
        self.presets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        presets_layout.addWidget(self.presets_table)
        
        # 预设文件操作按钮
        presets_btn_layout = QHBoxLayout()
        
        new_preset_btn = QPushButton("新建预设")
        new_preset_btn.clicked.connect(self._create_new_preset)
        
        open_preset_btn = QPushButton("打开预设")
        open_preset_btn.clicked.connect(self._open_preset)
        
        presets_btn_layout.addWidget(new_preset_btn)
        presets_btn_layout.addWidget(open_preset_btn)
        presets_btn_layout.addStretch()
        
        presets_layout.addLayout(presets_btn_layout)
        right_layout.addWidget(presets_group)
        
        # 预设文件编辑器
        editor_group = QGroupBox("预设文件编辑器")
        editor_layout = QVBoxLayout(editor_group)
        
        self.preset_editor = QTextEdit()
        self.preset_editor.setFont(QFont("Consolas", 10))
        editor_layout.addWidget(self.preset_editor)
        
        # 编辑器按钮
        editor_btn_layout = QHBoxLayout()
        
        save_preset_btn = QPushButton("保存预设")
        save_preset_btn.clicked.connect(self._save_preset)
        
        editor_btn_layout.addWidget(save_preset_btn)
        editor_btn_layout.addStretch()
        
        editor_layout.addLayout(editor_btn_layout)
        right_layout.addWidget(editor_group)
        
        return right_widget
        
    def _load_rules(self) -> Dict[str, Any]:
        """加载分类规则文件"""
        try:
            if os.path.exists(self.rules_file_path):
                with open(self.rules_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 创建默认规则文件
                default_rules = {
                    "classification_rules": [
                        {
                            "name": "Imacon扫描仪",
                            "conditions": [
                                {
                                    "type": "extension",
                                    "value": ".fff",
                                    "operator": "equals"
                                }
                            ],
                            "default_preset": "defaults/default_imacon.json"
                        }
                    ],
                    "fallback": "defaults/default.json"
                }
                return default_rules
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载规则文件失败: {e}")
            return {"classification_rules": [], "fallback": "defaults/default.json"}
            
    def _scan_preset_files(self) -> List[str]:
        """扫描预设文件"""
        preset_files = []
        try:
            from divere.utils.path_manager import list_default_presets
            preset_files = list_default_presets()
        except Exception:
            pass
        return sorted(preset_files)
        
    def _load_rules_to_table(self):
        """将规则加载到表格中"""
        self.rules_table.setRowCount(0)
        rules = self.rules_data.get("classification_rules", [])
        
        for i, rule in enumerate(rules):
            self.rules_table.insertRow(i)
            
            # 规则名称
            name_item = QTableWidgetItem(rule.get("name", ""))
            self.rules_table.setItem(i, 0, name_item)
            
            # 条件数量
            conditions_count = len(rule.get("conditions", []))
            count_item = QTableWidgetItem(str(conditions_count))
            self.rules_table.setItem(i, 1, count_item)
            
            # 默认预设
            preset_item = QTableWidgetItem(rule.get("default_preset", ""))
            self.rules_table.setItem(i, 2, preset_item)
            
    def _on_rule_selection_changed(self):
        """规则选择变化时的处理"""
        current_row = self.rules_table.currentRow()
        if current_row >= 0:
            self.current_rule_index = current_row
            self._load_rule_for_editing(current_row)
        else:
            self.current_rule_index = -1
            self._clear_rule_editing()
            
    def _load_rule_for_editing(self, rule_index: int):
        """加载规则到编辑界面"""
        rules = self.rules_data.get("classification_rules", [])
        if 0 <= rule_index < len(rules):
            rule = rules[rule_index]
            
            # 基本信息
            self.rule_name_edit.setText(rule.get("name", ""))
            
            preset_name = rule.get("default_preset", "")
            preset_index = self.default_preset_combo.findText(preset_name)
            if preset_index >= 0:
                self.default_preset_combo.setCurrentIndex(preset_index)
                
            # 条件列表
            self._load_conditions_to_table(rule.get("conditions", []))
            
    def _clear_rule_editing(self):
        """清空规则编辑界面"""
        self.rule_name_edit.clear()
        self.default_preset_combo.setCurrentIndex(0)
        self.conditions_table.setRowCount(0)
        
    def _load_conditions_to_table(self, conditions: List[Dict[str, Any]]):
        """将条件加载到条件表格中"""
        self.conditions_table.setRowCount(0)
        
        for i, condition in enumerate(conditions):
            self.conditions_table.insertRow(i)
            
            # 类型
            type_combo = QComboBox()
            type_combo.addItems(["extension", "filename_contains", "path_contains", "file_size", "regex"])
            type_combo.setCurrentText(condition.get("type", "extension"))
            type_combo.currentTextChanged.connect(lambda text, row=i: self._update_condition(row, "type", text))
            self.conditions_table.setCellWidget(i, 0, type_combo)
            
            # 操作符
            operator_combo = QComboBox()
            operator_combo.addItems(["equals", "contains", "contains_ignore_case", "contains_any", "contains_any_ignore_case", "in", "starts_with", "ends_with"])
            operator_combo.setCurrentText(condition.get("operator", "equals"))
            operator_combo.currentTextChanged.connect(lambda text, row=i: self._update_condition(row, "operator", text))
            self.conditions_table.setCellWidget(i, 1, operator_combo)
            
            # 值
            value_edit = QLineEdit()
            value_edit.setText(str(condition.get("value", "")))
            value_edit.textChanged.connect(lambda text, row=i: self._update_condition(row, "value", text))
            self.conditions_table.setCellWidget(i, 2, value_edit)
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda checked, row=i: self._delete_condition(row))
            self.conditions_table.setCellWidget(i, 3, delete_btn)
            
    def _update_condition(self, row: int, field: str, value: str):
        """更新条件字段"""
        if self.current_rule_index >= 0:
            rules = self.rules_data.get("classification_rules", [])
            if 0 <= self.current_rule_index < len(rules):
                rule = rules[self.current_rule_index]
                conditions = rule.get("conditions", [])
                if 0 <= row < len(conditions):
                    conditions[row][field] = value
                    
    def _add_new_rule(self):
        """添加新规则"""
        new_rule = {
            "name": "新规则",
            "conditions": [],
            "default_preset": "defaults/default.json"
        }
        
        self.rules_data["classification_rules"].append(new_rule)
        self._load_rules_to_table()
        
        # 选择新规则进行编辑
        new_row = len(self.rules_data["classification_rules"]) - 1
        self.rules_table.selectRow(new_row)
        
    def _delete_rule(self):
        """删除当前选中的规则"""
        if self.current_rule_index >= 0:
            reply = QMessageBox.question(
                self, "确认删除", 
                "确定要删除这个规则吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                rules = self.rules_data["classification_rules"]
                if 0 <= self.current_rule_index < len(rules):
                    del rules[self.current_rule_index]
                    self._load_rules_to_table()
                    self._clear_rule_editing()
                    self.current_rule_index = -1
                    
    def _add_condition(self):
        """添加新条件"""
        if self.current_rule_index >= 0:
            rules = self.rules_data["classification_rules"]
            if 0 <= self.current_rule_index < len(rules):
                rule = rules[self.current_rule_index]
                if "conditions" not in rule:
                    rule["conditions"] = []
                    
                new_condition = {
                    "type": "extension",
                    "operator": "equals",
                    "value": ""
                }
                
                rule["conditions"].append(new_condition)
                self._load_conditions_to_table(rule["conditions"])
                
    def _delete_condition(self, row: int):
        """删除条件"""
        if self.current_rule_index >= 0:
            rules = self.rules_data["classification_rules"]
            if 0 <= self.current_rule_index < len(rules):
                rule = rules[self.current_rule_index]
                conditions = rule.get("conditions", [])
                if 0 <= row < len(conditions):
                    del conditions[row]
                    self._load_conditions_to_table(conditions)
                    
    def _save_all_changes(self):
        """保存所有更改"""
        try:
            # 保存当前编辑的规则
            if self.current_rule_index >= 0:
                self._save_current_rule()
                
            # 保存到文件
            with open(self.rules_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.rules_data, f, indent=2, ensure_ascii=False)
                
            QMessageBox.information(self, "成功", "所有更改已保存")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            
    def _save_current_rule(self):
        """保存当前编辑的规则"""
        if self.current_rule_index >= 0:
            rules = self.rules_data["classification_rules"]
            if 0 <= self.current_rule_index < len(rules):
                rule = rules[self.current_rule_index]
                
                # 基本信息
                rule["name"] = self.rule_name_edit.text()
                rule["default_preset"] = self.default_preset_combo.currentText()
                
                # 条件
                conditions = []
                for row in range(self.conditions_table.rowCount()):
                    type_combo = self.conditions_table.cellWidget(row, 0)
                    operator_combo = self.conditions_table.cellWidget(row, 1)
                    value_edit = self.conditions_table.cellWidget(row, 2)
                    
                    if type_combo and operator_combo and value_edit:
                        condition = {
                            "type": type_combo.currentText(),
                            "operator": operator_combo.currentText(),
                            "value": value_edit.text()
                        }
                        conditions.append(condition)
                        
                rule["conditions"] = conditions
                
    def _reload_rules(self):
        """重新加载规则"""
        reply = QMessageBox.question(
            self, "确认重新加载", 
            "确定要重新加载规则文件吗？未保存的更改将丢失。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.rules_data = self._load_rules()
            self._load_rules_to_table()
            self._clear_rule_editing()
            self.current_rule_index = -1
            
    def _create_new_preset(self):
        """创建新预设文件"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "创建新预设文件", 
            resolve_data_path("config", "new_preset.json"),
            "JSON Files (*.json)"
        )
        
        if file_name:
            # 创建默认预设内容
            default_preset = {
                "version": 3,
                "type": "single",
                "metadata": {
                    "raw_file": "new_file.fff",
                    "orientation": 0,
                    "crop": [0.0, 0.0, 1.0, 1.0]
                },
                "idt": {
                    "name": "KodakEnduraPremier",
                    "gamma": 1.0,
                    "white": {"x": 0.3127, "y": 0.3290},
                    "primitives": {
                        "r": {"x": 0.6400, "y": 0.3300},
                        "g": {"x": 0.2100, "y": 0.7100},
                        "b": {"x": 0.1500, "y": 0.0600}
                    }
                },
                "cc_params": {
                    "density_gamma": 2.6,
                    "density_dmax": 2.0,
                    "rgb_gains": [1.0, 1.0, 1.0],
                    "density_matrix": {"name": "Cineon_States_M_to_Print_Density", "values": None},
                    "density_curve": {"name": "Kodak Endura Premier", "points": {"rgb": [[0.0, 0.0], [1.0, 1.0]]}}
                }
            }
            
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(default_preset, f, indent=2, ensure_ascii=False)
                    
                # 刷新预设文件列表
                self.preset_files = self._scan_preset_files()
                self.default_preset_combo.clear()
                self.default_preset_combo.addItems(self.preset_files)
                
                QMessageBox.information(self, "成功", "新预设文件已创建")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建预设文件失败: {e}")
                
    def _open_preset(self):
        """打开预设文件进行编辑"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开预设文件", 
            resolve_data_path("config"),
            "JSON Files (*.json)"
        )
        
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.preset_editor.setPlainText(content)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开预设文件失败: {e}")
                
    def _save_preset(self):
        """保存预设文件"""
        content = self.preset_editor.toPlainText()
        
        if not content.strip():
            QMessageBox.warning(self, "警告", "预设内容为空")
            return
            
        try:
            # 验证JSON格式
            json.loads(content)
            
            # 保存文件
            file_name, _ = QFileDialog.getSaveFileName(
                self, "保存预设文件", 
                resolve_data_path("config", "preset.json"),
                "JSON Files (*.json)"
            )
            
            if file_name:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                QMessageBox.information(self, "成功", "预设文件已保存")
                
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "错误", f"JSON格式错误: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = FileClassificationManager()
    window.show()
    sys.exit(app.exec())
