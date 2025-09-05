"""
CMA-ES 优化进度对话框
显示优化迭代进度、Delta E 值和其他相关信息
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont
import re


class CMAESProgressDialog(QDialog):
    """CMA-ES优化进度对话框"""
    
    # 线程安全的信号，用于从worker线程更新UI
    update_progress_signal = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("光谱锐化优化进度")
        self.setModal(True)
        self.resize(500, 400)
        
        # 优化状态
        self.is_running = False
        self.current_iteration = 0
        self.max_iterations = 300  # 默认值
        self.best_delta_e = float('inf')
        self.current_delta_e = float('inf')
        
        self._setup_ui()
        
        # 连接信号到槽函数，确保线程安全
        self.update_progress_signal.connect(self._update_progress_slot)
        
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("正在根据色卡优化光谱锐化参数...")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # 进度信息组
        progress_group = QGroupBox("优化进度")
        progress_layout = QVBoxLayout(progress_group)
        
        # 迭代进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # 进度信息标签
        info_layout = QHBoxLayout()
        self.iteration_label = QLabel("迭代: 0 / 300")
        self.delta_e_label = QLabel("当前 Delta E: --")
        self.best_delta_e_label = QLabel("最佳 Delta E: --")
        
        info_layout.addWidget(self.iteration_label)
        info_layout.addStretch()
        info_layout.addWidget(self.delta_e_label)
        info_layout.addStretch()
        info_layout.addWidget(self.best_delta_e_label)
        
        progress_layout.addLayout(info_layout)
        layout.addWidget(progress_group)
        
        # 详细日志
        log_group = QGroupBox("详细日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
    def set_max_iterations(self, max_iter: int):
        """设置最大迭代次数"""
        self.max_iterations = max_iter
        self.iteration_label.setText(f"迭代: {self.current_iteration} / {max_iter}")
        
    def start_optimization(self):
        """开始优化"""
        self.is_running = True
        self.current_iteration = 0
        self.best_delta_e = float('inf')
        self.current_delta_e = float('inf')
        
        self.cancel_button.setEnabled(True)
        self.close_button.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.add_log_message("开始CMA-ES优化...")
        
    def finish_optimization(self, success: bool, final_delta_e: float = None):
        """完成优化"""
        self.is_running = False
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        
        if success:
            if final_delta_e is not None:
                self.add_log_message(f"✓ 优化成功完成！最终 Delta E: {final_delta_e:.6f}")
            else:
                self.add_log_message("✓ 优化成功完成！")
            self.progress_bar.setValue(100)
        else:
            self.add_log_message("✗ 优化失败或被取消")
            
    def request_update_progress(self, message: str):
        """线程安全的进度更新请求接口"""
        # 发射信号，让槽函数在主线程中处理
        self.update_progress_signal.emit(message)
    
    @Slot(str)
    def _update_progress_slot(self, message: str):
        """槽函数：在主线程中更新进度信息"""
        print(f"[DEBUG] CMAESProgressDialog._update_progress_slot: '{message}'")
        print(f"[DEBUG] is_running: {self.is_running}")
        
        if not self.is_running:
            print(f"[DEBUG] 优化未运行，启动优化状态")
            self.start_optimization()
            
        # 解析CMA-ES消息
        if "迭代" in message:
            print(f"[DEBUG] 检测到迭代消息，解析中...")
            self._parse_iteration_message(message)
        elif "优化成功完成" in message:
            print(f"[DEBUG] 检测到优化完成消息")
            self.finish_optimization(True)
        elif "开始优化" in message:
            print(f"[DEBUG] 检测到开始优化消息")
            self.start_optimization()
        
        # 添加到日志
        self.add_log_message(message)
    
    # 为了向后兼容，保留原方法名但重定向到线程安全版本
    def update_progress(self, message: str):
        """更新进度信息（向后兼容方法）"""
        self.request_update_progress(message)
        
    def _parse_iteration_message(self, message: str):
        """解析迭代消息"""
        # 尝试提取迭代数和Delta E值
        patterns = [
            r'迭代\s*(\d+)\s*:\s*Delta E=([\d.]+)',
            r'迭代\s+(\d+)\s*:\s*Delta E=([\d.]+)',
            r'迭代.*?(\d+).*?Delta E=(\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match and len(match.groups()) >= 2:
                try:
                    iteration = int(match.group(1))
                    delta_e = float(match.group(2))
                    
                    self.current_iteration = iteration
                    self.current_delta_e = delta_e
                    
                    if delta_e < self.best_delta_e:
                        self.best_delta_e = delta_e
                    
                    # 更新界面
                    self.iteration_label.setText(f"迭代: {iteration} / {self.max_iterations}")
                    self.delta_e_label.setText(f"当前 Delta E: {delta_e:.6f}")
                    self.best_delta_e_label.setText(f"最佳 Delta E: {self.best_delta_e:.6f}")
                    
                    # 更新进度条
                    if self.max_iterations > 0:
                        progress = min(100, int(iteration * 100 / self.max_iterations))
                        self.progress_bar.setValue(progress)
                    
                    return
                except (ValueError, IndexError):
                    continue
                    
        # 如果无法解析，尝试只提取迭代数
        iteration_match = re.search(r'迭代.*?(\d+)', message)
        if iteration_match:
            try:
                iteration = int(iteration_match.group(1))
                self.current_iteration = iteration
                self.iteration_label.setText(f"迭代: {iteration} / {self.max_iterations}")
                
                if self.max_iterations > 0:
                    progress = min(100, int(iteration * 100 / self.max_iterations))
                    self.progress_bar.setValue(progress)
            except (ValueError, IndexError):
                pass
                
    def add_log_message(self, message: str):
        """添加日志消息"""
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_running:
            # 如果优化正在进行，询问是否取消
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "确认", "优化正在进行中，确定要取消吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reject()
            else:
                event.ignore()
                return
        
        super().closeEvent(event)