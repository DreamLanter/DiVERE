from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
import json
import os

# Import debug logger
try:
    from .debug_logger import debug, info, warning, error, log_file_operation
except ImportError:
    # Fallback if debug logger is not available
    def debug(msg, module=None): pass
    def info(msg, module=None): pass
    def warning(msg, module=None): pass
    def error(msg, module=None): pass
    def log_file_operation(op, path, success=True, err=None, module=None): pass

@dataclass
class ClassificationRule:
    name: str
    conditions: List[Dict[str, Any]]
    default_preset: str

@dataclass
class FileInfo:
    path: str
    filename: str
    extension: str
    size_bytes: int
    directory: str

class SmartFileClassifier:
    """智能文件分类器 - 完全解耦的独立模块"""
    
    def __init__(self):
        self._rules: List[ClassificationRule] = []
        self._fallback_preset: str = "default.json"
        self._load_classification_rules()
    
    def classify_file(self, file_path: str) -> str:
        """根据文件特征智能分类，返回对应的默认预设文件名"""
        info(f"Classifying file: {file_path}", "SmartFileClassifier")
        
        file_info = self._extract_file_info(file_path)
        debug(f"File info: {file_info}", "SmartFileClassifier")
        debug(f"Available rules: {len(self._rules)}", "SmartFileClassifier")
        
        # 按rules.json中的顺序依次匹配，找到第一个匹配的规则
        for i, rule in enumerate(self._rules):
            debug(f"Evaluating rule {i+1}: {rule.name}", "SmartFileClassifier")
            if self._evaluate_rule_conditions(file_info, rule.conditions):
                info(f"Rule matched: {rule.name} -> {rule.default_preset}", "SmartFileClassifier")
                return rule.default_preset
        
        # 没有匹配的规则，返回fallback
        info(f"No rules matched, using fallback: {self._fallback_preset}", "SmartFileClassifier")
        return self._fallback_preset
    
    def _load_classification_rules(self):
        """加载分类规则配置"""
        info("Loading file classification rules", "SmartFileClassifier")
        
        try:
            from divere.utils.path_manager import find_file
            rules_path = find_file("file_classification_rules.json", "config")
            if not rules_path:
                error("file_classification_rules.json not found", "SmartFileClassifier")
                raise FileNotFoundError("找不到文件分类规则配置文件")
            
            info(f"Found classification rules at: {rules_path}", "SmartFileClassifier")
            log_file_operation("Load classification rules", rules_path, True, None, "SmartFileClassifier")
                
            with open(rules_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 解析规则
            self._rules = []
            rules_data = config.get("classification_rules", [])
            debug(f"Found {len(rules_data)} classification rules", "SmartFileClassifier")
            
            for rule_data in rules_data:
                rule = ClassificationRule(
                    name=rule_data.get("name", ""),
                    conditions=rule_data.get("conditions", []),
                    default_preset=rule_data.get("default_preset", "")
                )
                self._rules.append(rule)
                debug(f"Loaded rule: {rule.name} -> {rule.default_preset}", "SmartFileClassifier")
            
            self._fallback_preset = config.get("fallback", "defaults/default.json")
            info(f"Fallback preset: {self._fallback_preset}", "SmartFileClassifier")
            
        except Exception as e:
            error_msg = str(e)
            error(f"Failed to load classification rules: {error_msg}", "SmartFileClassifier")
            # 使用内置默认规则
            self._rules = []
            self._fallback_preset = "defaults/default.json"
            warning(f"Using fallback preset: {self._fallback_preset}", "SmartFileClassifier")
    
    def _extract_file_info(self, file_path: str) -> FileInfo:
        """提取文件的详细信息"""
        path = Path(file_path)
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
            
        return FileInfo(
            path=str(file_path),
            filename=path.name,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            directory=str(path.parent)
        )
    
    def _evaluate_rule_conditions(self, file_info: FileInfo, conditions: List[Dict[str, Any]]) -> bool:
        """评估规则条件 - 所有条件都必须满足（AND逻辑）"""
        for condition in conditions:
            if not self._evaluate_single_condition(file_info, condition):
                return False
        return True
    
    def _evaluate_single_condition(self, file_info: FileInfo, condition: Dict[str, Any]) -> bool:
        """评估单个条件"""
        condition_type = condition.get("type")
        operator = condition.get("operator", "equals")
        value = condition.get("value")
        
        if condition_type == "extension":
            return self._compare_strings(file_info.extension, value, operator)
        elif condition_type == "filename_contains":
            return self._compare_strings(file_info.filename, value, operator)
        elif condition_type == "path_contains":
            return self._compare_strings(file_info.path, value, operator)
        elif condition_type == "file_size":
            return self._compare_file_size(file_info.size_bytes, value, condition.get("unit", "B"), operator)
        elif condition_type == "regex":
            return self._match_regex(file_info.path, value)
        else:
            return False
    
    def _compare_strings(self, actual: str, expected: Union[str, List[str]], operator: str) -> bool:
        """比较字符串"""
        if operator == "equals":
            return actual.lower() == expected.lower()
        elif operator == "contains":
            return expected.lower() in actual.lower()
        elif operator == "contains_ignore_case":
            return expected.lower() in actual.lower()
        elif operator == "contains_any":
            if isinstance(expected, list):
                return any(exp.lower() in actual.lower() for exp in expected)
            return expected.lower() in actual.lower()
        elif operator == "contains_any_ignore_case":
            if isinstance(expected, list):
                return any(exp.lower() in actual.lower() for exp in expected)
            return expected.lower() in actual.lower()
        elif operator == "in":
            if isinstance(expected, list):
                return actual.lower() in [exp.lower() for exp in expected]
            return actual.lower() == expected.lower()
        elif operator == "starts_with":
            return actual.lower().startswith(expected.lower())
        elif operator == "ends_with":
            return actual.lower().endswith(expected.lower())
        return False
    
    def _compare_file_size(self, actual_bytes: int, expected_value: Union[int, float], unit: str, operator: str) -> bool:
        """比较文件大小"""
        unit_multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 * 1024,
            "GB": 1024 * 1024 * 1024
        }
        multiplier = unit_multipliers.get(unit.upper(), 1)
        expected_bytes = int(expected_value * multiplier)
        
        if operator == "greater_than":
            return actual_bytes > expected_bytes
        elif operator == "less_than":
            return actual_bytes < expected_bytes
        elif operator == "equals":
            return actual_bytes == expected_bytes
        elif operator == "greater_than_or_equal":
            return actual_bytes >= expected_bytes
        elif operator == "less_than_or_equal":
            return actual_bytes <= expected_bytes
        return False
    
    def _match_regex(self, text: str, pattern: str) -> bool:
        """正则表达式匹配"""
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False
