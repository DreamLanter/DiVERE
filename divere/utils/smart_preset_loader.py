from typing import Optional
from divere.core.data_types import Preset
import json

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

class SmartPresetLoader:
    """智能预设加载器 - 完全解耦的独立模块"""
    
    def __init__(self):
        pass
    
    def load_preset_by_name(self, preset_name: str) -> Optional[Preset]:
        """根据预设文件名加载预设"""
        info(f"Loading preset by name: '{preset_name}'", "SmartPresetLoader")
        
        try:
            from divere.utils.path_manager import resolve_path
            info(f"Using path_manager.resolve_path() to find: '{preset_name}'", "SmartPresetLoader")
            
            preset_path = resolve_path(preset_name)
            if not preset_path:
                error(f"Path resolution failed for preset: '{preset_name}'", "SmartPresetLoader")
                raise FileNotFoundError(f"找不到预设文件: {preset_name}")
            
            info(f"Resolved preset path: {preset_path}", "SmartPresetLoader")
            log_file_operation("Load preset file", preset_path, True, None, "SmartPresetLoader")
                
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = Preset.from_dict(data)
            info(f"Successfully loaded preset: '{preset_name}' -> {preset.name}", "SmartPresetLoader")
            return preset
        except Exception as e:
            error_msg = str(e)
            error(f"Failed to load preset '{preset_name}': {error_msg}", "SmartPresetLoader")
            log_file_operation("Load preset file", preset_name, False, error_msg, "SmartPresetLoader")
            return None
    
    def get_smart_default_preset(self, file_path: str) -> Optional[Preset]:
        """获取文件的智能默认预设"""
        info(f"Getting smart default preset for file: {file_path}", "SmartPresetLoader")
        
        try:
            from divere.utils.smart_file_classifier import SmartFileClassifier
            classifier = SmartFileClassifier()
            preset_file = classifier.classify_file(file_path)
            
            info(f"File classification result: '{preset_file}'", "SmartPresetLoader")
            
            result = self.load_preset_by_name(preset_file)
            if result:
                info(f"Smart classification successful: {file_path} -> {preset_file}", "SmartPresetLoader")
            return result
            
        except Exception as e:
            error_msg = str(e)
            error(f"Smart classification failed for {file_path}: {error_msg}", "SmartPresetLoader")
            # 回退到通用默认
            info("Falling back to default.json", "SmartPresetLoader")
            return self.load_preset_by_name("default.json")
