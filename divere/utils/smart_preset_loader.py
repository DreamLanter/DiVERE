from typing import Optional
from divere.core.data_types import Preset
import json

class SmartPresetLoader:
    """智能预设加载器 - 完全解耦的独立模块"""
    
    def __init__(self):
        pass
    
    def load_preset_by_name(self, preset_name: str) -> Optional[Preset]:
        """根据预设文件名加载预设"""
        try:
            from divere.utils.path_manager import resolve_path
            preset_path = resolve_path(preset_name)
            if not preset_path:
                raise FileNotFoundError(f"找不到预设文件: {preset_name}")
                
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Preset.from_dict(data)
        except Exception as e:
            print(f"加载预设文件 {preset_name} 失败: {e}")
            return None
    
    def get_smart_default_preset(self, file_path: str) -> Optional[Preset]:
        """获取文件的智能默认预设"""
        try:
            from divere.utils.smart_file_classifier import SmartFileClassifier
            classifier = SmartFileClassifier()
            preset_file = classifier.classify_file(file_path)
            
            return self.load_preset_by_name(preset_file)
            
        except Exception as e:
            print(f"智能分类失败: {e}")
            # 回退到通用默认
            return self.load_preset_by_name("default.json")
