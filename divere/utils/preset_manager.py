"""
预设管理器
负责加载和保存调色预设文件
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from divere.core.data_types import Preset, ColorGradingParams


class PresetManager:
    """处理预设的加载和保存"""

    @staticmethod
    def save_preset(preset: Preset, file_path: str) -> None:
        """
        将Preset对象保存到JSON文件。

        Args:
            preset: 要保存的Preset对象。
            file_path: 目标文件路径。
        
        Raises:
            IOError: 如果文件写入失败。
        """
        try:
            path = Path(file_path)
            # 确保目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(preset.to_dict(), f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"无法保存预设到 {file_path}: {e}")

    @staticmethod
    def load_preset(file_path: str) -> Optional[Preset]:
        """
        从JSON文件加载Preset对象。

        Args:
            file_path: 源文件路径。

        Returns:
            加载的Preset对象，如果失败则返回None。
        
        Raises:
            IOError: 如果文件读取失败。
            ValueError: 如果JSON解析失败。
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"预设文件不存在: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return Preset.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析预设文件失败 {file_path}: {e}")
        except IOError as e:
            raise IOError(f"无法加载预设文件 {file_path}: {e}")

    @staticmethod
    def save_folder_default(preset_path: str, idt_data: Dict[str, Any], cc_params_data: Dict[str, Any]) -> None:
        """
        在divere_presets.json中保存文件夹默认设置。

        Args:
            preset_path: divere_presets.json文件路径
            idt_data: 输入变换数据
            cc_params_data: 色彩校正参数数据
        
        Raises:
            IOError: 如果文件读写失败。
        """
        try:
            path = Path(preset_path)
            
            # 读取现有数据，如果文件不存在则创建空字典
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            
            # 添加或更新folder_default条目
            data['folder_default'] = {
                'idt': idt_data,
                'cc_params': cc_params_data
            }
            
            # 确保目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存更新后的数据
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            raise IOError(f"无法保存文件夹默认设置到 {preset_path}: {e}")

    @staticmethod
    def load_folder_default(preset_path: str) -> Optional[Dict[str, Any]]:
        """
        从divere_presets.json中加载文件夹默认设置。

        Args:
            preset_path: divere_presets.json文件路径

        Returns:
            包含'idt'和'cc_params'的字典，如果不存在则返回None。
        """
        try:
            path = Path(preset_path)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get('folder_default')
        except (json.JSONDecodeError, IOError):
            return None

    @staticmethod
    def has_folder_default(preset_path: str) -> bool:
        """
        检查divere_presets.json中是否存在文件夹默认设置。

        Args:
            preset_path: divere_presets.json文件路径

        Returns:
            如果存在folder_default字段则返回True，否则返回False。
        """
        folder_default = PresetManager.load_folder_default(preset_path)
        return folder_default is not None


def apply_preset_to_params(preset: Preset, params: ColorGradingParams) -> None:
    """
    将预设中的参数“部分应用”到现有的ColorGradingParams实例。
    仅当 preset.grading_params 中存在键时才更新。

    Args:
        preset: 加载的Preset对象。
        params: 需要被更新的ColorGradingParams实例。
    """
    if preset and preset.grading_params:
        params.update_from_dict(preset.grading_params)
