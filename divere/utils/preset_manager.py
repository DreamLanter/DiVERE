"""
预设管理器
负责加载和保存调色预设文件
"""

import json
from pathlib import Path
from typing import Optional

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
