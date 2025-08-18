
"""
自动预设管理器
负责根据图片位置自动加载和保存调色预设。
"""

import json
from pathlib import Path
from typing import Dict, Optional

from divere.core.data_types import Preset


class AutoPresetManager:
    """
    管理与图像文件在同一目录下的自动预设文件。
    预设文件名为 'divere_presets.json'。
    """
    PRESET_FILENAME = "divere_presets.json"

    def __init__(self):
        self._presets: Dict[str, Preset] = {}
        self._preset_file_path: Optional[Path] = None

    def _load_presets_from_file(self) -> None:
        """从文件加载预设到缓存"""
        self._presets = {}
        if self._preset_file_path and self._preset_file_path.exists():
            try:
                with open(self._preset_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for filename, preset_data in data.items():
                        self._presets[filename] = Preset.from_dict(preset_data)
            except (IOError, json.JSONDecodeError):
                # 如果文件损坏或无法读取，则视为空文件
                pass

    def _save_presets_to_file(self) -> None:
        """将缓存中的预设保存到文件"""
        if self._preset_file_path:
            try:
                self._preset_file_path.parent.mkdir(parents=True, exist_ok=True)
                data_to_save = {
                    filename: preset.to_dict()
                    for filename, preset in self._presets.items()
                }
                with open(self._preset_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            except IOError:
                # 处理写入错误
                pass

    def set_active_directory(self, directory: str) -> None:
        """
        设置当前活动目录，并加载该目录下的预设文件。
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return

        new_preset_file = dir_path / self.PRESET_FILENAME
        if self._preset_file_path != new_preset_file:
            self._preset_file_path = new_preset_file
            self._load_presets_from_file()

    def get_preset_for_image(self, image_path: str) -> Optional[Preset]:
        """
        获取指定图像文件的预设。
        """
        image_filename = Path(image_path).name
        return self._presets.get(image_filename)

    def save_preset_for_image(self, image_path: str, preset: Preset) -> None:
        """
        为指定图像文件保存或更新预设。
        """
        image_filename = Path(image_path).name
        self._presets[image_filename] = preset
        self._save_presets_to_file()

    def get_current_preset_file_path(self) -> Optional[Path]:
        """返回当前预设文件的路径"""
        return self._preset_file_path
