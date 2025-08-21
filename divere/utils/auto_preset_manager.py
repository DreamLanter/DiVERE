
"""
自动预设管理器
负责根据图片位置自动加载和保存调色预设。
"""

import json
from pathlib import Path
from typing import Dict, Optional

from divere.core.data_types import Preset, PresetBundle


class AutoPresetManager:
    """
    管理与图像文件在同一目录下的自动预设文件。
    预设文件名为 'divere_presets.json'。
    """
    PRESET_FILENAME = "divere_presets.json"

    def __init__(self):
        # 兼容：同时支持单 Preset 与 Bundle
        self._presets: Dict[str, Preset] = {}
        self._bundles: Dict[str, PresetBundle] = {}
        self._preset_file_path: Optional[Path] = None

    def _load_presets_from_file(self) -> None:
        """从文件加载预设到缓存"""
        self._presets = {}
        self._bundles = {}
        if self._preset_file_path and self._preset_file_path.exists():
            try:
                with open(self._preset_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for filename, payload in data.items():
                        # 新格式：bundle
                        if isinstance(payload, dict) and payload.get("type") == "bundle":
                            try:
                                self._bundles[filename] = PresetBundle.from_dict(payload)
                            except Exception:
                                # 回退：尝试按单 preset 解析
                                self._presets[filename] = Preset.from_dict(payload)
                        else:
                            # 旧格式：单 preset
                            self._presets[filename] = Preset.from_dict(payload)
            except (IOError, json.JSONDecodeError):
                # 如果文件损坏或无法读取，则视为空文件
                pass

    def _save_presets_to_file(self) -> None:
        """将缓存中的预设保存到文件"""
        if self._preset_file_path:
            try:
                self._preset_file_path.parent.mkdir(parents=True, exist_ok=True)
                # 优先保存 bundle；若没有 bundle 条目，则保存单 preset；允许混合，便于渐进迁移
                data_to_save: Dict[str, dict] = {}
                for filename, bundle in self._bundles.items():
                    data_to_save[filename] = bundle.to_dict()
                for filename, preset in self._presets.items():
                    # 若同名 bundle 已存在，则跳过旧 preset
                    if filename not in data_to_save:
                        data_to_save[filename] = preset.to_dict()
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
        # 若存在 bundle，返回其 contactsheet 作为默认入口（保持旧接口语义）
        if image_filename in self._bundles:
            return self._bundles[image_filename].contactsheet
        return self._presets.get(image_filename)

    def save_preset_for_image(self, image_path: str, preset: Preset) -> None:
        """
        为指定图像文件保存或更新预设。
        """
        image_filename = Path(image_path).name
        # 若该图像已有 bundle，则更新其 contactsheet；否则按旧格式保存单 preset
        if image_filename in self._bundles:
            self._bundles[image_filename].contactsheet = preset
        else:
            self._presets[image_filename] = preset
        self._save_presets_to_file()

    def get_current_preset_file_path(self) -> Optional[Path]:
        """返回当前预设文件的路径"""
        return self._preset_file_path

    # === 新增：Bundle 接口 ===
    def get_bundle_for_image(self, image_path: str) -> Optional[PresetBundle]:
        image_filename = Path(image_path).name
        return self._bundles.get(image_filename)

    def save_bundle_for_image(self, image_path: str, bundle: PresetBundle) -> None:
        image_filename = Path(image_path).name
        self._bundles[image_filename] = bundle
        # 迁移期：若存在同名旧 preset，移除以避免歧义
        if image_filename in self._presets:
            del self._presets[image_filename]
        self._save_presets_to_file()
