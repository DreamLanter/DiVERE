from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool, QTimer
from typing import Optional
import numpy as np
from pathlib import Path

from .data_types import ImageData, ColorGradingParams, Preset
from .image_manager import ImageManager
from .color_space import ColorSpaceManager
from .the_enlarger import TheEnlarger
from ..utils.auto_preset_manager import AutoPresetManager


class _PreviewWorkerSignals(QObject):
    result = Signal(ImageData)
    error = Signal(str)
    finished = Signal()


class _PreviewWorker(QRunnable):
    def __init__(self, image: ImageData, params: ColorGradingParams, the_enlarger: TheEnlarger, color_space_manager: ColorSpaceManager):
        super().__init__()
        self.image = image
        self.params = params
        self.the_enlarger = the_enlarger
        self.color_space_manager = color_space_manager
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            result_image = self.the_enlarger.apply_full_pipeline(self.image, self.params)
            result_image = self.color_space_manager.convert_to_display_space(result_image, "DisplayP3")
            self.signals.result.emit(result_image)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n{tb}")
        finally:
            self.signals.finished.emit()


class ApplicationContext(QObject):
    """
    应用上下文，作为单一数据源 (Single Source of Truth)。
    管理应用状态、核心业务逻辑和与UI的交互。
    """
    # =================
    # 信号 (Signals)
    # =================
    image_loaded = Signal()
    preview_updated = Signal(ImageData)
    params_changed = Signal(ColorGradingParams)
    status_message_changed = Signal(str)
    autosave_requested = Signal()
    # 裁剪变化（None 或 (x,y,w,h) 归一化）
    crop_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # =================
        # 核心服务实例
        # =================
        self.image_manager = ImageManager()
        self.color_space_manager = ColorSpaceManager()
        self.the_enlarger = TheEnlarger()
        self.auto_preset_manager = AutoPresetManager()

        # =================
        # 状态变量
        # =================
        self._current_image: Optional[ImageData] = None
        self._current_proxy: Optional[ImageData] = None # 应用输入变换和工作空间变换后的代理
        self._current_params: ColorGradingParams = self._create_default_params()
        # 图像朝向（以90度为步进，取值 {0,90,180,270}），统一放在Context
        self._current_orientation: int = 0
        
        # =================
        # 后台处理
        # =================
        self._preview_busy: bool = False
        self._preview_pending: bool = False
        self.thread_pool: QThreadPool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)

        # AI自动校色迭代状态
        self._auto_color_iterations = 0
        self._get_preview_for_auto_color_callback = None

        # 防抖自动保存
        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(500) # 500ms delay for autosave
        self._autosave_timer.timeout.connect(self.autosave_requested.emit)
        
        # 裁剪状态
        self._crops: list = []  # list of dict: { 'id': str, 'rect': (x,y,w,h), 'name': str }
        self._active_crop_id: Optional[str] = None
        self._crop_focused: bool = False

    def _create_default_params(self) -> ColorGradingParams:
        params = ColorGradingParams()
        params.density_gamma = 2.6
        params.density_matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array("Cineon_States_M_to_Print_Density")
        params.density_matrix_name = "Cineon_States_M_to_Print_Density"
        params.enable_density_matrix = True
        
        # 加载默认曲线
        try:
            from ..utils.app_paths import resolve_data_path
            import json
            
            curve_path = resolve_data_path("config", "curves", "Kodak_Endura_Paper.json")
            if curve_path.exists():
                with open(curve_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'curves' in data and 'RGB' in data['curves']:
                    params.curve_points = data['curves']['RGB']
                    params.density_curve_name = "Kodak_Endura_Paper"
        except Exception as e:
            print(f"加载默认曲线失败: {e}")
            
        return params

    # =================
    # 属性访问器 (Getters)
    # =================
    def get_current_image(self) -> Optional[ImageData]:
        return self._current_image
        
    def get_current_params(self) -> ColorGradingParams:
        return self._current_params

    def get_input_color_space(self) -> str:
        return self._current_params.input_color_space_name

    # =================
    # 裁剪（Crops）API
    # =================
    def set_single_crop(self, rect_norm: tuple[float, float, float, float]) -> None:
        """设置/替换单一裁剪，并设为激活；不自动聚焦。"""
        try:
            x, y, w, h = [float(max(0.0, min(1.0, v))) for v in rect_norm]
            # 规范到图像范围内
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            crop_id = 'c1'
            self._crops = [{ 'id': crop_id, 'rect': (x, y, w, h), 'name': '裁剪1' }]
            self._active_crop_id = crop_id
            self._crop_focused = False
            self.crop_changed.emit((x, y, w, h))
            # 触发自动保存（裁剪变化也应持久化）
            self._autosave_timer.start()
        except Exception:
            pass

    def get_active_crop(self) -> Optional[tuple[float, float, float, float]]:
        if not self._active_crop_id:
            return None
        for c in self._crops:
            if c.get('id') == self._active_crop_id:
                return tuple(c.get('rect'))  # type: ignore
        return None

    def clear_crop(self) -> None:
        self._crops = []
        self._active_crop_id = None
        self._crop_focused = False
        self.crop_changed.emit(None)
        self._autosave_timer.start()

    def focus_on_active_crop(self) -> None:
        if self.get_active_crop() is None or not self._current_image:
            return
        self._crop_focused = True
        self._prepare_proxy()
        self._trigger_preview_update()
        self._autosave_timer.start()

    def restore_crop_preview(self) -> None:
        if not self._current_image:
            return
        self._crop_focused = False
        self._prepare_proxy()
        self._trigger_preview_update()
        self._autosave_timer.start()

    # =================
    # 核心业务逻辑 (Actions)
    # =================
    def load_image(self, file_path: str):
        try:
            self.status_message_changed.emit(f"正在加载图像: {file_path}...")
            self._current_image = self.image_manager.load_image(file_path)
            # 重置朝向
            self._current_orientation = 0
            # 重置裁剪
            self._crops = []
            self._active_crop_id = None
            self._crop_focused = False
            self.crop_changed.emit(None)
            
            # 检查并应用自动预设
            self.auto_preset_manager.set_active_directory(str(Path(file_path).parent))
            preset = self.auto_preset_manager.get_preset_for_image(file_path)

            if preset:
                self.load_preset(preset)
                self.status_message_changed.emit(f"已为图像加载自动预设: {preset.name}")
            else:
                # 如果没有预设，则重置为默认参数
                self.reset_params()
                self.status_message_changed.emit("未找到预设，已应用默认参数")

            self._prepare_proxy()
            
            self.image_loaded.emit()
            self._trigger_preview_update()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.status_message_changed.emit(f"无法加载图像: {e}")

    def load_preset(self, preset: Preset):
        """从Preset对象加载状态"""
        # 1. 更新方向
        self._current_orientation = preset.orientation

        # 2. 更新输入色彩变换
        if preset.input_transformation and preset.input_transformation.name:
            new_params = self._current_params.copy() # Create a copy to avoid modifying the current params directly
            new_params.input_color_space_name = preset.input_transformation.name
            self.update_params(new_params)

        # 3. 更新调色参数
        new_params = ColorGradingParams.from_dict(preset.grading_params or {})
        
        # 兼容处理矩阵和曲线
        if preset.density_matrix:
            new_params.density_matrix_name = preset.density_matrix.name
            if preset.density_matrix.values:
                new_params.density_matrix = np.array(preset.density_matrix.values)
                # 预设显式包含矩阵数值时，自动启用密度矩阵
                new_params.enable_density_matrix = True
            elif preset.density_matrix.name:
                matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array(preset.density_matrix.name)
                new_params.density_matrix = matrix
                # 预设指定了矩阵名称（且可解析）时，自动启用密度矩阵
                if matrix is not None:
                    new_params.enable_density_matrix = True

        if preset.density_curve:
            new_params.density_curve_name = preset.density_curve.name
            if preset.density_curve.points:
                new_params.curve_points = preset.density_curve.points
        
        self.update_params(new_params)

        # 4. 加载裁剪（多裁剪优先；兼容旧字段）
        try:
            if preset.crops and isinstance(preset.crops, list) and len(preset.crops) > 0:
                # 仅映射第一项到单裁剪（当前阶段）
                first = preset.crops[0]
                rect = tuple(first.get('rect_norm') or first.get('rect') or [])
                if rect and len(rect) == 4:
                    self.set_single_crop(rect)  # type: ignore[arg-type]
                    self._active_crop_id = first.get('id', 'c1')
                    self._crop_focused = False
            elif preset.crop and len(preset.crop) == 4:
                self.set_single_crop(tuple(preset.crop))
                self._crop_focused = False
        except Exception:
            pass
        
    def update_params(self, new_params: ColorGradingParams):
        """由UI调用以更新参数"""
        self._current_params = new_params
        self.params_changed.emit(self._current_params)
        self._trigger_preview_update()
        self._autosave_timer.start() # 每次参数变更都启动自动保存计时器
    
    def set_input_color_space(self, space_name: str):
        self._current_params.input_color_space_name = space_name
        self.params_changed.emit(self._current_params)
        if self._current_image:
            self._prepare_proxy()
            self._trigger_preview_update()

    def reset_params(self):
        self._current_params = self._create_default_params()
        self.params_changed.emit(self._current_params)
        self.status_message_changed.emit("参数已重置")

    def run_auto_color_correction(self, get_preview_callback):
        """执行AI自动白平衡"""
        preview_image = get_preview_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit("自动校色失败：无预览图像")
            return

        try:
            gains_t = self.the_enlarger.calculate_auto_gain_learning_based(preview_image)
            gains = np.array(gains_t[:3])
            
            current_gains = np.array(self._current_params.rgb_gains)
            new_gains = np.clip(current_gains + gains, -2.0, 2.0)
            
            self._current_params.rgb_gains = tuple(new_gains)
            self.params_changed.emit(self._current_params)
            self._trigger_preview_update()
            self._autosave_timer.start() # AI校准后也触发自动保存
            self.status_message_changed.emit(f"AI自动校色完成. Gains: R={gains[0]:.2f}, G={gains[1]:.2f}, B={gains[2]:.2f}")
        except Exception as e:
            self.status_message_changed.emit(f"AI自动校色失败: {e}")

    def run_iterative_auto_color(self, get_preview_callback, max_iterations=10):
        """执行迭代式AI自动白平衡"""
        self._auto_color_iterations = max_iterations
        self._get_preview_for_auto_color_callback = get_preview_callback
        self._perform_auto_color_iteration() # Start the first iteration

    def _perform_auto_color_iteration(self):
        if self._auto_color_iterations <= 0 or not self._get_preview_for_auto_color_callback:
            self._get_preview_for_auto_color_callback = None # Clean up
            return

        preview_image = self._get_preview_for_auto_color_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit("自动校色迭代中止：无预览图像")
            self._get_preview_for_auto_color_callback = None # Clean up
            return

        try:
            gains_t = self.the_enlarger.calculate_auto_gain_learning_based(preview_image)
            gains = np.array(gains_t[:3])

            current_gains = np.array(self._current_params.rgb_gains)
            new_gains = np.clip(current_gains + gains, -2.0, 2.0)
            
            self._auto_color_iterations -= 1
            
            # If gains are very small, stop iterating
            if np.allclose(current_gains, new_gains, atol=1e-3):
                self.status_message_changed.emit("AI自动校色收敛，已停止")
                self._auto_color_iterations = 0
                self._get_preview_for_auto_color_callback = None
                self._autosave_timer.start() # Save on convergence
                return

            self._current_params.rgb_gains = tuple(new_gains)
            self.params_changed.emit(self._current_params)
            self._trigger_preview_update() # This will trigger the next iteration via _on_preview_result
            self._autosave_timer.start() # Save after each iteration step
            self.status_message_changed.emit(f"AI校色迭代剩余: {self._auto_color_iterations}")

        except Exception as e:
            self.status_message_changed.emit(f"AI自动校色迭代失败: {e}")
            self._auto_color_iterations = 0
            self._get_preview_for_auto_color_callback = None


    def _prepare_proxy(self):
        if not self._current_image:
            return
        
        # 源图（可能用于裁剪）
        src_image = self._current_image
        orig_h, orig_w = src_image.height, src_image.width

        # 若聚焦裁剪：先裁切原图副本
        if self._crop_focused:
            rect = self.get_active_crop()
            if rect is not None and src_image.array is not None:
                try:
                    x, y, w, h = rect
                    x0 = int(round(x * orig_w)); y0 = int(round(y * orig_h))
                    x1 = int(round((x + w) * orig_w)); y1 = int(round((y + h) * orig_h))
                    x0 = max(0, min(orig_w - 1, x0)); x1 = max(x0 + 1, min(orig_w, x1))
                    y0 = max(0, min(orig_h - 1, y0)); y1 = max(y0 + 1, min(orig_h, y1))
                    cropped_arr = src_image.array[y0:y1, x0:x1, :].copy()
                    src_image = src_image.copy_with_new_array(cropped_arr)
                except Exception:
                    pass

        proxy = self.image_manager.generate_proxy(
            src_image,
            self.the_enlarger.preview_config.get_proxy_size_tuple()
        )
        proxy = self.color_space_manager.set_image_color_space(
            proxy, self._current_params.input_color_space_name
        )
        self._current_proxy = self.color_space_manager.convert_to_working_space(
            proxy
        )
        # 应用朝向旋转到代理图像（仅预览路径）
        if self._current_proxy and self._current_proxy.array is not None and self._current_orientation % 360 != 0:
            try:
                import numpy as np
                k = (self._current_orientation // 90) % 4
                if k != 0:
                    rotated = np.rot90(self._current_proxy.array, k=int(k))
                    self._current_proxy = self._current_proxy.copy_with_new_array(rotated)
            except Exception:
                pass
        
        # 注入裁剪可视化元数据（供PreviewWidget绘制）
        try:
            md = self._current_proxy.metadata
            md['source_wh'] = (int(orig_w), int(orig_h))
            md['crop_overlay'] = self.get_active_crop()
            md['crop_focused'] = bool(self._crop_focused)
            md['active_crop_id'] = self._active_crop_id
            md['orientation'] = int(self._current_orientation)
        except Exception:
            pass

    def _trigger_preview_update(self):
        if not self._current_proxy:
            return

        if self._preview_busy:
            self._preview_pending = True
            return

        self._preview_busy = True
        
        from copy import deepcopy
        proxy_copy = self._current_proxy.copy()
        params_copy = deepcopy(self._current_params)

        worker = _PreviewWorker(
            image=proxy_copy,
            params=params_copy,
            the_enlarger=self.the_enlarger,
            color_space_manager=self.color_space_manager,
        )
        worker.signals.result.connect(self._on_preview_result)
        worker.signals.error.connect(self._on_preview_error)
        worker.signals.finished.connect(self._on_preview_finished)
        self.thread_pool.start(worker)

    def _on_preview_result(self, result_image: ImageData):
        self.preview_updated.emit(result_image)
        # If an iterative auto color is in progress, trigger the next step
        if self._auto_color_iterations > 0 and self._get_preview_for_auto_color_callback:
            QTimer.singleShot(0, self._perform_auto_color_iteration)


    def _on_preview_error(self, message: str):
        self.status_message_changed.emit(f"预览更新失败: {message}")
        self._auto_color_iterations = 0 # Stop iteration on error
        self._get_preview_for_auto_color_callback = None

    def _on_preview_finished(self):
        self._preview_busy = False
        if self._preview_pending:
            self._preview_pending = False
            self._trigger_preview_update()

    # =================
    # 方向与旋转（UI调用）
    # =================
    def get_current_orientation(self) -> int:
        return int(self._current_orientation)

    def set_orientation(self, degrees: int):
        try:
            deg = int(degrees) % 360
            # 规范到 0/90/180/270
            choices = [0, 90, 180, 270]
            closest = min(choices, key=lambda x: abs(x - deg))
            self._current_orientation = closest
            if self._current_image:
                self._prepare_proxy()
                self._trigger_preview_update()
        except Exception:
            pass

    def rotate(self, direction: int):
        """direction: 1=左旋+90°, -1=右旋-90°"""
        try:
            step = 90 if int(direction) >= 0 else -90
            new_deg = (self._current_orientation + step) % 360
            self.set_orientation(new_deg)
        except Exception:
            pass
