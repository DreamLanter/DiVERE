## 批量处理与预设系统设计（DiVERE）
### 最初的prompt：
  我在想，用一个优雅的方式支持批量模式。有两种批量方式：
  - 一个图中好多个scan。这需要管理一下crops
  - 好几个独立的scan
  我希望用一个统一方便的逻辑来实现，并且尽可能高效、优雅。包括：
  - 一批胶片，每一张都可能好几百MB，不可能全load到内存中，因此需要各自生成proxy
  - 一批胶片，每一张的调色都可能不同，因此要想办法实现一个预设系统，保存所有的、ui上可能进行调整的操作，在文件夹内可以放一个json文件。保存每一张照片，或者每一张照片里的每一个crop，的所有设置（例如输入色彩空间、曲线等，不能只保存名字，还要保存基色、曲线选点等等）。

  帮我根据现在的实现，仔细思考一下，怎么样才能实现得很优雅。写一个markdown格式的计划出来。
### 目标
- **统一处理单元**: 以 Crop 为最小处理单元，兼容两种批量模式：
  - 单图包含多个扫描帧（多个 Crop）
  - 多个独立扫描（每图一个全幅 Crop）
- **分层参数继承**: Project → ScanItem → Crop 逐层合并，得到最终输入色彩空间与 `ColorGradingParams`。
- **高效与可重现**:
  - 预览：每张大图仅生成一次代理（proxy），Crop 仅做等比例裁切。
  - 导出：原图按 Crop 做窗口裁切，再走分块全精度管线，控制峰值内存。
  - JSON 预设落地包含所有“值”（矩阵、曲线点、基色等），保证跨机器可重放。

### 数据设计
- **文件位置**: 批处理目录内放置 `divere.json`（UTF-8，`ensure_ascii=False`），所有图像使用相对路径。
 - **核心结构（示意，已加入“名称+数值冗余”）**:
 ```json
 {
   "version": 1,
   "project": {
     "name": "MyBatch",
     "default_preset": {
       "input_colorspace": {
         "name": "Film_KodakRGB_Linear",
         "def": { "primaries_xy": [[...],[...],[...]], "white_point_xy": [...], "gamma": 1.0 },
         "hash": "cs_ab12"
       },
       "matrix": {
         "name": "Cineon_States_M_to_Print_Density",
         "values": [[1,0,0],[0,1,0],[0,0,1]],
         "hash": "mx_cdef"
       },
       "curve": {
         "name": "Kodak_Endura_Paper",
         "points": [[0,0],[1,1]],
         "hash": "cv_3456"
       },
       "params": { /* ColorGradingParams.to_dict() 完整字典（含数值曲线/矩阵/开关等） */ }
     }
   },
   "scans": [
     {
       "file": "scans/Roll01.tif",
       "preset": {
         "input_colorspace": {
           "name": "AdobeRGB_Linear",
           "def": { "primaries_xy": [[...],[...],[...]], "white_point_xy": [...], "gamma": 1.0 },
           "hash": "cs_9abc"
         },
         "matrix": { "name": "Identity", "values": [[1,0,0],[0,1,0],[0,0,1]], "hash": "mx_0000" },
         "curve": { "name": null, "points": [[0,0],[1,1]], "hash": "cv_1111" },
         "params": { }
       },
       "crops": [
         {
           "name": "frame-01",
           "bbox": [x, y, w, h],
           "rotation": 0,
           "preset": {
             "input_colorspace": {
               "name": "custom",
               "def": { "primaries_xy": [[...],[...],[...]], "white_point_xy": [...], "gamma": 1.0 },
               "hash": "cs_zzzz"
             },
             "matrix": { "name": "Cineon_States_M_to_Print_Density", "values": [[...],[...],[...]], "hash": "mx_7777" },
             "curve": { "name": "Kodak_Endura_Paper", "points": [[...],[...],[...]], "hash": "cv_8888" },
             "params": { }
           },
           "notes": ""
         }
       ]
     }
   ],
   "export": { "output_dir": "out", "format": "TIFF", "color_space": "DisplayP3", "bit_depth": 16 }
 }
 ```
- **说明**
  - `preset` 统一采用“名称+数值冗余”策略：
    - `input_colorspace` 为对象：`name`（参考名或 `custom`）+ `def`（`primaries_xy`/`white_point_xy`/`gamma`）+ `hash`（对 `def` 计算的短哈希，便于一致性校验）。
    - `matrix` 为对象：`name` + `values`（3x3）+ `hash`（对 `values` 计算的短哈希）。
    - `curve` 为对象：`name`（若来自内置曲线，否则可为 null）+ `points`（完整点位）+ `hash`（对 `points` 计算的短哈希）。
    - `params` 仍保存完整数值（曲线点、矩阵、开关等）。
  - `crops` 始终存在；“独立扫描”默认一个全幅 Crop `[0,0,w,h]`（或 `0` 宽高代表全幅）。
  - `rotation` 可选（0/90/180/270 或小角度）。

### 代码落点与职责
- 新增 `divere/utils/batch_manager.py`
  - 载入/校验/保存 `divere.json`；相对路径解析；三层 preset 合并（Project→Scan→Crop）。
  - 代理管理：每个 Scan 仅生成一次全图 proxy（`ImageManager.generate_proxy`），内存缓存；可选落磁盘 `.divere/proxies/<hash>.jpg`。
  - 预览：将 Crop 的 `[x,y,w,h]` 等比例映射到 proxy 上做裁切，交给 `TheEnlarger.apply_preview_pipeline`。
  - 导出：优先按格式做窗口裁切（TIFF 可尝试惰性裁切，失败则全图+裁切），调用 `FilmPipelineProcessor.apply_full_precision_pipeline`（启用分块）。
  - 执行器：顺序或限并发导出任务，控制内存峰值与吞吐。
- 轻量数据类（添加到 `divere/core/data_types.py`）
  - `ProcessingPreset`：`input_colorspace` | `custom_colorspace` | `params_dict`（即 `ColorGradingParams` 字典）。
  - `CropRegion`：`name`、`bbox`、`rotation`、`preset?`。
  - `ScanItemSpec`：`file`、`preset?`、`crops`。
  - `BatchProject`：`project`、`scans`、`export`。
- UI 最小改动（解耦，可后做）
  - 新增 `divere/ui/batch_dialog.py`：树/列表展示 `scans/crops`，点击将合并后的 preset 应用到现有 `ParameterPanel` 与 `MainWindow.input_color_space`，触发预览。
  - 在 `divere/ui/main_window.py`“工具”菜单加入口（信号在创建处连接）。
- 命令行入口
  - 扩展 `divere/__main__.py`：支持 `--batch /path/to/divere.json --export` 头less 导出。

### 预设与色彩空间（名称+数值冗余与一致性策略）
- `ColorGradingParams`：直接复用 `to_dict()/from_dict()`，曲线点、矩阵值、RGB 增益、开关等完整保存。
- 输入色彩空间：始终保存 `name + def`，加载时按以下优先级：
  1) 若本地存在同名空间且计算出的 `hash` 与保存一致，则直接使用本地定义；
  2) 否则使用 `def` 注册临时空间（名称可派生为 `SavedCS_<hash>`），并使用该临时空间，确保跨设备一致；
  3) 若同时提供 `name` 与 `def` 且二者不一致，优先 `def`（保证重放一致性），并记录告警日志。
- 密度校正矩阵：始终保存 `name + values`，加载时优先比较 `hash`，规则同上；`params` 中也保留 `correction_matrix`（数值）。
- 曲线：`params` 中已有完整点位；同时保存 `curve.name + points + hash` 仅用于校验与来源追踪。加载时以点位为准，若需要 UI 显示来源再尝试匹配名称+哈希。

### 代理/内存策略
- 预览：
  - 每张大图仅生成一次 proxy（尺寸用 `PreviewConfig.proxy_max_size`），Crop 按比例裁切代理 → 传入预览管线。
- 导出：
  - 优先窗口裁切后再进入全精度管线（分块参数复用 `FilmPipelineProcessor`：`full_pipeline_tile_size`、`full_pipeline_max_workers`）。
  - 若无法部分读取，回落为全图加载+裁切（仍由分块控制峰值内存）。

### 接口草案（示意）
```python
# divere/utils/batch_manager.py
class BatchManager:
    def __init__(self, batch_root: Path, preview_config: PreviewConfig):
        ...

    def load_project(self) -> BatchProject: ...
    def save_project(self, project: BatchProject) -> None: ...

    def merge_presets(self, project, scan, crop) -> tuple[str, ColorGradingParams, dict]:
        """
        返回 (effective_input_space_name, params, provenance):
          - 根据“名称+数值冗余”与优先级解析 input colorspace；
          - params 合并后确保包含数值矩阵与曲线点；
          - provenance 提供解析决策（是否使用了临时空间/是否哈希不一致等）。
        若 crop/scan 未提供，逐层回退使用上层默认。
        """

    def get_or_create_scan_proxy(self, scan_path: Path) -> ImageData: ...
    def crop_proxy(self, proxy_img: ImageData, bbox: tuple[int,int,int,int]) -> ImageData: ...

    def export_crop(self, scan_path: Path, bbox, preset, export_cfg) -> Path:
        """导出单个 Crop，返回输出文件路径。"""

    def run_export_all(self, project: BatchProject, max_workers: int = 1) -> None: ...
```

### 实施步骤（最小改动，含冗余保存）
1. `divere/core/data_types.py` 增加轻量数据类：`ProcessingPreset`、`CropRegion`、`ScanItemSpec`、`BatchProject`。
2. 新增 `divere/utils/batch_manager.py`：读写 JSON、路径解析、三层合并（实现名称+数值冗余策略与哈希校验）、proxy 管理、导出与任务执行。
3. 扩展 `divere/__main__.py`：添加 `--batch` / `--export` 入口，调用 `BatchManager`。
4. UI（可选）：`batch_dialog.py` + 菜单入口，应用合并后的 preset 到 `ParameterPanel` 与 `MainWindow.input_color_space`。
5. 回归测试：
   - 单图多 Crop；多图单 Crop；非 TIFF/大 TIFF；`custom_colorspace` 回放；曲线/矩阵数值一致性。
   - 不同设备/不同内置配置下的重放一致性（通过哈希比对与临时空间注册验证）。

### 风险与对策
- 非 TIFF 局部读取：首版若不稳定，回退为全图+裁切（由分块机制控内存）；后续可评估 `tifffile`。
- 代理磁盘缓存：视需要开启；跨会话秒开列表时落地 JPEG/WEBP 到 `.divere/proxies/`。

### 约束与规范
- JSON 读写统一 `utf-8` 与 `ensure_ascii=False`。
- 资源路径：以批处理目录为根的相对路径；全局路径解析仍通过 `divere/utils/app_paths.py` 提供的入口用于内置资源。
- UI 信号在创建处连接；主题通过 `divere/ui/theme.py::apply_theme` 统一处理；不在局部硬编码颜色。
- 性能：预览使用 proxy；导出分块与限并发执行，避免峰值内存过高。


