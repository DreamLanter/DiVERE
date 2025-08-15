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
  - 缩略图选择：提供 Lightroom 式小图网格用于选择 Scan/Crop，选中后再生成“所选项专属的较大 proxy”用于主预览。

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

### 批量导出（自动读取设置并导出到指定文件夹）
- 导出配置（位于 `divere.json.export`，名称+数值冗余策略贯穿）：
  - `output_dir`: 目标输出目录（相对批处理根目录或绝对路径）。
  - `format`: 输出格式（`TIFF`/`PNG`/`JPEG`/`EXR` 等，初版以 `TIFF`/`PNG` 为主）。
  - `bit_depth`: 位深（8/16/32f，随格式与通道支持约束）。
  - `color_space`: 目标显示/工作色彩空间名称（如 `DisplayP3`），内部按本地定义或回退定义注册；可选 `profile_embed`（是否内嵌ICC）。
  - `compression`: 压缩选项（如 `tiff_lzw`/`tiff_deflate`/`png_default`/`jpeg_q90` 等）。
  - `naming_template`（可选）：文件命名模板（默认：`{scan_base}/{crop_name}.tif`）。
  - `structure`（可选）：目录组织方式（`by_scan`|`flat`，默认 `by_scan`）。
  - `max_workers`（可选）：导出并发度（默认1，大图建议1-2）。
  - `resume`/`force`（可选）：是否断点续跑、是否强制重算。
- 命名模板占位符（便于简洁且可扩展）：
  - `{scan_base}` 源文件名（不含扩展）；`{crop_name}`；`{index}`（同一scan内从1开始）；
  - `{width}x{height}` 导出尺寸；`{cs_name}`/`{cs_hash}` 输入色彩空间；`{params_hash}` 合并参数哈希；`{ts}` 时间戳。
  - 例：`"{scan_base}/{index:02d}-{crop_name}-{params_hash}.tif"`。
- 处理流程（每个 Crop 一条“任务”）：
  1) 合并 preset（Project→Scan→Crop）并解析 `input_colorspace`（名称+定义+哈希）；
  2) 基于“区域裁切后再缩放”生成 Selection 级 proxy（仅用于预览），导出时使用原图裁切块；
  3) 进入 `FilmPipelineProcessor.apply_full_precision_pipeline`（chunked=true，tile 可配置），应用合并后的 `ColorGradingParams`；
  4) 转换到 `export.color_space`；
  5) 保存图像：按 `format/bit_depth/compression/profile_embed` 写出；
  6) 写入 sidecar 元数据（可选，`*.json` 同名文件）与/或在 `.divere/export_manifest.json` 记录清单项；
  7) 记录日志与耗时、像素数、峰值内存近似值（可选）。
- 清单与可重现（优雅简洁）：
  - `.divere/export_manifest.json`：为每个输出记录 `source`、`bbox`、`rotation`、`effective_input_cs`（name+def+hash）、`matrix_hash`、`curve_hash`、`params_hash`、`export_cfg_hash`、`output_path`、`status`、`duration_ms`、`time`。
  - 跳过策略：若 `resume=true` 且清单存在同 `job_hash`（由 source+bbox+rotation+全部参数哈希+导出配置哈希 组成）且 `status=done` 且目标文件存在，则跳过；`force=true` 则忽略并重算。
- 并发与内存：
  - 默认串行（`max_workers=1`）以控制峰值内存；当图像较小可提高并发。
  - 策略：`by_scan` 优先（同一大图内串行处理各 crop，避免多次加载原图）；多 scan 之间可并发受控执行。
- UI 集成（可选、最小侵入）：
  - “工具”菜单新增“批量导出”对话框：
    - 选择/确认 `output_dir`、`format`、`bit_depth`、`color_space`、`compression`、`naming_template`；
    - 复选 `resume`/`force`；并发度选择；
    - 进度条+可取消；错误汇总列表。
  - 逻辑仍委托 `BatchManager.run_export_all()`，UI 仅为参数收集与显示。
- 可扩展性：
  - 导出器薄接口（后续可独立模块化）：`Exporter` 协议：`save(array, path, options, icc_profile)`；
  - 初版直接由 `BatchManager` 内部根据 `format` 路由至 `PIL/imageio` 实现，后续可抽离到 `divere/utils/exporters.py` 并注册表驱动。
  - 通过命名模板与哈希策略实现无需修改代码即可扩展命名规范与跳过策略。

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

### 缩略图与选择机制（高效、优雅、最小改动）
- 分级产物与缓存：
  - Scan 级 proxy（中等尺寸，用于生成 crop 缩略图与快速预览）。
  - Thumbnail（小图 ≤256px）：
    - Scan 缩略图：由 Scan 级 proxy 直接等比缩小。
    - Crop 缩略图：在 Scan 级 proxy 上按 bbox 做 ROI 裁切后再缩小（避免为每个 Crop 单独读取原图）。
  - Selection 级 proxy（所选项专属的较大 proxy，用于主预览；尺寸= `PreviewConfig.proxy_max_size`）。
- 生成与触发：
  - 初次载入项目时：仅按需、懒生成小图（后台线程池），优先显示占位/渐进加载。
  - 用户点击缩略图后：根据所选 Scan/Crop 生成 Selection 级 proxy：
    - 优先从原图做“区域裁切后再缩放”生成（TIFF 可尝试惰性裁切），以获得更高质量；
    - 若不便或开销大，则回退为在 Scan 级 proxy 上做裁切并按需放大（速度优先）。
  - 以上产物统一以哈希命名并落地缓存：`.divere/thumbs/`（小图）、`.divere/proxies/`（中等/所选项专属 proxy），避免重复计算。
- 命名与一致性：
  - 哈希键包含：`文件路径 + bbox(若有) + rotation(若有) + 预览配置版本 + CS/矩阵/曲线哈希`。
  - 当参数变化影响预览呈现（如曲线/矩阵改变）时，可选择仅失效 Selection 级 proxy；小图仍可复用以保证交互流畅。
- UI 集成（最小侵入）：
  - 在 `divere/ui/main_window.py` 新增一个 `QDockWidget`（例如“缩略图”），内部用 `QListView/QListWidget` 自定义 delegate 以网格形式展示 Scan/Crop 缩略图；设置 `objectName` 由主题控制。
  - 数据源来自 `BatchManager` 提供的缩略图 API，后台异步填充；
  - 点击某项：
    1) 调用 `merge_presets` 得到该项的输入色彩空间与参数；
    2) 生成 Selection 级 proxy；
    3) 将参数与 input colorspace 应用到 `ParameterPanel` 与 `MainWindow`，触发布局与主预览刷新；
    4) 首次选择时调用 `fit_to_window()`（遵循约定）。


### 预览模式（双 Tab：接触印像 / 单张）
- **总体目标**：
  - **接触印像**：当选择≥2张时，自动将所选按顺序合并为一个“大照片”进行预览。
  - **单张**：当且仅当选择1张时可操作；在此 Tab 内提供一个极简控件按顺序切换所选集合中的照片。

- **Tab 切换与选择规则**：
  - **选择≥2**：自动切换到“接触印像”Tab；生成并显示合并后的大代理图（Contact Sheet Proxy）。
  - **选择=1**：自动切换到“单张”Tab；显示该项的 Selection 级 proxy。
  - **选择=0**：保留当前 Tab 状态并显示空态占位或提示。
  - **单张 Tab 内框选≥2**：自动切换到“接触印像”Tab。

- **接触印像渲染策略（预览）**：
  - 由 `BatchManager` 提供 `build_contact_sheet_proxy(selected_items, layout_cfg)`：
    - 各项先在各自 Selection 级代理上应用预览管线（每张独立按其合并后的参数计算），再按顺序拼接为一个大图；
    - 初版拼接策略简化为按顺序横向或纵向拼接，由 `PreviewConfig.contact_sheet` 控制（如 `mode: 'vertical'|'horizontal'`，`tile_max_size`）；后续可扩展为自适应网格。
  - 尺寸受 `PreviewConfig.contact_max_size` 限制；结果缓存为 `SelectionContactProxy`，Key 包含：所选路径+顺序+预览配置版本+各项参数哈希聚合。
  - 仅作为预览产物，不改变导出行为（本版不导出接触印像）。

- **单张 Tab 的极简切换控件**：
  - 提供“上一张/下一张”按钮与当前序号/总数指示；
  - 切换时：
    - 高亮缩略图中的对应项；
    - 将该项合并参数应用到 `ParameterPanel` 与 `MainWindow`，刷新预览；
    - 保持 `fit_to_window()` 语义不变（首次显示时触发）。

- **事件流与线程**：
  - 选择变化 → 判定目标 Tab → 触发相应代理生成请求：
    - “接触印像”：后台线程池拼接 Contact Sheet Proxy，生成期间显示渐进式占位；完成后替换主预览；
    - “单张”：后台生成或读取 Selection 级 proxy。
  - 参数变化导致预览失效时：仅失效受影响的 Selection 级代理；Contact Sheet Proxy 可选择整图失效或增量重建（首版整图重建）。

- **代码落点（最小侵入）**：
  - `divere/ui/main_window.py`：将主预览区域改为 `QTabWidget`（`objectName`: `previewTabs`），包含“接触印像”和“单张”两个 Tab；在创建处连接选择变化与框选信号，控制 Tab 自动切换；
  - `divere/ui/preview_widget.py`：
    - 保持现有单图预览；
    - 新增显示 Contact Sheet Proxy 的模式（仅接收已拼接好的大图并渲染，是否显示分隔边界由 QSS 控制，不在此处硬编码颜色）；
  - `divere/ui/parameter_panel.py`：无需变更；参数应用仍面向“当前所选单张”或“各自项在拼接前的单独处理”。
  - `divere/ui/theme.py`：为 `previewTabs` 与切换控件（`objectName`: `singleNav`）提供轻量 QSS。
  - `divere/utils/batch_manager.py`：新增 Contact Sheet 代理构建与缓存 API（见“接口草案”）。

- **非目标（首版）**：
  - 不提供接触印像的导出；
  - 不做复杂版式编辑（固定简单顺序拼接即可）。

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

    def run_export_all(self, project: BatchProject, max_workers: int = 1, resume: bool = True, force: bool = False) -> None: ...

    def compute_job_hash(self, scan_path: Path, bbox, rotation, preset, export_cfg) -> str: ...
    def build_output_path(self, export_cfg, placeholders: dict) -> Path: ...

    # 新增：接触印像（预览用）
    def build_contact_sheet_proxy(
        self,
        selected_items: list[tuple[Path, CropRegion]],
        layout_cfg: dict | None = None,
    ) -> ImageData: ...
```

### 实施步骤（最小改动，含冗余保存）
1. `divere/core/data_types.py` 增加轻量数据类：`ProcessingPreset`、`CropRegion`、`ScanItemSpec`、`BatchProject`。
2. 新增 `divere/utils/batch_manager.py`：读写 JSON、路径解析、三层合并（实现名称+数值冗余策略与哈希校验）、proxy 管理、导出与任务执行。
3. 扩展 `divere/__main__.py`：添加 `--batch` / `--export` 入口，调用 `BatchManager`。
4. UI（可选）：
   - `batch_dialog.py` + 菜单入口，应用合并后的 preset 到 `ParameterPanel` 与 `MainWindow.input_color_space`；
   - 主窗口预览引入双 Tab：`接触印像`/`单张`；在创建处连接选择变化与框选信号，实现自动切换；单张 Tab 增加极简顺序切换控件（`objectName: singleNav`）。
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


