下面给出“裁剪（Crop）”的可扩展架构设计报告，兼顾当前单裁剪需求与未来“同一张照片多个裁剪”的演进路径。保持单向数据流、UI/核心解耦、小改动与可维护性。

### 目标与边界
- 当前需求
  - 预览区上方新增三个按钮：裁剪、关注、恢复。
  - 框选矩形（转圈虚线），保存为归一化百分比（基于原图尺寸）。
  - 关注：基于裁剪区域重建 proxy 并放大居中。
  - 恢复：回到“原图完整 proxy”并继续在其上绘制虚线。
  - 打开图像若预设包含裁剪：默认显示完整 proxy，并用虚线标注范围。
- 扩展需求（规划）
  - 支持一个图像多个裁剪（具名、可选中、可聚焦/恢复、可删改）。
  - 保持 preset 兼容（旧版单裁剪 → 新版多裁剪的一致映射）。

### 数据建模（先为多裁剪设计，当前场景只用一个）
- 新增数据类型（逻辑结构）
  - CropROI
    - id: str（UUID）
    - name: str（可选，默认“裁剪1/2/3...”）
    - rect_norm: (x, y, w, h) 浮点百分比；基于“原图原始朝向”的归一化坐标
    - enabled: bool（保留扩展）
    - tags: list[str]（保留扩展）
- ApplicationContext 状态
  - crops: list[CropROI]（当前实现可持有 0 或 1 个；未来直连 UI 列表/面板）
  - active_crop_id: Optional[str]
  - crop_focused: bool（是否以 active_crop 生成 proxy）
  - 兼容保留：单一 `crop` 的读取/写入映射到 crops[0]
- ImageData.metadata（用于 UI 显示 overlay 的无耦合信息）
  - source_wh: (orig_w, orig_h) 原图尺寸（每次构建 proxy 时设置）
  - crop_overlay: Optional[(x, y, w, h)] 供 UI 直接画；值为“原图归一化裁剪框”
  - crop_focused: bool（给 UI 判断是否应该绘制“贴边虚线”或省略）
  - active_crop_id: Optional[str]

这样，UI 只读 `ImageData.metadata` 就能知道怎么画虚线与如何换算坐标，避免向下查询上下文状态。

### 业务流与职责

- ApplicationContext（高内聚）
  - set_crops(crops: list[CropROI]) / set_single_crop(rect_norm)
  - set_active_crop(crop_id) / clear_active_crop()
  - set_crop_focused(bool)
  - focus_on_active_crop()：`crop_focused=True` → `_prepare_proxy()` 使用裁剪
  - restore_crop_preview()：`crop_focused=False` → `_prepare_proxy()` 用完整图，`metadata.crop_overlay` 仍传递
  - load_preset(preset)：若存在裁剪
    - 兼容：若只有 `crop` 字段 → 转为 `crops=[{id:'c1', rect_norm=...}]，active_crop_id='c1'`
    - 默认 `crop_focused=False`，以完整 proxy 显示
  - _prepare_proxy()（唯一改动点）
    - 计算 `orig_w, orig_h`（原图尺寸）
    - 如果 `crop_focused and active_crop`：
      - 在“原图数组”上先裁切（像素坐标 = rect_norm × (orig_w, orig_h)）
      - 再走既有流程（色彩、工作空间、密度管线）
    - 否则完整图生成 proxy
    - 补充 `metadata.source_wh = (orig_w, orig_h)`、`metadata.crop_overlay = active_crop.rect_norm or None`、`metadata.crop_focused = crop_focused`、`metadata.active_crop_id`
- PreviewWidget（低耦合）
  - UI：新增三个按钮（裁剪/关注/恢复）
  - 裁剪编辑模式（仅 UI 本地状态，产生矩形；支持拖动/缩放；未来可扩展多 ROI 编辑器）
  - 事件
    - crop_committed(rect_norm) → 仅把归一化结果交给上层（不直接处理图/数据）
    - request_focus_crop() → 上层决定是否重建 proxy
    - request_restore_crop() → 上层决定是否恢复完整 proxy
  - 绘制 overlay（marching ants）
    - 从 `current_image.metadata` 读取 `source_wh` 与 `crop_overlay`
    - 非聚焦时：按 `rect_norm * scale`（scale = proxy_size / source_wh）绘制虚线框
    - 聚焦时：可画一层贴到边缘的虚线（表示当前 proxy 就是裁剪区域），也可简化不画（按需）
    - marching ants：局部 QTimer 调整 dash offset；不触碰 Context
- MainWindow（协调）
  - 连接信号
    - PreviewWidget.crop_committed(rect_norm) → context.set_single_crop(rect_norm) + context.set_active_crop('c1' 或已有 id)
    - request_focus_crop() → context.focus_on_active_crop() + fit_after_next_preview=True
    - request_restore_crop() → context.restore_crop_preview() + fit_after_next_preview=False（或保留当前视图）
  - 预设保存/读取
    - 保存：把 `context.crops`（或单裁剪）写入 `Preset.crops`；兼容旧字段 `Preset.crop`
    - 读取：统一映射到 `context.crops`，默认 `crop_focused=False`

### 坐标/朝向的一致性
- 裁剪坐标始终基于“原图原始朝向”（即 load_image 时的自然方向）归一化记录，避免随代理或旋转变化而漂移。
- 预览朝向（旋转）在 `_prepare_proxy()` 的后期应用（我们当前实现已经是“生成代理后再旋转”），因此：
  - 聚焦裁剪：先对原图裁切，再处理色彩与密度，再应用朝向；视觉正确。
  - 恢复时：完整图 proxy + overlay 使用 `source_wh` 做换算，不受朝向影响（Overlay 在 UI 上用 pan/zoom 映射渲染）。

### 预设 Schema（向后兼容）
- 新（建议）
  - crops: [{ id, name, rect_norm:[x,y,w,h], enabled:true, tags:[] }, ...]
  - active_crop_id: Optional[str]
  - crop_focused: Optional[bool]（可不保存，运行态）
- 旧（兼容）
  - crop: [x,y,w,h]
- 写入策略
  - 若只有一个裁剪，既写新字段 `crops`，也可镜像写 `crop` 以便老版本读取（可选）
- 读取策略
  - 优先 `crops`，否则 fallback 到 `crop`

### 交互细节（当前与未来）
- 当前（单裁剪）
  - 裁剪按钮：进入一次性“框选-提交”模式 → 发 crop_committed(rect_norm)
  - 关注：focus_on_active_crop() → 重建 proxy → fit_to_window()
  - 恢复：restore_crop_preview() → 完整 proxy，保留 overlay
- 未来（多裁剪）
  - 需要一个“裁剪列表面板”（在 `ParameterPanel` 新增一页或右侧 `DockWidget`），支持增/删/改/选中/命名
  - PreviewWidget 仍然只负责框选与展示，列表管理交由面板
  - MainWindow 只做信号转发；Context 管状态与构建

### 性能与线程
- 聚焦模式下先裁切再生成 proxy，数据更小，整体更快。
- 所有重建在 `ApplicationContext` 的预览后台线程执行，不阻塞 UI。
- marching ants 动画仅在 UI 层，消耗可控。

### 失败与回退
- 没有 active_crop：关注按钮灰显；恢复按钮仅在有裁剪时可用。
- 数值异常（w/h 太小、越界）：UI 框选时归一化并 clamp 到 [0,1] 区间，最小尺寸阈值（如 8×8 像素）。

### 实施顺序（小步快跑）
1) ApplicationContext：新增 `crops`、`active_crop_id`、`crop_focused` 与 API；在 `_prepare_proxy()` 增加裁切逻辑；在 `ImageData.metadata` 写入 `source_wh` 与 `crop_overlay`。
2) PreviewWidget：新增 3 按钮（紧挨“居中”），框选模式与虚线绘制，发出信号（commit/focus/restore），订阅 `preview_updated` 即可。
3) MainWindow：连线；Preset 保存/读取对接 `Preset.crops` 与兼容字段。
4) 仅落地“单裁剪”路径：`crops` 只持 1 项（id 固定 'c1'），API 用 list 封装，方便未来直接扩展多裁剪。

