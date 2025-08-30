### 预设文件规范（Preset Schema）v3

本规范描述了预设文件在存储与交换时的统一结构。目标：
- 保持与现有数据流兼容（非侵入式迁移）。
- 将文件分为 metadata 与 核心参数 两大部分，语义清晰、便于扩展。
- 始终同时保存“预设名 + 预设数据”；当只有名字缺少数据时，运行时应按名字查找内置/外部资源。

---

#### 顶层结构

```json
{
  "version": 3,
  "type": "single | contactsheet",
  "metadata": { ... },
  "idt": { ... },
  "cc_params": { ... },
  "active_crop_id": "crop_1",   
  "crops": [ ... ]
}
```

- version: 数字。schema 版本，当前为 3。
- type: 字符串。取值 "single" 或 "contactsheet"。
  - single: 单张图像（或单个裁剪）的预设文件。
  - contactsheet: 一组裁剪及其各自参数的合集（联系表）。
- metadata: 文件级元数据（详见下文）。
- idt: 输入设备变换（Input Device Transform），即原 `input_transformation` 的语义升级。
- cc_params: 调色参数集合（Color Correction Parameters）。
- crops: 仅当 `type = contactsheet` 时出现；每个元素描述一个裁剪的元数据与其核心参数。
 - active_crop_id: 仅当 `type = contactsheet` 时使用；记录上次活跃裁剪的标识。

---

### 一、metadata（元数据）

文件级 metadata 字段建议：

- json 字段
  - version: 同顶层 version（为方便解析，可在 metadata 内镜像保存；解析时以顶层为准）。
  - type: 同顶层 type（可镜像）。

- 源素材信息
  - raw_file: string，必填。原始文件名（不含路径）。
  - scanner: string，可选。扫描仪/数字化设备的型号或标识。
  - light_source: string，可选。光源信息（如 D50/日光/LED 等）。
  - film_stock: string，可选。胶卷类型/批号（如 Kodak Portra 400）。
  - note: string，可选。任意备注。
  - created_at / updated_at: string，可选。ISO8601 时间戳。
  - app_version: string，可选。生成/最后编辑的应用版本。

- 视图/几何
  - orientation: number，取值 {0, 90, 180, 270}；以度为单位，右手规则，绝对旋转。
  - crop: [x, y, w, h]，均为 0~1 的归一化比例，表示相对于原始图像的裁剪窗口。

当 `type = contactsheet`：

- crops: 数组，每个元素包含：
  - metadata: 对应裁剪的元数据（建议至少包含 note、orientation、crop；可选 id/name 标签）。
  - idt: 该裁剪的 IDT（若省略，继承文件级 idt）。
  - cc_params: 该裁剪的调色参数（若省略，继承文件级 cc_params）。
- active_crop_id: 可选。上次活跃裁剪的标识。

裁剪级 metadata 字段建议：

- id: string，可选。裁剪唯一标识（推荐使用）。
- name: string，可选。裁剪名称（便于 UI 展示）。
- note: string，可选。裁剪备注。
- orientation: number，{0,90,180,270}。
- crop: [x, y, w, h]，0~1 归一化。

---

### 二、核心参数

核心参数分为两类：`idt` 与 `cc_params`。

#### 2.1 IDT（输入设备变换）

```json
"idt": {
  "name": "CCFLGeneric_Linear",
  "gamma": 1.0,
  "white": { "x": 0.34567, "y": 0.35850 },
  "primitives": {
    "r": { "x": 0.64, "y": 0.33 },
    "g": { "x": 0.30, "y": 0.60 },
    "b": { "x": 0.15, "y": 0.06 }
  }
}
```

- name: string。预设（或色彩空间/变换）的名称（必填）。
- gamma: number。IDT 的 Gamma，目前未实现，规范上固定写入 1.0（占位，便于未来扩展）。
- white: 对象。白点的 xy 色度坐标（必填）。
  - x: number ∈ (0,1)
  - y: number ∈ (0,1)
- primitives: 对象。基色原语（必填）。
  - r/g/b: 各自为对象，包含 xy 色度坐标（必填）。
    - x: number ∈ (0,1)
    - y: number ∈ (0,1)

解析策略：
- 若同时存在 `name` 与 `primitives`/`white`，以具体数值为准，name 仅作标识与回显。
- 若仅有 `name`，运行时按 name 在系统内查找变换定义；`gamma` 在当前版本视为 1.0。

#### 2.2 cc_params（调色参数）

```json
"cc_params": {
  "density_gamma": 1.0,
  "density_dmax": 2.5,
  "density_matrix": { "name": "Cineon_States_M_to_Print_Density", "values": [[...],[...],[...]] },
  "density_curve": {
    "name": "Kodak Endura Premier",
    "points": {
      "rgb": [[0,0], [1,1]],
      "r": [[...], [...]],
      "g": [[...], [...]],
      "b": [[...], [...]]
    }
  },
  "screen_glare_compensation": 0.0
}
```

- density_gamma: number。密度反演 Gamma。推荐范围 [0.1, 5.0]。
- density_dmax: number。密度 Dmax。推荐范围 [0.1, 6.0]。
- density_matrix: 对象。密度矩阵。
  - name: string。矩阵名（用于查找/回显）。
  - values: number[3][3] | null。若提供则优先生效；为空时按 name 查找。
  - 约定行优先（row-major），数值为 float。
- density_curve: 对象。密度曲线。
  - name: string。曲线名（用于查找/回显）。
  - points: 对象。曲线点集：
    - rgb: [[x,y], ...]（整体曲线）。
    - r/g/b: [[x,y], ...]（每通道曲线，可选）。
  - 约束：x 单调非降（0~1），y 建议在 0~1 范围内；至少包含一条曲线（rgb 或任一通道）。
- screen_glare_compensation: number。屏幕反光补偿量。推荐范围 [0.0, 0.2]，默认值 0.0。在线性空间中应用减法补偿。

解析优先级：
1) 提供了具体数值（matrix.values / curve.points）则优先使用。
2) 数值缺失但提供 name 时，按 name 查找系统内定义。
3) 两者皆无，使用默认值或保持为空（由应用层决定）。

---

### 三、contactsheets 的结构

当 `type = "contactsheet"` 时，顶层 `metadata` 描述原图/整体默认上下文；`crops` 为数组，每一项结构如下：

```json
{
  "metadata": {
    "id": "crop_1",
    "name": "裁剪 1",
    "note": "示例裁剪",
    "orientation": 0,
    "crop": [0.1, 0.1, 0.3, 0.3]
  },
  "idt": { ... },
  "cc_params": { ... }
}
```

继承规则（建议实现）：
- 若裁剪级缺失 `idt` 或 `cc_params`，则从顶层继承对应对象的“名与数据”。

---

### 四、示例

#### 4.1 single 示例

```json
{
  "version": 3,
  "type": "single",
  "metadata": {
    "raw_file": "Portra400鲜艳街头.tif",
    "scanner": "Nikon 5000ED",
    "light_source": "D50",
    "film_stock": "Kodak Portra 400",
    "note": "室外自然光",
    "orientation": 0,
    "crop": [0.0, 0.0, 1.0, 1.0]
  },
  "idt": {
    "name": "CCFLGeneric_Linear",
    "gamma": 1.0,
    "white": { "x": 0.34567, "y": 0.35850 },
    "primitives": {
      "r": { "x": 0.64, "y": 0.33 },
      "g": { "x": 0.30, "y": 0.60 },
      "b": { "x": 0.15, "y": 0.06 }
    }
  },
  "cc_params": {
    "density_gamma": 1.0,
    "density_dmax": 2.5,
    "density_matrix": {
      "name": "Cineon_States_M_to_Print_Density",
      "values": [[1.0197,0.0317,0.0091],[-0.0052,0.8933,0.0521],[0.0131,-0.0011,0.9712]]
    },
    "density_curve": {
      "name": "Kodak Endura Premier",
      "points": { "rgb": [[0.0,0.0],[1.0,1.0]] }
    },
    "screen_glare_compensation": 0.0
  }
}
```

#### 4.2 contactsheet 示例

```json
{
  "version": 3,
  "type": "contactsheet",
  "metadata": {
    "raw_file": "Portra400鲜艳街头.tif",
    "scanner": "Nikon 5000ED",
    "light_source": "D50",
    "film_stock": "Kodak Portra 400",
    "orientation": 0,
    "crop": [0.0, 0.0, 1.0, 1.0]
  },
  "idt": {
    "name": "CCFLGeneric_Linear",
    "gamma": 1.0,
    "white": { "x": 0.34567, "y": 0.35850 },
    "primitives": {
      "r": { "x": 0.64, "y": 0.33 },
      "g": { "x": 0.30, "y": 0.60 },
      "b": { "x": 0.15, "y": 0.06 }
    }
  },
  "cc_params": { "density_gamma": 1.0, "density_dmax": 2.5 },
  "active_crop_id": "crop_1",
  "crops": [
    {
      "metadata": { "id": "crop_1", "name": "主裁剪", "orientation": 0, "crop": [0.1,0.1,0.4,0.3] },
      "idt": {
        "name": "CCFLGeneric_Linear",
        "gamma": 1.0,
        "white": { "x": 0.34567, "y": 0.35850 },
        "primitives": {
          "r": { "x": 0.64, "y": 0.33 },
          "g": { "x": 0.30, "y": 0.60 },
          "b": { "x": 0.15, "y": 0.06 }
        }
      },
      "cc_params": {
        "density_gamma": 1.0,
        "density_dmax": 2.5,
        "density_matrix": { "name": "Cineon_States_M_to_Print_Density" },
        "density_curve": { "name": "Kodak Endura Premier", "points": { "rgb": [[0,0],[1,1]] } },
        "screen_glare_compensation": 0.0
      }
    }
  ]
}
```

---

### 五、兼容性与迁移建议

- 读取：
  - 若检测到 v3（顶层 version=3 或存在 `metadata`/`idt`/`cc_params`），按本规范解析。
  - 若为旧格式（存在 `grading_params`/`density_matrix`/`density_curve` 等旧字段），走旧解析并尽量映射到运行态的同义字段。
- 写入（过渡期）：
  - 推荐仅写 v3 字段；若需要与旧工具双向兼容，可镜像写旧字段（如 `grading_params`）一段时间。
- 语义映射（旧→新）：
  - input_transformation → idt
  - grading_params.density_gamma → cc_params.density_gamma
  - grading_params.density_dmax → cc_params.density_dmax
  - density_matrix / grading_params.density_matrix{,_file} → cc_params.density_matrix
  - density_curve / grading_params.{curve_points*,density_curve_name} → cc_params.density_curve
  - bundle/contactsheets → type = "contactsheet" + crops 数组

---

### 六、校验与约束（建议）

- orientation ∈ {0,90,180,270}
- crop: 0 ≤ x,y ≤ 1；0 ≤ w,h ≤ 1；并满足 x+w ≤ 1、y+h ≤ 1。
- density_matrix.values: 3×3 浮点矩阵（row-major）。
- 曲线点：x 单调非降，x,y 推荐在 [0,1]；每条曲线至少 2 个点。
- 若对象同时包含 "name" 与具体数值，则以数值为准，name 仅用于标识与查找。
 - idt.white 与 idt.primitives.r/g/b 中的 x,y ∈ (0,1)；建议满足 x ≥ 0, y ≥ 0, x + y ≤ 1。
 - idt.gamma 当前版本固定为 1.0（未启用）。

---

### 七、可能的遗漏与可选增强

- active_crop_id：在 contactsheet 模式下记录上次活跃裁剪（已在示例中包含）。
- locale / author：记录本地化信息或作者标识，便于协作。
- tags：在 metadata 与裁剪 metadata 中加入标签数组，支持检索与过滤。
- color_management：可在 metadata 中增加显示/工作空间信息（例如 DisplayP3、ACEScg），便于跨系统可复现性。
- provenance：记录处理链或引用资源（如参考矩阵、曲线来源）的 URI。

---

### 八、实现提示（非规范）

- 解析器应“宽进严出”：
  - 读取时尽量容错并填充默认值；
  - 写入时输出满足本规范的最小必要字段与范围约束。
- 运行时优先级：具体数值 > 名字查找 > 默认值。
- 数据流非侵入：内部运行态依然可使用现有 `ColorGradingParams`/`Preset` 模型，新增的 v3 仅在序列化/反序列化层完成映射。


