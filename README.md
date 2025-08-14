# DiVERE - 胶片校色工具

[![Python](https://img.shields.io/badge/Python-3.9~3.11-blue.svg)](https://www.python.org/downloads/) ![Version](https://img.shields.io/badge/Version-v0.1.10-orange)
[![PySide6](https://img.shields.io/badge/PySide6-6.5+-green.svg)](https://pypi.org/project/PySide6/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于ACEScg Linear工作流的胶片数字化后期处理工具，为胶片摄影师提供校色解决方案。

## 🌟 功能特性

- 扫描件的简单色彩管理：从一个通过光谱算的胶片基色（我叫它Kodak RGB），转换到工作空间ACEScg Linear。源色彩空间可以用json来任意定义。
- 基于密度的工作流，包括反相、密度矩阵、rgb曝光、曲线等等。
- 用了Status M to Print Density的密度矩阵。这个深入了解胶片数子化的都懂。实测epson平板扫描扫出来基本就是Status M（但翻拍可能不是，取决于光源和相机）。矩阵可交互式调节，就像做dye-transfer一样。并且可以保存为json。
- 用了一个机器学习模型做初步的校色。要多点几次（因为色彩正常之后cnn才能正常识别图片语义）效果比基于统计的方法强多了。
- 一个横纵都是密度的曲线工具，可以非常自然地模拟相纸的暗部偏色特性。我内置了一个endura相纸曲线。曲线可保存为json
- 全精度的图片输出（导出：全精度+分块并行，禁用低精度LUT）。
- 各种精度、各种pipeline的3D LUT生成功能。以及，因为密度曲线非常好用，我单独开了一个密度曲线的1D LUT导出功能

## 📦 安装部署

### 系统要求

- Python 3.9–3.11（推荐 3.11）
- 操作系统：macOS 12+（Intel/Apple Silicon）、Windows 10/11、Ubuntu 20.04+
- 显卡：非必须。GPU 加速（可选）：
  - macOS Metal（推荐）：通过 PyObjC 访问 Metal（Apple Silicon/Intel）
  - OpenCL（可选）：跨平台（Windows/macOS/Linux）
  - CUDA（可选）：NVIDIA 显卡
- 包管理：pip 或 conda

### 🚀 快速开始

#### 方法零：手动下载
- .首先点Code -> Download ZIP 下载本项目源码（400多MB，大多是校色示例图片）
- .安装python
- .安装依赖、运行程序：
```bash
# 安装依赖
pip install -r requirements.txt

# 如需 macOS Metal 加速（可选）
pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders

# 如需 OpenCL（可选）
# pip install pyopencl

# 运行应用
python -m divere
```

#### 方法一：使用pip

```bash
# 克隆项目
git clone https://github.com/V7CN/DiVERE.git
cd DiVERE

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行应用
python -m divere
```

#### 方法二：使用conda

```bash
# 克隆项目
git clone https://github.com/V7CN/DiVERE.git
cd DiVERE

# 创建conda环境（推荐 Python 3.11）
conda create -n divere python=3.11 -y
conda activate divere

# 安装依赖
pip install -r requirements.txt

# 运行应用
python -m divere
```

### 依赖包说明

#### 必需依赖
```
PySide6>=6.5.0          # GUI框架
numpy>=1.24.0           # 数值计算
opencv-python>=4.8.0    # 图像处理
pillow>=10.0.0          # 图像I/O
scipy>=1.11.0           # 科学计算
 imageio>=2.31.0         # 图像格式支持
colour-science>=0.4.2   # 色彩科学计算
scikit-learn>=1.3.0     # 算法/工具（KMeans等）
onnxruntime>=1.15.0     # ONNX 推理（自动校色）
```

- 可选（GPU 加速）
```
# macOS Metal
pyobjc-framework-Metal
pyobjc-framework-MetalPerformanceShaders

# OpenCL（跨平台）
pyopencl

# CUDA（NVIDIA，可选）
cupy-cuda11x  # 按你的CUDA版本选择
```

- macOS Apple Silicon（arm64）：直接使用 `pip install onnxruntime`，官方已原生支持 arm64，不需要 `onnxruntime-silicon`。
- 可用以下命令简单验证环境：
```bash
python -c "import platform, onnxruntime as ort; print(platform.machine(), ort.__version__)"
```

## 🚀 使用指南

后续将补充使用视频与文档。

## 🔧 技术架构

### 整体Pipeline

```
输入图像 → 色彩空间转换 → 密度反相 → 校正矩阵 → RGB增益 → 密度曲线 → 转线性 → 输出转换 → 最终图像
    ↓           ↓           ↓         ↓         ↓         ↓         ↓
  图像管理    色彩管理     调色引擎   调色引擎   调色引擎   调色引擎   色彩管理
```

### 核心模块

#### 1. 图像管理模块 (ImageManager)
- 功能：图像加载、代理生成、缓存管理
- 特性：支持多种格式、代理生成、内存管理

#### 2. 色彩空间管理模块 (ColorSpaceManager)
- 功能：色彩空间转换、ICC配置文件处理
- 特性：基于colour-science、ACEScg工作流

#### 3. 调色引擎模块 (TheEnlarger)
- 功能：密度反相、校正矩阵、RGB增益、密度曲线
- 特性：线性处理、LUT生成

#### 4. LUT处理器 (LUTProcessor)
- 功能：3D/1D LUT生成、缓存管理
- 特性：缓存机制、文件格式支持

### 色彩处理Pipeline详解

#### 1. 密度反相 (Density Inversion)
```python
# 公式路径（全精度导出使用公式）
safe = clip(linear, 1e-10, 1)
density = -log10(safe)
adjusted = pivot + (density - pivot) * gamma - dmax
linear_out = pow(10.0, adjusted)
```

#### 2. 校正矩阵 (Correction Matrix)
```python
# 应用3x3校正矩阵
corrected_rgb = matrix @ original_rgb
```

#### 3. RGB增益 (RGB Gains)
```python
# 在密度空间应用增益
adjusted_density = density - gain
```

#### 4. 密度曲线 (Density Curves)
```python
# 使用单调三次插值生成曲线
curve_output = monotonic_cubic_interpolate(input, curve_points)
```

### 预览与导出精度策略

- 预览：以交互速度优先，密度反相等步骤默认使用 1D LUT（32K，缓存），并支持多线程/分块。
- 导出：强制全精度公式运算 + 分块并行（零重叠、零精度损失），禁用低精度 LUT。
- 分块：默认在超大图（>16MP）时自动启用；可在代码中调整 tile 大小与并行度。

### 统一的预览配置（PreviewConfig）

通过 `PreviewConfig` 统一管理预览/代理尺寸与质量：

```python
from divere.core.data_types import PreviewConfig
from divere.core.the_enlarger import TheEnlarger

cfg = PreviewConfig(
    preview_max_size=2048,
    proxy_max_size=2000,
)
enlarger = TheEnlarger(preview_config=cfg)
```

### GPU 加速

- 优先级：Metal > OpenCL > CUDA > CPU（自动选择，失败自动回退）。
- macOS（Metal）建议安装：
```bash
pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
```
导出默认仍走 CPU/Metal 全精度公式路径，且开启分块以降低内存峰值。

## 📁 项目结构

```
DiVERE/
├── divere/                    # 主程序包
│   ├── core/                 # 核心模块
│   │   ├── image_manager.py  # 图像管理
│   │   ├── color_space.py    # 色彩空间管理
│   │   ├── the_enlarger.py   # 调色引擎
│   │   ├── lut_processor.py  # LUT处理
│   │   └── data_types.py     # 数据类型定义
│   ├── ui/                   # 用户界面
│   │   ├── main_window.py    # 主窗口
│   │   ├── preview_widget.py # 预览组件
│   │   ├── parameter_panel.py # 参数面板
│   │   └── curve_editor_widget.py # 曲线编辑器
│   ├── utils/                # 工具函数
│   │   ├── config_manager.py # 配置管理
│   │   └── lut_generator/    # LUT生成器
│   └── models/               # AI自动校色（ONNX）
│       ├── deep_wb_wrapper.py
│       ├── utils/
│       └── net_awb.onnx
├── config/                   # 配置文件
│   ├── colorspace/          # 色彩空间配置
│   ├── curves/              # 预设曲线
│   └── matrices/            # 校正矩阵
├── requirements.txt         # Python依赖
├── pyproject.toml           # 项目配置
└── README.md                # 项目文档
```

## 🤝 致谢

### 深度学习自动校色

本项目的学习型自动校色基于以下优秀的开源研究成果：

#### Deep White Balance
- 论文: "Deep White-Balance Editing" (CVPR 2020)
- 作者: Mahmoud Afifi, Konstantinos G. Derpanis, Björn Ommer, Michael S. Brown
- GitHub: https://github.com/mahmoudnafifi/Deep_White_Balance
- 许可证: MIT License
- 说明: 模型来源于上述研究，已转换为 ONNX 并随项目分发使用（`divere/models/net_awb.onnx`）。

### 开源库

- PySide6: GUI框架
- NumPy: 数值计算
- OpenCV: 图像处理
- colour-science: 色彩科学计算
- ONNX Runtime: 模型推理

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

**V7** - vanadis@yeah.net

## 🐛 问题反馈

如果您发现任何问题或有功能建议，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/V7CN/DiVERE/issues)
- 发送邮件至：vanadis@yeah.net

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 贡献指南

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📈 开发计划

- [ ] 支持更多图像格式
- [ ] 添加更多预设曲线
- [ ] 优化性能
- [ ] 支持批量处理
- [ ] 添加更多AI算法

---

**DiVERE** - 胶片校色工具 