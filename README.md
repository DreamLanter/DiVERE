# DiVERE - 胶片校色工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.5+-green.svg)](https://pypi.org/project/PyQt6/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于ACEScg Linear工作流的胶片数字化后期处理工具，为胶片摄影师提供校色解决方案。

## 🌟 功能特性

- 🎨 **色彩空间管理**：支持ACEScg Linear工作流和多种色彩空间转换
- ⚡ **实时预览**：基于LUT的实时图像预览
- 📸 **代理处理**：自动生成低分辨率代理文件用于预览
- 🎛️ **调色功能**：密度反相、校正矩阵、RGB增益、密度曲线调整
- 🤖 **自动校色**：集成深度学习自动白平衡算法（可选）
- 📁 **目录记忆**：记住上次打开和保存文件的目录
- 🔧 **模块化设计**：清晰的代码结构，便于维护和扩展
- 🖥️ **跨平台**：支持Windows、macOS、Linux系统

## 📦 安装部署

### 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **内存**: 建议 4GB 或更多
- **存储**: 至少 1GB 可用空间
- **Git**: 支持子模块的Git版本

### 🚀 快速开始

```bash
# 1. 克隆项目（包含所有子模块）
git clone --recursive https://github.com/V7CN/DiVERE.git
cd DiVERE

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行应用
python -m divere
```

### 快速安装

#### 方法一：使用pip

```bash
# 克隆项目（包含子模块）
git clone --recursive https://github.com/V7CN/DiVERE.git
cd DiVERE

# 如果克隆时没有包含子模块，请运行：
# git submodule init
# git submodule update

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
# 克隆项目（包含子模块）
git clone --recursive https://github.com/V7CN/DiVERE.git
cd DiVERE

# 如果克隆时没有包含子模块，请运行：
# git submodule init
# git submodule update

# 创建conda环境
conda create -n divere python=3.9
conda activate divere

# 安装依赖
pip install -r requirements.txt

# 运行应用
python -m divere
```

### 依赖包说明

#### 核心依赖
```
PyQt6>=6.5.0          # GUI框架
numpy>=1.24.0         # 数值计算
opencv-python>=4.8.0  # 图像处理
pillow>=10.0.0        # 图像I/O
scipy>=1.11.0         # 科学计算
imageio>=2.31.0       # 图像格式支持
rawpy>=0.18.0         # RAW文件支持
colour-science>=0.4.2 # 色彩科学计算
```

#### AI功能依赖（自动校色）
```
torch>=2.0.0          # 深度学习框架
torchvision>=0.15.0   # 计算机视觉库
scikit-learn>=1.3.0   # 机器学习工具
matplotlib>=3.7.0     # 绘图库
tqdm>=4.65.0          # 进度条
```

### 子模块说明

本项目使用Git子模块管理第三方库：

- **Deep_White_Balance**: 深度学习自动白平衡算法
  - 来源: https://github.com/mahmoudnafifi/Deep_White_Balance
  - 用途: 提供AI自动校色功能
  - 位置: `divere/colorConstancyModels/Deep_White_Balance/`

如果克隆时没有包含子模块，请运行：
```bash
git submodule init
git submodule update
```

## 🚀 使用指南

### 基本操作流程

1. **打开图像**
   - 点击"文件" → "打开图像"或使用快捷键 `Ctrl+O`
   - 支持格式：TIFF、PNG、JPEG、BMP、WebP、RAW

2. **设置输入色彩空间**
   - 在"输入色彩管理"标签页选择正确的输入色彩空间
   - 常用选项：sRGB、Adobe RGB、ProPhoto RGB、Film_KodakRGB_Linear

3. **应用调色参数**
   - **密度与矩阵**：调整密度反相参数和校正矩阵
   - **RGB曝光**：微调红绿蓝通道的曝光
   - **密度曲线**：使用曲线工具调整对比度

4. **导出结果**
   - 点击"文件" → "保存图像"或使用快捷键 `Ctrl+S`
   - 选择输出格式和色彩空间

### 高级功能

#### 自动校色
- 点击"RGB曝光"标签页中的"自动校色"按钮
- 基于深度学习的自动白平衡算法
- 支持多次迭代优化

#### LUT导出
- **3D LUT**：导出包含所有调色功能的3D查找表
- **输入色彩管理LUT**：导出输入色彩空间转换LUT
- **密度曲线1D LUT**：导出密度曲线1D查找表

## 🔧 技术架构

### 整体Pipeline

```
输入图像 → 色彩空间转换 → 密度反相 → 校正矩阵 → RGB增益 → 密度曲线 → 输出转换 → 最终图像
    ↓           ↓           ↓         ↓         ↓         ↓         ↓
  图像管理    色彩管理     调色引擎   调色引擎   调色引擎   调色引擎   色彩管理
```

### 核心模块

#### 1. 图像管理模块 (ImageManager)
- **功能**：图像加载、代理生成、缓存管理
- **特性**：支持多种格式、代理生成、内存管理

#### 2. 色彩空间管理模块 (ColorSpaceManager)
- **功能**：色彩空间转换、ICC配置文件处理
- **特性**：基于colour-science、ACEScg工作流

#### 3. 调色引擎模块 (TheEnlarger)
- **功能**：密度反相、校正矩阵、RGB增益、密度曲线
- **特性**：线性处理、LUT生成

#### 4. LUT处理器 (LUTProcessor)
- **功能**：3D/1D LUT生成、缓存管理
- **特性**：缓存机制、文件格式支持

### 色彩处理Pipeline详解

#### 1. 密度反相 (Density Inversion)
```python
# 将线性值转换为密度值
density = -log10(linear_value)

# 应用密度反相参数
adjusted_density = density * gamma + dmax
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
│   └── colorConstancyModels/ # AI自动校色
│       ├── deep_wb_wrapper.py # Deep White Balance包装器
│       ├── utils/            # 工具函数
│       └── Deep_White_Balance/ # Git子模块
├── config/                   # 配置文件
│   ├── colorspace/          # 色彩空间配置
│   ├── curves/              # 预设曲线
│   └── matrices/            # 校正矩阵
├── .gitmodules              # Git子模块配置
├── requirements.txt         # Python依赖
├── pyproject.toml          # 项目配置
└── README.md               # 项目文档
```

## 🤝 致谢

### 深度学习自动校色

本项目通过Git子模块集成了以下优秀的开源项目：

#### Deep White Balance
- **论文**: "Deep White-Balance Editing" (CVPR 2020)
- **作者**: Mahmoud Afifi, Konstantinos G. Derpanis, Björn Ommer, Michael S. Brown
- **GitHub**: https://github.com/mahmoudnafifi/Deep_White_Balance
- **许可证**: MIT License
- **集成方式**: Git子模块

Deep White Balance提供了基于深度学习的自动白平衡算法，我们将其作为子模块集成到DiVERE中，实现了自动校色功能。

### 开源库

- **PyQt6**: GUI框架
- **NumPy**: 数值计算
- **OpenCV**: 图像处理
- **colour-science**: 色彩科学计算
- **PyTorch**: 深度学习框架

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