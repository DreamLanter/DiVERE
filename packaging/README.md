# DiVERE 多平台打包指南

## 概述

本目录包含 DiVERE 应用的多平台打包工具和脚本，支持 macOS、Windows 和 Linux 平台。

## 目录结构

```
packaging/
├── build/                    # 构建临时文件
│   ├── macos/
│   ├── windows/
│   └── linux/
├── dist/                     # 最终分发文件
│   ├── macos/
│   ├── windows/
│   └── linux/
├── scripts/                  # 打包脚本
│   ├── build_macos.sh       # macOS 打包脚本
│   ├── build_windows.bat    # Windows 打包脚本
│   ├── build_linux.sh       # Linux 打包脚本
│   └── build_all.sh         # 通用打包脚本
├── resources/                # 资源文件
└── nuitka_config/           # Nuitka 配置文件
```

## 快速开始

### 自动打包（推荐）

在项目根目录运行：

```bash
./packaging/scripts/build_all.sh
```

脚本会自动检测当前平台并使用相应的打包脚本。

### 手动打包

#### macOS

```bash
./packaging/scripts/build_macos.sh
```

#### Windows

```cmd
packaging\scripts\build_windows.bat
```

#### Linux

```bash
./packaging/scripts/build_linux.sh
```

## 依赖要求

### 通用依赖

- Python 3.8+
- pip
- Nuitka（会自动安装）

### 平台特定依赖

#### macOS
- Xcode Command Line Tools
- 开发者证书（用于代码签名）

#### Windows
- Visual Studio Build Tools
- Windows SDK

#### Linux
- GCC
- libc6-dev
- 可选：appimagetool（用于创建 AppImage）

## 打包配置

### 应用信息

- 应用名称：DiVERE
- 版本：1.0.0
- 描述：专业胶片校色工具
- 类别：图形处理

### 包含的文件

- 主程序：`divere/__main__.py`
- 配置文件：`config/` 目录
- 模型文件：`divere/colorConstancyModels/net_awb.onnx`
- 核心代码：`divere/` 目录

### 平台特定配置

#### macOS
- 创建 `.app` 应用包
- 支持代码签名和公证
- 包含应用图标

#### Windows
- 生成 `.exe` 可执行文件
- 包含版本信息和图标
- 支持 UAC 管理员权限

#### Linux
- 生成 ELF 二进制文件
- 创建桌面文件
- 支持 AppImage 格式

## 输出文件

### macOS
- `DiVERE.app` - 应用包
- `DiVERE-1.0.0-macOS.tar.gz` - 分发包

### Windows
- `DiVERE.exe` - 可执行文件
- `DiVERE-1.0.0-Windows.zip` - 分发包

### Linux
- `DiVERE` - 可执行文件
- `DiVERE-1.0.0-Linux.tar.gz` - 分发包
- `DiVERE-1.0.0-Linux.AppImage` - AppImage（可选）

## 用户配置目录

打包后的应用会在以下位置创建用户配置目录：

### macOS
```
~/Library/Application Support/DiVERE/
```

### Windows
```
%LOCALAPPDATA%\DiVERE\
```

### Linux
```
~/.config/DiVERE/
```

用户可以在这些目录中：
- 添加自定义色彩空间配置
- 添加自定义曲线配置
- 添加自定义矩阵配置
- 修改应用设置

## 故障排除

### 常见问题

1. **Nuitka 安装失败**
   ```bash
   pip install --upgrade pip
   pip install nuitka
   ```

2. **权限错误**
   ```bash
   chmod +x packaging/scripts/*.sh
   ```

3. **依赖缺失**
   ```bash
   pip install -r requirements.txt
   ```

4. **打包失败**
   - 检查 Python 版本（需要 3.8+）
   - 确保所有依赖已安装
   - 检查磁盘空间

### 调试模式

在打包脚本中添加 `--debug` 参数以启用详细输出：

```bash
python -m nuitka --debug --standalone divere/__main__.py
```

## 自定义配置

### 修改应用信息

编辑相应的打包脚本中的以下参数：
- `--macos-app-name`
- `--windows-product-name`
- `--linux-app-name`

### 添加图标

将图标文件放在 `config/` 目录中：
- macOS: `app_icon.icns`
- Windows: `app_icon.ico`
- Linux: `app_icon.png`

### 修改版本号

更新所有打包脚本中的版本号：
- `--macos-app-version`
- `--windows-file-version`
- `--linux-app-version`

## 发布检查清单

在发布前请检查：

- [ ] 所有平台的分发包已创建
- [ ] 应用图标已正确包含
- [ ] 版本信息已更新
- [ ] 用户配置目录功能正常
- [ ] 应用能正常启动和运行
- [ ] 所有功能正常工作
- [ ] 文档已更新

## 支持

如有问题，请查看：
1. 项目 README.md
2. 用户配置结构文档
3. 应用内帮助菜单
