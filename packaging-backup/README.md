# DiVERE 本地打包工具（备用方案）

## ⚠️ 重要说明

**这是本地打包的备用方案，主要用于：**
- GitHub Actions 构建失败时的备用方案
- 本地快速测试和调试
- 离线环境下的打包需求

**推荐使用方式：GitHub Actions 自动化构建**

## 📁 目录结构

```
packaging-backup/
├── build/           # 构建临时目录
├── dist/            # 最终输出目录
├── scripts/         # 平台特定构建脚本
├── resources/       # 资源文件
└── README.md        # 本文件
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 确保 ONNX 模型文件存在
ls divere/colorConstancyModels/net_awb.onnx
```

### 2. 构建应用

#### macOS
```bash
./packaging-backup/scripts/build_macos.sh
```

#### Windows
```bash
packaging-backup\scripts\build_windows.bat
```

#### Linux
```bash
./packaging-backup/scripts/build_linux.sh
```

#### 自动检测平台
```bash
./packaging-backup/scripts/build_all.sh
```

## 📋 构建产物

构建完成后，在 `packaging-backup/dist/{platform}/` 目录下会生成：

- **可执行文件**：`DiVERE` 或 `DiVERE.exe`
- **模型文件**：`models/net_awb.onnx`
- **配置文件**：`config/` 目录
- **分发包**：`.tar.gz` 或 `.zip` 文件

## 🔧 配置选项

### Nuitka 参数

可以在脚本中修改以下参数：

```bash
# 输出目录
--output-dir="$DIST_DIR"

# 输出文件名
--output-filename=DiVERE

# 包含配置文件
--include-data-dir=config=config

# 包含模型文件
--include-data-file=divere/colorConstancyModels/net_awb.onnx=models/net_awb.onnx

# 启用 PySide6 插件
--enable-plugin=pyside6

# 独立模式
--standalone
```

## 🐛 故障排除

### 常见问题

1. **权限问题**
   ```bash
   chmod +x packaging-backup/scripts/*.sh
   ```

2. **路径问题**
   - 确保在项目根目录运行脚本
   - 检查 `PROJECT_ROOT` 变量设置

3. **依赖问题**
   - 确保所有 Python 依赖已安装
   - 检查 ONNX 模型文件是否存在

4. **Nuitka 问题**
   - 确保 Nuitka 已正确安装
   - 检查插件名称（`pyside6` 不是 `py-side6`）

### 调试建议

- 查看脚本输出的详细日志
- 检查临时构建目录 `build/`
- 验证最终输出目录 `dist/`

## 📚 与 GitHub Actions 的区别

| 特性 | 本地打包 | GitHub Actions |
|------|----------|----------------|
| **触发方式** | 手动运行 | 自动/手动触发 |
| **平台支持** | 当前系统 | 多平台并行 |
| **构建时间** | 5-10分钟 | 10-20分钟 |
| **资源占用** | 本地资源 | GitHub 资源 |
| **调试便利性** | 高 | 中等 |
| **维护成本** | 低 | 低 |

## 🎯 使用场景

### 适合使用本地打包的情况：
- ✅ 快速测试构建配置
- ✅ 调试 Nuitka 参数
- ✅ 离线环境
- ✅ GitHub Actions 构建失败

### 推荐使用 GitHub Actions 的情况：
- 🚀 正式发布版本
- 🚀 多平台构建
- 🚀 自动化部署
- 🚀 团队协作

## 📝 更新日志

- **v1.0.0**：初始版本，支持三平台本地打包
- **v1.1.0**：重构为备用方案，优化脚本结构
- **v1.2.0**：修复路径和依赖问题

## 🤝 贡献

如果需要修改本地打包脚本，请：

1. 先在 GitHub Actions 中测试
2. 确保修改不会影响自动化构建
3. 更新相关文档
4. 测试所有平台脚本

---

**注意：此备用方案将在 GitHub Actions 稳定运行后逐步减少维护。**
