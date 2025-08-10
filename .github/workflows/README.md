# GitHub Actions 自动化构建

## 概述

本项目使用 GitHub Actions 自动构建 DiVERE 应用程序，支持 Windows、macOS 和 Linux 三个平台。

## 工作流文件

- `build.yml` - 主要的构建工作流

## 触发条件

### 自动触发
- **标签推送**：当推送 `v*` 格式的标签时（如 `v1.0.0`）
- **Pull Request**：当向 `main` 或 `develop` 分支提交 PR 时

### 手动触发
- 在 GitHub 仓库页面的 "Actions" 标签页中手动运行

## 构建矩阵

| 平台 | 运行环境 | Python 版本 | 输出格式 |
|------|----------|-------------|----------|
| Windows | `windows-latest` | 3.11 | `.exe` + `.zip` |
| macOS | `macos-latest` | 3.11 | 可执行文件 + `.tar.gz` |
| Linux | `ubuntu-latest` | 3.11 | 可执行文件 + `.tar.gz` |

## 构建产物

每个平台都会生成以下文件：
- 可执行文件（`DiVERE` 或 `DiVERE.exe`）
- `models/` 目录（包含 ONNX 模型）
- `config/` 目录（包含配置文件）
- 压缩包（`.zip` 或 `.tar.gz`）

## 使用方法

### 1. 本地测试构建

在推送代码之前，建议先在本地测试：

```bash
# 安装依赖
pip install -r requirements.txt

# 测试运行
python -m divere
```

### 2. 触发自动构建

#### 方法 1：推送标签（推荐用于发布）
```bash
git tag v1.0.0
git push origin v1.0.0
```

#### 方法 2：创建 Pull Request
- Fork 仓库
- 创建功能分支
- 提交更改
- 创建 PR 到 `main` 分支

#### 方法 3：手动触发
- 在 GitHub 仓库页面点击 "Actions"
- 选择 "Build DiVERE Application" 工作流
- 点击 "Run workflow"

### 3. 查看构建结果

- 构建完成后，可以在 "Actions" 页面查看详细日志
- 构建产物会自动上传为 Artifacts
- 可以下载对应平台的压缩包进行测试

## 配置说明

### Nuitka 参数

```yaml
script: divere/__main__.py                    # 入口脚本
output-dir: dist                              # 输出目录
output-filename: DiVERE                       # 输出文件名
include-data-dir: config=config              # 包含配置文件
include-data-file: divere/colorConstancyModels/net_awb.onnx=models/net_awb.onnx  # 包含模型文件
enable-plugin: pyside6                        # 启用 PySide6 插件
standalone: true                              # 独立模式
```

### 平台特定配置

- **Windows**：`windows-disable-console: true` - 禁用控制台窗口
- **macOS**：`macos-create-app-bundle: false` - 不创建 .app 包
- **Linux**：默认配置

## 故障排除

### 常见问题

1. **构建失败**
   - 检查 Python 依赖是否正确安装
   - 查看构建日志中的错误信息
   - 确保所有必需文件都存在

2. **模型文件找不到**
   - 检查 `net_awb.onnx` 文件路径
   - 确保 `include-data-file` 参数正确

3. **PySide6 插件问题**
   - 确保 `requirements.txt` 中包含 `PySide6`
   - 检查 Qt 依赖是否正确安装

### 调试建议

- 先在本地环境测试构建
- 检查 GitHub Actions 的详细日志
- 对比成功和失败的构建差异

## 发布流程

### 当前状态
自动发布功能已注释，仅用于测试。

### 启用自动发布
要启用自动发布到 GitHub Releases，需要：

1. 取消注释 `build.yml` 中的发布相关步骤
2. 确保仓库有适当的权限设置
3. 推送版本标签触发构建

### 手动发布
1. 下载构建产物
2. 在 GitHub 创建新的 Release
3. 上传对应平台的压缩包
4. 编写发布说明

## 注意事项

1. **构建时间**：完整构建可能需要 10-20 分钟
2. **存储空间**：构建产物会占用 GitHub Actions 的存储配额
3. **依赖缓存**：pip 依赖会被缓存以加速后续构建
4. **安全考虑**：确保敏感信息不会在构建日志中泄露

## 更新日志

- **v1.0.0**：初始版本，支持三平台构建
- 使用 Nuitka-Action@v1.3 进行自动化构建
- 包含 ONNX 模型和配置文件
- 支持手动和自动触发
