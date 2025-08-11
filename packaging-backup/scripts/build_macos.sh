#!/bin/bash
# macOS 打包脚本

set -e

echo "开始 macOS 打包..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$PROJECT_ROOT/build/macos"
DIST_DIR="$PROJECT_ROOT/dist/macos"

# 清理之前的构建
echo "清理之前的构建..."
rm -rf "$BUILD_DIR"
rm -rf "$DIST_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# 创建模型目录
mkdir -p "$DIST_DIR/models"

# 复制模型文件
echo "复制模型文件..."
if [ -f "$PROJECT_ROOT/divere/models/net_awb.onnx" ]; then
    cp "$PROJECT_ROOT/divere/models/net_awb.onnx" "$DIST_DIR/models/"
    echo "模型文件复制成功"
else
    echo "警告: 模型文件不存在: $PROJECT_ROOT/divere/models/net_awb.onnx"
    echo "请确保 ONNX 模型文件已正确放置"
    exit 1
fi

# 复制配置文件
echo "复制配置文件..."
cp -r "$PROJECT_ROOT/config" "$DIST_DIR/"

# 检查 Nuitka 是否安装
if ! command -v python -m nuitka &> /dev/null; then
    echo "安装 Nuitka..."
    pip install nuitka
fi

# 使用 Nuitka 打包
echo "使用 Nuitka 打包应用..."
cd "$PROJECT_ROOT"

python -m nuitka \
    --standalone \
    --include-data-dir=config=config \
    --include-data-file=divere/models/net_awb.onnx=models/net_awb.onnx \
    --output-dir="$DIST_DIR" \
    --output-filename=DiVERE \
    --assume-yes-for-downloads \
    --enable-plugin=pyside6 \
    divere/__main__.py

# 创建分发包
echo "创建分发包..."
cd "$DIST_DIR"
tar -czf "DiVERE-1.0.0-macOS.tar.gz" __main__.dist/DiVERE models config

echo "macOS 打包完成！"
echo "分发包位置: $DIST_DIR/DiVERE-1.0.0-macOS.tar.gz"
echo "可执行文件位置: $DIST_DIR/DiVERE"
