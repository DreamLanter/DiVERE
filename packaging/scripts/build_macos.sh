#!/bin/bash
# macOS 打包脚本

set -e

echo "开始 macOS 打包..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
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
cp "$PROJECT_ROOT/divere/colorConstancyModels/net_awb.onnx" "$DIST_DIR/models/"

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
    --macos-create-app-bundle \
    --include-data-dir=config:config \
    --include-data-dir=divere:divere \
    --include-data-file=divere/colorConstancyModels/net_awb.onnx:divere/colorConstancyModels/net_awb.onnx \
    --output-dir="$DIST_DIR" \
    --output-filename=DiVERE \
    --assume-yes-for-downloads \
    --enable-plugin=py-side6 \
    --macos-app-icon=config/app_icon.icns \
    --macos-app-name="DiVERE" \
    --macos-app-version="1.0.0" \
    --macos-app-identifier="com.divere.app" \
    --macos-sign-identity="Developer ID Application" \
    --macos-sign-notarization \
    --macos-sign-entitlements=packaging/macos/entitlements.plist \
    divere/__main__.py

# 创建应用包结构
echo "创建应用包结构..."
APP_BUNDLE="$DIST_DIR/DiVERE.app"
if [ -d "$APP_BUNDLE" ]; then
    # 创建 Contents/Resources 目录
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # 复制模型文件到应用包内
    cp -r "$DIST_DIR/models" "$APP_BUNDLE/Contents/Resources/"
    
    # 复制配置文件到应用包内
    cp -r "$DIST_DIR/config" "$APP_BUNDLE/Contents/Resources/"
    
    echo "应用包创建完成: $APP_BUNDLE"
else
    echo "警告: 应用包未创建成功"
fi

# 创建分发包
echo "创建分发包..."
cd "$DIST_DIR"
tar -czf "DiVERE-1.0.0-macOS.tar.gz" DiVERE.app models config

echo "macOS 打包完成！"
echo "分发包位置: $DIST_DIR/DiVERE-1.0.0-macOS.tar.gz"
echo "应用包位置: $APP_BUNDLE"
