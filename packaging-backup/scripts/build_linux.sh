#!/bin/bash
# Linux 打包脚本

set -e

echo "开始 Linux 打包..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$PROJECT_ROOT/build/linux"
DIST_DIR="$PROJECT_ROOT/dist/linux"

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
    --linux-onefile-icon=config/app_icon.png \
    --linux-app-name="DiVERE" \
    --linux-app-version="1.0.0" \
    --linux-app-category="Graphics" \
    --linux-app-comments="专业胶片校色工具" \
    divere/__main__.py

# 设置可执行权限
echo "设置可执行权限..."
chmod +x "$DIST_DIR/DiVERE"

# 创建桌面文件
echo "创建桌面文件..."
cat > "$DIST_DIR/DiVERE.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DiVERE
Comment=专业胶片校色工具
Exec=DiVERE
Icon=app_icon
Terminal=false
Categories=Graphics;
EOF

# 复制图标
if [ -f "$PROJECT_ROOT/config/app_icon.png" ]; then
    cp "$PROJECT_ROOT/config/app_icon.png" "$DIST_DIR/"
fi

# 创建分发包
echo "创建分发包..."
cd "$DIST_DIR"
tar -czf "DiVERE-1.0.0-Linux.tar.gz" DiVERE models config DiVERE.desktop app_icon.png

# 创建 AppImage（可选）
if command -v appimagetool &> /dev/null; then
    echo "创建 AppImage..."
    mkdir -p DiVERE.AppDir/{usr/bin,usr/share/applications,usr/share/icons/hicolor/256x256/apps}
    cp DiVERE DiVERE.AppDir/usr/bin/
    cp DiVERE.desktop DiVERE.AppDir/usr/share/applications/
    cp app_icon.png DiVERE.AppDir/usr/share/icons/hicolor/256x256/apps/
    cp -r models config DiVERE.AppDir/
    
    cat > DiVERE.AppDir/AppRun << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
exec "\$(dirname "\$0")/usr/bin/DiVERE" "\$@"
EOF
    chmod +x DiVERE.AppDir/AppRun
    
    appimagetool DiVERE.AppDir DiVERE-1.0.0-Linux.AppImage
    echo "AppImage 创建完成: DiVERE-1.0.0-Linux.AppImage"
fi

echo "Linux 打包完成！"
echo "分发包位置: $DIST_DIR/DiVERE-1.0.0-Linux.tar.gz"
echo "可执行文件位置: $DIST_DIR/DiVERE"
