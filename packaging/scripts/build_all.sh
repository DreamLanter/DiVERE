#!/bin/bash
# 通用打包脚本

set -e

echo "DiVERE 多平台打包工具"
echo "======================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检测当前平台
PLATFORM=$(uname -s)
echo "检测到平台: $PLATFORM"

# 检查依赖
echo "检查依赖..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "错误: 未找到 pip"
    exit 1
fi

# 安装 Nuitka
echo "检查 Nuitka..."
if ! python -c "import nuitka" 2>/dev/null; then
    echo "安装 Nuitka..."
    pip install nuitka
fi

# 根据平台选择打包脚本
case $PLATFORM in
    Darwin)
        echo "在 macOS 上构建..."
        chmod +x "$SCRIPT_DIR/build_macos.sh"
        "$SCRIPT_DIR/build_macos.sh"
        ;;
    Linux)
        echo "在 Linux 上构建..."
        chmod +x "$SCRIPT_DIR/build_linux.sh"
        "$SCRIPT_DIR/build_linux.sh"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "在 Windows 上构建..."
        if [ -f "$SCRIPT_DIR/build_windows.bat" ]; then
            "$SCRIPT_DIR/build_windows.bat"
        else
            echo "错误: Windows 打包脚本不存在"
            exit 1
        fi
        ;;
    *)
        echo "不支持的平台: $PLATFORM"
        echo "支持的平台: macOS, Linux, Windows"
        exit 1
        ;;
esac

echo ""
echo "打包完成！"
echo "分发包位置: $PROJECT_ROOT/dist/"
echo ""
echo "各平台分发包:"
if [ -d "$PROJECT_ROOT/dist/macos" ]; then
    echo "  macOS: $PROJECT_ROOT/dist/macos/DiVERE-1.0.0-macOS.tar.gz"
fi
if [ -d "$PROJECT_ROOT/dist/windows" ]; then
    echo "  Windows: $PROJECT_ROOT/dist/windows/DiVERE-1.0.0-Windows.zip"
fi
if [ -d "$PROJECT_ROOT/dist/linux" ]; then
    echo "  Linux: $PROJECT_ROOT/dist/linux/DiVERE-1.0.0-Linux.tar.gz"
fi
