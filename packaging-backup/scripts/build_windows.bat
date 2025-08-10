@echo off
REM Windows 打包脚本

echo 开始 Windows 打包...

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set BUILD_DIR=%PROJECT_ROOT%\build\windows
set DIST_DIR=%PROJECT_ROOT%\dist\windows

REM 清理之前的构建
echo 清理之前的构建...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%BUILD_DIR%"
mkdir "%DIST_DIR%"

REM 创建模型目录
mkdir "%DIST_DIR%\models"

REM 复制模型文件
echo 复制模型文件...
if exist "%PROJECT_ROOT%\divere\colorConstancyModels\net_awb.onnx" (
    copy "%PROJECT_ROOT%\divere\colorConstancyModels\net_awb.onnx" "%DIST_DIR%\models\"
    echo 模型文件复制成功
) else (
    echo 警告: 模型文件不存在: %PROJECT_ROOT%\divere\colorConstancyModels\net_awb.onnx
    echo 请确保 ONNX 模型文件已正确放置
    exit /b 1
)

REM 复制配置文件
echo 复制配置文件...
xcopy "%PROJECT_ROOT%\config" "%DIST_DIR%\config\" /E /I /Y

REM 检查 Nuitka 是否安装
python -c "import nuitka" 2>nul
if errorlevel 1 (
    echo 安装 Nuitka...
    pip install nuitka
)

REM 使用 Nuitka 打包
echo 使用 Nuitka 打包应用...
cd /d "%PROJECT_ROOT%"

python -m nuitka ^
    --standalone ^
    --include-data-dir=config=config ^

    --include-data-file=divere/colorConstancyModels/net_awb.onnx=models/net_awb.onnx ^
    --output-dir="%DIST_DIR%" ^
    --output-filename=DiVERE.exe ^
    --assume-yes-for-downloads ^
    --enable-plugin=pyside6 ^
    --windows-icon-from-ico=config/app_icon.ico ^
    --windows-company-name="DiVERE" ^
    --windows-product-name="DiVERE" ^
    --windows-file-version=1.0.0.0 ^
    --windows-product-version=1.0.0.0 ^
    --windows-file-description="专业胶片校色工具" ^
    --windows-uac-admin ^
    divere\__main__.py

REM 创建分发包
echo 创建分发包...
cd /d "%DIST_DIR%"
powershell -Command "Compress-Archive -Path 'DiVERE.exe', 'models', 'config' -DestinationPath 'DiVERE-1.0.0-Windows.zip' -Force"

echo Windows 打包完成！
echo 分发包位置: %DIST_DIR%\DiVERE-1.0.0-Windows.zip
echo 可执行文件位置: %DIST_DIR%\DiVERE.exe
