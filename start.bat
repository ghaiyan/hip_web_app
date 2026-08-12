@echo off
REM 髋关节关键点检测与参数计算系统 - Windows 启动脚本
REM =====================================================
REM 用法:
REM   start.bat                              默认 8000 端口 CPU
REM   start.bat --port 8080 --gpu            GPU 8080 端口
REM   start.bat --weights .\weights\best.pth

set PORT=8000
set DEVICE=cpu
set WEIGHTS=
set MODEL=drfnet

:parse
if "%~1"=="" goto run
if "%~1"=="--port"    (set PORT=%~2 & shift & shift & goto parse)
if "%~1"=="--gpu"     (set DEVICE=cuda & shift & goto parse)
if "%~1"=="--weights" (set WEIGHTS=%~2 & shift & shift & goto parse)
if "%~1"=="--model"   (set MODEL=%~2 & shift & shift & goto parse)
shift
goto parse

:run
echo ============================================
echo   髋关节关键点检测与参数计算系统
echo ============================================
echo   端口: %PORT%
echo   设备: %DEVICE%
echo   模型: %MODEL%
if not "%WEIGHTS%"=="" echo   权重: %WEIGHTS%
echo ============================================
echo.

set VENV_PYTHON=C:\Users\mfqus\.workbuddy\binaries\python\envs\hip_web\Scripts\python.exe
if "%WEIGHTS%"=="" (
    "%VENV_PYTHON%" app.py --host 0.0.0.0 --port %PORT% --device %DEVICE% --model %MODEL%
) else (
    "%VENV_PYTHON%" app.py --host 0.0.0.0 --port %PORT% --device %DEVICE% --model %MODEL% --weights "%WEIGHTS%"
)
