@echo off
setlocal enabledelayedexpansion

REM 管理者権限チェック
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo このスクリプトは管理者権限で実行する必要があります。
    echo 右クリックして「管理者として実行」を選択してください。
    pause
    exit /b 1
)

if exist C:\vcpkg (
    echo C:\vcpkg は既に存在します。削除してから実行してください。
    pause
    exit /b 1
)

REM スクリプトのディレクトリを取得
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM CPUスレッド数を取得
for /f "tokens=2 delims==" %%a in ('wmic cpu get NumberOfLogicalProcessors /value ^| find "="') do set THREAD_COUNT=%%a
echo 検出されたスレッド数: %THREAD_COUNT%

echo vcpkg install
git clone https://github.com/microsoft/vcpkg.git
if %errorlevel% neq 0 (
    echo git clone に失敗しました。
    pause
    exit /b 1
)

echo multithread install

REM コピー先ディレクトリの存在確認
if not exist "vcpkg\ports\openblas" (
    echo エラー: vcpkg\ports\openblas ディレクトリが見つかりません。
    pause
    exit /b 1
)

cd vcpkg
call bootstrap-vcpkg.bat -disableMetrics
if %errorlevel% neq 0 (
    echo bootstrap-vcpkg.bat の実行に失敗しました。
    pause
    exit /b 1
)

REM スレッド数を指定してOpenBLASをインストール
echo OpenBLASを%THREAD_COUNT%スレッドでビルドします...
set VCPKG_CMAKE_BUILD_PARALLEL_LEVEL=%THREAD_COUNT%
vcpkg install openblas[threads]:x64-windows
if %errorlevel% neq 0 (
    echo vcpkg install openblas に失敗しました。
    pause
    exit /b 1
)

cd ..
move vcpkg C:\vcpkg
if %errorlevel% neq 0 (
    echo vcpkg ディレクトリの移動に失敗しました。
    pause
    exit /b 1
)

echo セットアップが正常に完了しました。
pause