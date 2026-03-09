@echo off

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

echo vcpkg install
git clone https://github.com/microsoft/vcpkg.git
echo multithread install
copy portfile.cmake vcpkg\ports\openblas\portfile.cmake
cd vcpkg
call bootstrap-vcpkg.bat -disableMetrics
vcpkg.exe install openblas
move \vcpkg C:\vcpkg
pause