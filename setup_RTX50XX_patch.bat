chcp 65001 > NUL
@echo off

echo.
echo  =================================================
echo  Beatrice Trainer 2.0.0-rc.0
echo  Unofficial Simple WebUI
echo.
echo  Patch for  RTX50Series
echo.
echo  2025.09.04
echo  =================================================
echo.

set TOOLS=%~dp0dev_tools
set USERPROFILE=%TOOLS%
set PS=PowerShell -ExecutionPolicy Bypass

set PATH=%TOOLS%\PortableGit\bin;%TOOLS%\python;%TOOLS%\python\Scripts;%PATH%
set PYTHONPATH=%TOOLS%\python;
set PY="%TOOLS%\python\python.exe"
set PIP_CACHE_DIR=%TOOLS%\pip

echo.
echo [Packages install]
echo.

cd beatrice-trainer
%PY% -m virtualenv --copies venv
call venv\Scripts\activate.bat

pip uninstall -y torch torchaudio
pip install torch==2.7.0 torchaudio==2.7.0 --upgrade --index-url https://download.pytorch.org/whl/cu128


echo.
echo All processes are complete.
echo.

pause
exit
