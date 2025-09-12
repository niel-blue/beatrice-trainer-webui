chcp 65001 > NUL
@echo off

set TOOLS=%~dp0dev_tools
set USERPROFILE=%TOOLS%

set PATH=%TOOLS%\PortableGit\bin;%TOOLS%\python;%TOOLS%\python\Scripts;%PATH%
set PYTHONPATH=%TOOLS%\python;
set PY="%TOOLS%\python\python.exe"
set PIP_CACHE_DIR=%TOOLS%\pip

echo.
echo Beatrice-Trainer Simple webui Run
echo.

cd beatrice-trainer
call venv\Scripts\activate.bat
set PYTHONWARNINGS=ignore
python webui.py

echo.
echo.
pause
exit
