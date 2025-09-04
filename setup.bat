chcp 65001 > NUL
@echo off

echo.
echo  =================================================
echo  Beatrice Trainer 2.0.0-rc.0
echo  Unofficial Simple WebUI Installer
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


if exist "%TOOLS%\python" (goto :DLGit)
echo.
echo [Python download and Setup]
echo.

%PS% Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip -OutFile python.zip
%PS% Expand-Archive -Path python.zip -DestinationPath %TOOLS%\python
del python.zip

echo python310.zip> %TOOLS%\python\python310._pth
echo .>> %TOOLS%\python\python310._pth
echo.>> %TOOLS%\python\python310._pth
echo # Uncomment to run site.main() automatically>> %TOOLS%\python\python310._pth
echo import site>> %TOOLS%\python\python310._pth

%PS% Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile %TOOLS%\python\get-pip.py
%PY% "%TOOLS%\python\get-pip.py" --no-warn-script-location
del %TOOLS%\python\get-pip.py
%PY% -m pip install virtualenv --no-warn-script-location


:DLGit
if exist "%TOOLS%\PortableGit" (goto :Gitclone)
echo.
echo [PortableGit Download]
echo.

%PS% Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/PortableGit-2.44.0-64-bit.7z.exe -OutFile %TOOLS%\PortableGit-2.44.0-64-bit.7z.exe
%TOOLS%\PortableGit-2.44.0-64-bit.7z.exe -y
del %TOOLS%\PortableGit-2.44.0-64-bit.7z.exe
rmdir /s /q Microsoft

:Gitclone
echo.
echo [Clone a repository]
echo.

ren beatrice-trainer temp

git lfs install
git config --global advice.detachedHead false
git clone https://huggingface.co/fierce-cats/beatrice-trainer beatrice-trainer

xcopy /E /Y temp\* beatrice-trainer\
rmdir /S /Q temp

echo.
echo [Packages install]
echo.

cd beatrice-trainer
%PY% -m virtualenv --copies venv
call venv\Scripts\activate.bat

pip install -e .[cu128]
pip install gradio==5.5
pip install TensorFlow

echo.
echo [Run WebUI]
echo.

python webui.py %*

echo.
echo.
pause
exit
