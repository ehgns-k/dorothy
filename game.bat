@echo off
REM ===========================================================================
REM  game.bat - launch the chess-robot kiosk and keep it running.
REM
REM  Relaunches kiosk.py automatically if it ever exits unexpectedly (crash,
REM  window closed, power glitch). Stops the loop only when an operator quits
REM  on purpose with Ctrl+Shift+Q (kiosk.py writes kiosk.stop on that path).
REM
REM  CONDA_ENV / CONDA_ROOT below are auto-detected for common installs; set
REM  CONDA_ROOT by hand if conda lives somewhere unusual on the kiosk.
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "CONDA_ENV=cnsci"
REM set "CONDA_ROOT=C:\Path\To\miniconda3"    REM <- uncomment to force a path

if not defined CONDA_ROOT if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" set "CONDA_ROOT=C:\ProgramData\Anaconda3"

set "LOG=%~dp0kiosk.log"
set "STOP=%~dp0kiosk.stop"

if not defined CONDA_ROOT (
  echo [game.bat] %date% %time% Could not find conda; edit CONDA_ROOT in game.bat.>> "%LOG%"
  echo Could not find a conda install. Edit CONDA_ROOT at the top of game.bat.
  timeout /t 15 /nobreak >nul
  exit /b 1
)

REM Clear any stale stop flag from a previous session.
if exist "%STOP%" del "%STOP%"

call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 (
  echo [game.bat] %date% %time% Failed to activate conda env "%CONDA_ENV%".>> "%LOG%"
  echo Failed to activate conda env "%CONDA_ENV%".
  timeout /t 15 /nobreak >nul
  exit /b 1
)

:loop
echo [game.bat] %date% %time% launching kiosk>> "%LOG%"
start "ChessKiosk" /wait pythonw "%~dp0kiosk.py"

if exist "%STOP%" (
  del "%STOP%"
  echo [game.bat] %date% %time% operator stop requested; launcher exiting.>> "%LOG%"
  goto :end
)

echo [game.bat] %date% %time% kiosk exited unexpectedly; relaunching in 3s>> "%LOG%"
timeout /t 3 /nobreak >nul
goto :loop

:end
endlocal
