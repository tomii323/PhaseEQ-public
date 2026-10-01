@echo off
setlocal enabledelayedexpansion
set "PHASEEQ_LAUNCH_ARGS=%*"

cd /d "%~dp0"

rem ============================================================
rem APPY Common Launcher - Windows
rem Change this block for other APPY apps
rem ============================================================
set APP_NAME=PhaseEQ
set MAIN_FILE=phaseeq.py
set COMPOSITE_FILE=composite_streamlit_app.py
set BASE_PORT=8501
set MAX_PORT=8520
set VENV_DIR=.venv
set LOG_DIR=logs
set ENABLE_LOG=0
set FORCE_UPDATE=0
set MIN_PYTHON_MAJOR=3
set MIN_PYTHON_MINOR=12
set MAX_PYTHON_MAJOR=3
set MAX_PYTHON_MINOR=14
set WINDOWS_ARM64=0
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set WINDOWS_ARM64=1
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set WINDOWS_ARM64=1
set PYTHON_VERSION_FILE=%VENV_DIR%\.launcher-python-version

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--log" (
    set ENABLE_LOG=1
    shift
    goto parse_args
)
if /I "%~1"=="/log" (
    set ENABLE_LOG=1
    shift
    goto parse_args
)
if /I "%~1"=="--update" (
    set FORCE_UPDATE=1
    shift
    goto parse_args
)
if /I "%~1"=="/update" (
    set FORCE_UPDATE=1
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto usage
if /I "%~1"=="/help" goto usage
if /I "%~1"=="-h" goto usage
echo ERROR: Unknown option: %~1
goto usage_error

:args_done

if "%ENABLE_LOG%"=="1" (
    if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
    set LOG_FILE=%LOG_DIR%\%APP_NAME%_launcher.log
    echo ==== %DATE% %TIME% ==== > "%LOG_FILE%"
    call :main >> "%LOG_FILE%" 2>&1
) else (
    call :main
)

exit /b %errorlevel%

:usage
echo Usage: %~nx0 [--log] [--update]
echo   default  Start offline when the existing runtime is complete
echo   --update Update pip and dependencies before starting (network required)
echo   --log    Write launcher output to %LOG_DIR%\%APP_NAME%_launcher.log
exit /b 0

:usage_error
echo Usage: %~nx0 [--log] [--update]
exit /b 2

:main
echo == %APP_NAME% Launcher ==

if not exist "%MAIN_FILE%" (
    echo ERROR: %MAIN_FILE% was not found.
    pause
    exit /b 1
)
if not exist "%COMPOSITE_FILE%" (
    echo ERROR: %COMPOSITE_FILE% was not found.
    pause
    exit /b 1
)

call :find_python
if errorlevel 1 exit /b 1

call :check_python_version
if errorlevel 1 exit /b 1

if exist ".phaseeq-updates\pending.json" goto apply_app_update
if exist ".phaseeq-updates\journal.json" goto apply_app_update
goto app_update_done
:apply_app_update
(
    %PY% runtime\app_update.py --apply
    if errorlevel 10 (
        call "%~f0" %PHASEEQ_LAUNCH_ARGS%
        exit /b
    )
    if errorlevel 1 exit /b 1
)
:app_update_done

call :ensure_venv
if errorlevel 1 exit /b 1

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    call :recreate_venv
    if errorlevel 1 exit /b 1
    call "%VENV_DIR%\Scripts\activate.bat"
    if errorlevel 1 (
        echo ERROR: Failed to activate recreated virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

python -c "import sys; print('Python:', sys.executable)"
if errorlevel 1 (
    echo ERROR: Virtual environment Python is not working.
    call :recreate_venv
    if errorlevel 1 exit /b 1
    call "%VENV_DIR%\Scripts\activate.bat"
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found.
    pause
    exit /b 1
)

set RUNTIME_ACTION=
for /f "delims=" %%A in ('python runtime\check_runtime_environment.py --action') do set "RUNTIME_ACTION=%%A"
if "%RUNTIME_ACTION%"=="" set RUNTIME_ACTION=recreate

if "%FORCE_UPDATE%"=="1" (
    echo Online update requested.
    call :prepare_runtime_repair "%RUNTIME_ACTION%"
    if errorlevel 1 exit /b 1
    call :install_dependencies
    if errorlevel 1 exit /b 1
) else (
    set OFFLINE_RUNTIME_READY=0
    if /I "%RUNTIME_ACTION%"=="ready" (
        call :runtime_dependencies_ready
        if not errorlevel 1 set OFFLINE_RUNTIME_READY=1
    )
    if "!OFFLINE_RUNTIME_READY!"=="1" (
        echo Runtime dependencies are ready. Starting without network access.
    ) else (
        echo Runtime dependencies require action: %RUNTIME_ACTION%
        echo Online installation is required for this setup.
        call :prepare_runtime_repair "%RUNTIME_ACTION%"
        if errorlevel 1 exit /b 1
        call :install_dependencies
        if errorlevel 1 exit /b 1
    )
)

call :runtime_dependencies_ready
if errorlevel 1 (
    echo ERROR: Runtime dependency verification failed.
    pause
    exit /b 1
)

if "%PHASEEQ_LAUNCHER_CHECK_ONLY%"=="1" (
    echo Launcher environment check completed.
    exit /b 0
)

call :find_free_port
if errorlevel 1 exit /b 1
set PHASEEQ_SERVER_PORT=%PORT%
set PHASEEQ_PORT=%PORT%
set /A BASE_PORT=%PHASEEQ_PORT%+1
call :find_free_port
if errorlevel 1 exit /b 1
set COMPOSITE_ENGINE_PORT=%PORT%
set PHASEEQ_URL=http://localhost:%PHASEEQ_PORT%
set COMPOSITE_ENGINE_URL=http://localhost:%COMPOSITE_ENGINE_PORT%
if "%PHASEEQ_DATA_DIR%"=="" (
    set PHASEEQ_COMPOSITE_EXCHANGE_DIR=%CD%\data\tmp\composite_exchange
) else (
    set PHASEEQ_COMPOSITE_EXCHANGE_DIR=%PHASEEQ_DATA_DIR%\tmp\composite_exchange
)

echo.
echo Starting %APP_NAME%...
echo URL: http://localhost:%PHASEEQ_PORT%
echo Composite Engine: http://localhost:%COMPOSITE_ENGINE_PORT%
echo Network: local-only ^(external web features remain optional^)
echo.

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
powershell -NoProfile -ExecutionPolicy Bypass -File "runtime\run_linked_streamlit_windows.ps1" -PythonExe "%VENV_DIR%\Scripts\python.exe" -MainFile "%MAIN_FILE%" -CompositeFile "%COMPOSITE_FILE%" -MainPort %PHASEEQ_PORT% -CompositePort %COMPOSITE_ENGINE_PORT%
set STREAMLIT_EXIT_CODE=%errorlevel%
if not "%STREAMLIT_EXIT_CODE%"=="0" echo ERROR: Streamlit exited with code %STREAMLIT_EXIT_CODE%.

pause
exit /b %STREAMLIT_EXIT_CODE%

:find_python
set PY=
call :refresh_python_paths

call :try_python "py -3.13"
if not errorlevel 1 exit /b 0

call :try_python "py -3.12"
if not errorlevel 1 exit /b 0

call :try_python "py -3"
if not errorlevel 1 exit /b 0

call :try_python "python"
if not errorlevel 1 exit /b 0

echo Python 3.12 or 3.13 was not found.
if "%WINDOWS_ARM64%"=="1" (
    echo Windows ARM64 detected. Installing x64 Python 3.12 for binary package compatibility...
) else (
    echo Trying to install Python 3.12 by winget...
)

where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: winget was not found.
    echo Please install Python 3.12 or 3.13 manually.
    pause
    exit /b 1
)

if "%WINDOWS_ARM64%"=="1" (
    winget install -e --id Python.Python.3.12 --architecture x64 --scope user --force
) else (
    winget install -e --id Python.Python.3.12
)
if errorlevel 1 (
    echo ERROR: Python installation failed.
    pause
    exit /b 1
)

rem The current cmd.exe does not receive PATH changes made by the Python
rem installer. Refresh the known per-user and all-user install locations,
rem including the directory names used by Windows on ARM64.
call :refresh_python_paths

call :try_python "py -3.12"
if not errorlevel 1 exit /b 0

call :try_python "py -3"
if not errorlevel 1 exit /b 0

call :try_python "python"
if not errorlevel 1 exit /b 0

echo ERROR: Python 3.12 or 3.13 could not be found.
pause
exit /b 1

:refresh_python_paths
rem Prepend 3.12 first so that the later 3.13 entries take precedence.
call :prepend_python_path "%ProgramFiles%\Python312-arm64"
call :prepend_python_path "%ProgramFiles%\Python312"
call :prepend_python_path "%LocalAppData%\Programs\Python\Python312-arm64"
call :prepend_python_path "%LocalAppData%\Programs\Python\Python312"
call :prepend_python_path "%ProgramFiles%\Python313-arm64"
call :prepend_python_path "%ProgramFiles%\Python313"
call :prepend_python_path "%LocalAppData%\Programs\Python\Python313-arm64"
call :prepend_python_path "%LocalAppData%\Programs\Python\Python313"
call :prepend_registered_python "HKCU\Software\Python\PythonCore\3.12\InstallPath"
call :prepend_registered_python "HKLM\Software\Python\PythonCore\3.12\InstallPath"
call :prepend_registered_python "HKCU\Software\Python\PythonCore\3.13\InstallPath"
call :prepend_registered_python "HKLM\Software\Python\PythonCore\3.13\InstallPath"
if exist "%LocalAppData%\Programs\Python" (
    for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do call :prepend_python_path "%%~fD"
)
exit /b 0

:prepend_registered_python
set "REGISTERED_PYTHON_PATH="
for /f "tokens=2,*" %%A in ('reg query "%~1" /ve 2^>nul ^| findstr /c:"REG_SZ"') do set "REGISTERED_PYTHON_PATH=%%B"
if defined REGISTERED_PYTHON_PATH call :prepend_python_path "!REGISTERED_PYTHON_PATH!"
exit /b 0

:prepend_python_path
if not exist "%~1\python.exe" exit /b 0
"%~1\python.exe" -c "import sys, sysconfig; raise SystemExit(0 if sys.version_info[:2] in [(3, 12), (3, 13)] and sysconfig.get_platform().lower() == 'win-amd64' else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PATH=%~1;%~1\Scripts;!PATH!"
echo Detected Python installation: %~1
exit /b 0

:try_python
set "CANDIDATE=%~1"
set "PYTHON_PROBE_RESULT="
for /f "delims=" %%V in ('%CANDIDATE% -c "import sys, sysconfig; print(313121 if sys.version_info[:2] in [(3, 12), (3, 13)] and sysconfig.get_platform().lower() == 'win-amd64' else 0)" 2^>nul') do set "PYTHON_PROBE_RESULT=%%V"
if not "!PYTHON_PROBE_RESULT!"=="313121" exit /b 1
set "PY=%CANDIDATE%"
exit /b 0

:check_python_version
%PY% -c "import sys; raise SystemExit(0 if (%MIN_PYTHON_MAJOR%, %MIN_PYTHON_MINOR%) <= sys.version_info[:2] < (%MAX_PYTHON_MAJOR%, %MAX_PYTHON_MINOR%) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.12 or 3.13 is required. Python 3.14 is not supported by the validated runtime.
    %PY% --version
    pause
    exit /b 1
)
%PY% --version
exit /b 0

:ensure_venv
set NEED_RECREATE=0
set CURRENT_PY_VERSION=

for /f "delims=" %%V in ('%PY% -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "CURRENT_PY_VERSION=%%V"
if "%CURRENT_PY_VERSION%"=="" (
    echo ERROR: Failed to check Python version.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%" (
    set NEED_RECREATE=1
) else if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo activate script is missing.
    set NEED_RECREATE=1
) else if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo python executable is missing.
    set NEED_RECREATE=1
) else if not exist "%VENV_DIR%\Scripts\pip.exe" (
    echo pip executable is missing.
    set NEED_RECREATE=1
) else (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo python executable is broken.
        set NEED_RECREATE=1
    ) else (
        "%VENV_DIR%\Scripts\python.exe" -c "import sys, sysconfig; raise SystemExit(0 if sys.version_info[:2] in [(3, 12), (3, 13)] and sysconfig.get_platform().lower() == 'win-amd64' else 1)" >nul 2>nul
        if errorlevel 1 (
            echo Virtual environment uses an unsupported Python version or architecture.
            "%VENV_DIR%\Scripts\python.exe" --version
            set NEED_RECREATE=1
        )
    )
)

if "%NEED_RECREATE%"=="1" (
    echo Virtual environment is missing, broken, or outdated.
    call :recreate_venv
    if errorlevel 1 exit /b 1
)

if "%NEED_RECREATE%"=="0" (
    for /f "delims=" %%V in ('"%VENV_DIR%\Scripts\python.exe" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "VENV_ACTUAL_PY_VERSION=%%V"
    echo !VENV_ACTUAL_PY_VERSION!>"%PYTHON_VERSION_FILE%"
)

exit /b 0

:recreate_venv
echo Recreating virtual environment...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
%PY% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Virtual environment is incomplete.
    pause
    exit /b 1
)
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ERROR: Virtual environment is incomplete.
    pause
    exit /b 1
)
if not exist "%VENV_DIR%\Scripts\pip.exe" (
    echo ERROR: Virtual environment is incomplete.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('"%VENV_DIR%\Scripts\python.exe" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do set "CREATED_PY_VERSION=%%V"
echo %CREATED_PY_VERSION%>"%PYTHON_VERSION_FILE%"
exit /b 0

:find_free_port
set PORT=
for /L %%P in (%BASE_PORT%,1,%MAX_PORT%) do (
    netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul
    if errorlevel 1 (
        set PORT=%%P
        goto :port_found
    )
)

echo ERROR: No free port found from %BASE_PORT% to %MAX_PORT%.
pause
exit /b 1

:port_found
exit /b 0

:runtime_dependencies_ready
python runtime\check_runtime_environment.py >nul 2>nul
if errorlevel 1 (
    python runtime\check_runtime_environment.py
    exit /b 1
)
python runtime\check_runtime_environment.py --optional-warnings
python -c "import altair, numpy, pandas, scipy, soundfile, streamlit; import composite_engine" >nul 2>nul
if errorlevel 1 (
    echo Runtime import check failed. Showing the Python traceback:
    python -c "import altair, numpy, pandas, scipy, soundfile, streamlit; import composite_engine"
    exit /b 1
)
exit /b 0

:prepare_runtime_repair
if /I not "%~1"=="recreate" exit /b 0
echo Multiple incompatible or broken packages were found.
if exist "%VENV_DIR%\Scripts\deactivate.bat" call "%VENV_DIR%\Scripts\deactivate.bat"
call :recreate_venv
if errorlevel 1 exit /b 1
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1
exit /b 0

:install_dependencies
python -m pip --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: pip is not available in virtual environment.
    call :recreate_venv
    if errorlevel 1 exit /b 1
    call "%VENV_DIR%\Scripts\activate.bat"
)

set TOO_NEW_PACKAGES=
for /f "delims=" %%P in ('python runtime\check_runtime_environment.py --too-new') do set "TOO_NEW_PACKAGES=%%P"
if not "%TOO_NEW_PACKAGES%"=="" (
    echo Removing packages newer than the supported environment: %TOO_NEW_PACKAGES%
    python -m pip uninstall -y %TOO_NEW_PACKAGES%
    if errorlevel 1 exit /b 1
)

echo Updating validated packaging tools...
python -m pip install --constraint constraints\test-environment.txt pip setuptools wheel
if errorlevel 1 (
    echo ERROR: pip update failed.
    pause
    exit /b 1
)

echo Installing the latest validated runtime dependencies...
python -m pip install --upgrade --constraint constraints\test-environment.txt -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)
exit /b 0
