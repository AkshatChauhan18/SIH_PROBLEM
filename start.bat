@echo off
title AI World Model - SOC Attack Forecasting Dashboard
cd /d "%~dp0"

echo ======================================================================
echo    AI-Based Network Attack Forecasting via World Model (SIH26153)
echo    Cybersecurity Operations Center (SOC) Command Dashboard
echo ======================================================================
echo.

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=false

echo Initializing SOC Dashboard server on http://localhost:8501 ...
echo Opening your default browser in 3 seconds...
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8501"

".venv\Scripts\streamlit.exe" run dashboard\app.py --server.port 8501 --server.address localhost

pause
