@echo off
rem Double-click to start the dashboard (first run installs the libraries).
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
%PY% -m pip install -q -r requirements.txt
%PY% -m streamlit run dashboard/app.py
pause
