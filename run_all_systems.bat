@echo off
title AI Vision Master Controller
echo Starting AI Vision Dashboard...
start cmd /k "Dashboard\run_dashboard.bat"
timeout /t 3
echo Starting Trading Engine...
start cmd /k "run_server.bat"
echo All systems launching. Visit http://localhost:8000 to see the dashboard.
pause
