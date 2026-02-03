@echo off
title Quant Trading System Server
:start
echo [%date% %time%] Starting Trading Server...
python main.py
echo [%date% %time%] Server exited with code %errorlevel%. Restarting in 5 seconds...
timeout /t 5
goto start
