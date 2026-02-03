@echo off
echo Cleaning up ALL Python trading processes...
taskkill /F /IM python.exe /T
taskkill /F /IM python3.11.exe /T
taskkill /F /IM python3.exe /T
echo Done! All ports should be released now.
echo You can now run run_all_systems.bat again.
pause