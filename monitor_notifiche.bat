@echo off
REM Script batch per monitorare solo le notifiche
echo ========================================
echo   MONITORAGGIO NOTIFICHE LIVE
echo ========================================
echo.
echo Premi Ctrl+C per fermare
echo.

cd /d "%~dp0"

REM Monitora solo le notifiche
powershell -Command "Get-Content automation_24h.log -Wait | Select-String -Pattern 'opportunità|notifica|Live betting opportunity|confidence: [6-9][0-9]%%'"


















