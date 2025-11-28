@echo off
REM Script batch per monitorare i log (non richiede permessi PowerShell)
echo ========================================
echo   MONITORAGGIO LOG LIVE BETTING
echo ========================================
echo.
echo Premi Ctrl+C per fermare il monitoraggio
echo.

cd /d "%~dp0"

REM Monitora il file di log in tempo reale
powershell -Command "Get-Content automation_24h.log -Wait -Tail 50"


















