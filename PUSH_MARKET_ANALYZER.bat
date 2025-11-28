@echo off
echo ========================================
echo   MARKET ANALYZER - PUSH SU GITHUB
echo ========================================
echo.

REM Verifica PowerShell
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRORE: PowerShell non trovato!
    pause
    exit /b 1
)

REM Esegui script PowerShell
powershell.exe -ExecutionPolicy Bypass -File "%~dp0PUSH_MARKET_ANALYZER.ps1"

pause

