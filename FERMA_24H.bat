@echo off
chcp 65001 >nul
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     🛑 ARRESTO SISTEMA 24/7                                  ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo 🔍 Cercando processi in esecuzione...

REM Trova tutti i processi Python che potrebbero essere il nostro sistema
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /I "PID"') do (
    echo    Processo trovato: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo    ✅ Processo %%a terminato
    )
)

echo.
echo ✅ Sistema fermato!
echo.
pause

















