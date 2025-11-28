@echo off
chcp 65001 >nul
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     📊 STATO SISTEMA 24/7                                    ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo 🔍 Verifica processi Python in esecuzione...
echo.

tasklist /FI "IMAGENAME eq python.exe" /FO TABLE 2>nul | find /I "python.exe" >nul
if errorlevel 1 (
    echo ❌ Nessun processo Python trovato
    echo    Il sistema NON è in esecuzione
) else (
    echo ✅ Processi Python trovati:
    tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
    echo.
    echo Il sistema potrebbe essere in esecuzione
)

echo.
echo 📋 Ultimi log disponibili:
echo.

if exist "logs\automation_service_*.log" (
    for /f "delims=" %%f in ('dir /b /o-d logs\automation_service_*.log 2^>nul') do (
        echo    📄 logs\%%f
        echo    Ultime 5 righe:
        powershell -Command "Get-Content 'logs\%%f' -Tail 5 -ErrorAction SilentlyContinue"
        goto :found
    )
    :found
) else (
    echo    ⚠️  Nessun file di log trovato
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo.
pause

















