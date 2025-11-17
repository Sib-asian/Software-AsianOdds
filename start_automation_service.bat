@echo off
REM Script batch per avviare manualmente l'automazione 24/7
REM Utile per test o avvio manuale

echo ========================================
echo   AVVIO AUTOMAZIONE 24/7
echo ========================================
echo.

cd /d "%~dp0"

REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato!
    echo Installa Python da https://www.python.org/
    pause
    exit /b 1
)

echo Avvio servizio automazione...
echo.

python automation_service_wrapper.py

if errorlevel 1 (
    echo.
    echo ERRORE durante l'esecuzione!
    pause
)

