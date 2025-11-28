@echo off
chcp 65001 >nul
echo ================================================================================
echo 🚀 AVVIO SISTEMA 24/7
echo ================================================================================
echo.

cd /d "%~dp0"

REM Ferma processi Python esistenti
echo 🛑 Fermo processi Python esistenti...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Avvia sistema
echo ▶️  Avvio sistema...
start /B python avvia_sistema_robusto.py

echo.
echo ✅ Sistema avviato in background
echo 📊 I log sono in: logs\automation_service_*.log
echo.
echo 💡 Per fermare il sistema, esegui: FERMA_24H.bat
echo.
pause
