@echo off
REM Script per avviare dashboard Streamlit in modalità remota
REM Accessibile da cellulare sulla stessa rete WiFi

echo ========================================
echo   AVVIO DASHBOARD STREAMLIT REMOTO
echo ========================================
echo.

cd /d "%~dp0"

REM Trova IP locale
echo Trovando IP del PC...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%

echo.
echo IP del PC: %IP%
echo.
echo Dashboard accessibile su:
echo   http://%IP%:8501
echo.
echo Sul cellulare (stessa WiFi):
echo   1. Apri browser
echo   2. Vai a: http://%IP%:8501
echo.
echo Premi Ctrl+C per fermare il server
echo.

REM Avvia Streamlit in modalità remota
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501

pause

