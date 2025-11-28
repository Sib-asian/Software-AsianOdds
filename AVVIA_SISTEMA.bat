@echo off
chcp 65001 >nul
title Sistema 24/7 - Live Betting
color 0A

echo ================================================================================
echo 🚀 AVVIO SISTEMA 24/7 - LIVE BETTING
echo ================================================================================
echo.
echo Il sistema monitorerà le partite live ogni 5 minuti
echo e invierà messaggi Telegram quando troverà opportunità valide.
echo.
echo ⚠️  IMPORTANTE: Non chiudere questa finestra!
echo    Il sistema deve rimanere aperto per funzionare.
echo.
echo 💡 Per fermare il sistema, premi CTRL+C o chiudi questa finestra.
echo.
echo ================================================================================
echo.

cd /d "%~dp0"

REM Ferma processi Python esistenti
echo 🛑 Fermo processi Python esistenti...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Avvia sistema
echo ▶️  Avvio sistema...
echo.

python avvia_sistema_robusto.py

echo.
echo ================================================================================
echo ⏸️  Sistema fermato
echo ================================================================================
pause
