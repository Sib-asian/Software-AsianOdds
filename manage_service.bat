@echo off
REM Script batch per gestire il servizio automazione 24/7

:menu
cls
echo ========================================
echo   GESTIONE AUTOMAZIONE 24/7
echo ========================================
echo.
echo 1. Avvia servizio
echo 2. Ferma servizio
echo 3. Riavvia servizio
echo 4. Stato servizio
echo 5. Visualizza log
echo 6. Installa come servizio Windows (richiede admin)
echo 7. Rimuovi servizio Windows (richiede admin)
echo 8. Esci
echo.
set /p choice="Scegli opzione (1-8): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto status
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto install
if "%choice%"=="7" goto uninstall
if "%choice%"=="8" goto end
goto menu

:start
echo.
echo Avvio servizio...
powershell -Command "Start-ScheduledTask -TaskName 'Automation24H_BettingSystem'"
timeout /t 2 >nul
goto status

:stop
echo.
echo Arresto servizio...
powershell -Command "Stop-ScheduledTask -TaskName 'Automation24H_BettingSystem'"
timeout /t 2 >nul
goto status

:restart
echo.
echo Riavvio servizio...
powershell -Command "Restart-ScheduledTask -TaskName 'Automation24H_BettingSystem'"
timeout /t 2 >nul
goto status

:status
echo.
echo Stato servizio:
powershell -Command "Get-ScheduledTask -TaskName 'Automation24H_BettingSystem' | Format-List"
echo.
pause
goto menu

:logs
echo.
echo Apertura cartella log...
if exist "logs" (
    explorer logs
) else (
    echo Cartella log non trovata!
    pause
)
goto menu

:install
echo.
echo Installazione servizio Windows...
echo (Richiede privilegi amministratore)
powershell -ExecutionPolicy Bypass -File "%~dp0install_service.ps1"
pause
goto menu

:uninstall
echo.
echo Rimozione servizio Windows...
echo (Richiede privilegi amministratore)
powershell -ExecutionPolicy Bypass -File "%~dp0uninstall_service.ps1"
pause
goto menu

:end
exit

