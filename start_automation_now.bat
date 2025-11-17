@echo off
cd /d "%~dp0"
start /min powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "start_automation_background.ps1"
