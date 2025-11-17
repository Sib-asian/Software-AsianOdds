# Script per aggiungere automazione al Startup di Windows (NON richiede admin)
# Questo aggiunge uno shortcut nella cartella Startup

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INSTALLAZIONE AUTOMAZIONE 24/7" -ForegroundColor Cyan
Write-Host "  (Senza privilegi amministratore)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$startupScript = Join-Path $scriptDir "start_automation_background.ps1"
$startupFolder = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupFolder "Automation24H.lnk"

Write-Host "Configurazione:" -ForegroundColor Green
Write-Host "   Directory: $scriptDir" -ForegroundColor Gray
Write-Host "   Startup Folder: $startupFolder" -ForegroundColor Gray
Write-Host ""

# Verifica che lo script esista
if (-not (Test-Path $startupScript)) {
    Write-Host "ERRORE: Script startup non trovato: $startupScript" -ForegroundColor Red
    exit 1
}

# Crea shortcut nella cartella Startup
try {
    $WScriptShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
    $Shortcut.TargetPath = "powershell.exe"
    $Shortcut.Arguments = "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$startupScript`""
    $Shortcut.WorkingDirectory = $scriptDir
    $Shortcut.Description = "Avvia automazione 24/7 betting system"
    $Shortcut.Save()
    
    Write-Host "Shortcut creato nella cartella Startup!" -ForegroundColor Green
    Write-Host "   Path: $shortcutPath" -ForegroundColor Gray
} catch {
    Write-Host "ERRORE durante la creazione dello shortcut: $_" -ForegroundColor Red
    exit 1
}

# Crea anche uno script batch per avvio immediato
$batchFile = Join-Path $scriptDir "start_automation_now.bat"
@"
@echo off
cd /d "%~dp0"
start /min powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "start_automation_background.ps1"
"@ | Out-File -FilePath $batchFile -Encoding ASCII

Write-Host ""
Write-Host "INSTALLAZIONE COMPLETATA!" -ForegroundColor Green
Write-Host ""
Write-Host "Il sistema e configurato per:" -ForegroundColor Cyan
Write-Host "   - Avviarsi automaticamente all'accesso Windows" -ForegroundColor Green
Write-Host "   - Girare in background senza finestre" -ForegroundColor Green
Write-Host ""
Write-Host "Per avviare subito, esegui: start_automation_now.bat" -ForegroundColor Yellow
Write-Host ""
Write-Host "TUTTO PRONTO!" -ForegroundColor Green
Write-Host ""

