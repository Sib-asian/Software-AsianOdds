# Script PowerShell per installare automazione 24/7 come servizio Windows
# Versione automatica senza input utente

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INSTALLAZIONE AUTOMAZIONE 24/7" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica privilegi amministratore
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERRORE: Questo script richiede privilegi amministratore!" -ForegroundColor Red
    exit 1
}

# Percorsi
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    Write-Host "ERRORE: Python non trovato!" -ForegroundColor Red
    exit 1
}

$wrapperScript = Join-Path $scriptDir "automation_service_wrapper.py"
$taskName = "Automation24H_BettingSystem"

Write-Host "Configurazione:" -ForegroundColor Green
Write-Host "   Directory: $scriptDir" -ForegroundColor Gray
Write-Host "   Python: $pythonExe" -ForegroundColor Gray
Write-Host "   Script: $wrapperScript" -ForegroundColor Gray
Write-Host "   Task Name: $taskName" -ForegroundColor Gray
Write-Host ""

# Verifica che lo script esista
if (-not (Test-Path $wrapperScript)) {
    Write-Host "ERRORE: Script wrapper non trovato: $wrapperScript" -ForegroundColor Red
    exit 1
}

# Crea cartella logs se non esiste
$logsDir = Join-Path $scriptDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    Write-Host "Cartella logs creata" -ForegroundColor Green
}

# Rimuovi task esistente se presente
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Rimozione task esistente..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# Crea action
$action = New-ScheduledTaskAction -Execute $pythonExe -Argument "`"$wrapperScript`"" -WorkingDirectory $scriptDir

# Crea trigger: all'avvio + ogni 5 minuti (per sicurezza)
$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)

# Impostazioni: riavvia automaticamente se fallisce
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -MultipleInstances IgnoreNew

# Crea principal (esegui come utente corrente)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Registra task
Write-Host "Creazione task schedulato..." -ForegroundColor Green
try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger @($trigger1, $trigger2) `
        -Settings $settings `
        -Principal $principal `
        -Description "Automazione 24/7 per sistema betting" `
        -Force | Out-Null
    
    Write-Host "Task creato con successo!" -ForegroundColor Green
} catch {
    Write-Host "Errore durante la creazione del task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "INSTALLAZIONE COMPLETATA!" -ForegroundColor Green
Write-Host ""
Write-Host "Il servizio e configurato per:" -ForegroundColor Cyan
Write-Host "   - Avviarsi automaticamente all'avvio Windows" -ForegroundColor Green
Write-Host "   - Riavviarsi automaticamente se crasha" -ForegroundColor Green
Write-Host "   - Verificare ogni 5 minuti che sia in esecuzione" -ForegroundColor Green
Write-Host ""

# Avvia subito il task
Write-Host "Avvio immediato del servizio..." -ForegroundColor Green
try {
    Start-ScheduledTask -TaskName $taskName
    Start-Sleep -Seconds 3
    
    $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
    $taskState = (Get-ScheduledTask -TaskName $taskName).State
    
    Write-Host "Servizio avviato!" -ForegroundColor Green
    Write-Host "   Stato: $taskState" -ForegroundColor Gray
    if ($taskInfo.LastRunTime) {
        Write-Host "   Ultimo avvio: $($taskInfo.LastRunTime)" -ForegroundColor Gray
    }
} catch {
    Write-Host "Errore durante l'avvio: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Log file: $logsDir\automation_service_*.log" -ForegroundColor Cyan
Write-Host ""
Write-Host "TUTTO PRONTO! Il sistema ora gira 24/7 automaticamente!" -ForegroundColor Green
Write-Host ""
