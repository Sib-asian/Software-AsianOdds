# Script per avviare automazione in background senza privilegi admin
# Questo script può essere aggiunto al Startup di Windows

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$wrapperScript = Join-Path $scriptDir "automation_service_wrapper.py"
$logsDir = Join-Path $scriptDir "logs"
$logFile = Join-Path $logsDir "startup.log"

# Crea cartella logs se non esiste
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

# Log avvio
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "[$timestamp] Avvio automazione in background..."

# Verifica se Python è disponibile
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}

if (-not $pythonExe) {
    Add-Content -Path $logFile -Value "[$timestamp] ERRORE: Python non trovato!"
    exit 1
}

# Verifica se lo script esiste
if (-not (Test-Path $wrapperScript)) {
    Add-Content -Path $logFile -Value "[$timestamp] ERRORE: Script wrapper non trovato: $wrapperScript"
    exit 1
}

# Avvia processo in background
try {
    $process = Start-Process -FilePath $pythonExe -ArgumentList "`"$wrapperScript`"" -WorkingDirectory $scriptDir -WindowStyle Hidden -PassThru
    Add-Content -Path $logFile -Value "[$timestamp] Processo avviato con PID: $($process.Id)"
} catch {
    Add-Content -Path $logFile -Value "[$timestamp] ERRORE durante l'avvio: $_"
    exit 1
}

