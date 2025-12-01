# Script PowerShell per Monitorare i Log in Tempo Reale
# Uso: .\monitor_logs.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MONITORAGGIO LOG LIVE BETTING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Premi Ctrl+C per fermare il monitoraggio" -ForegroundColor Yellow
Write-Host ""

# Vai nella cartella del progetto
$logPath = Join-Path $PSScriptRoot "automation_24h.log"

if (-not (Test-Path $logPath)) {
    Write-Host "❌ File di log non trovato: $logPath" -ForegroundColor Red
    exit 1
}

Write-Host "📊 Monitoraggio avviato..." -ForegroundColor Green
Write-Host "📁 File: $logPath" -ForegroundColor Gray
Write-Host ""

# Monitora solo le righe importanti
Get-Content $logPath -Wait -Tail 100 | ForEach-Object {
    $line = $_
    
    # Notifiche inviate (verde)
    if ($line -match "opportunità notificata|Live betting opportunity") {
        Write-Host $line -ForegroundColor Green
    }
    # Partite analizzate (giallo)
    elseif ($line -match "Analizzando partita LIVE") {
        # Evidenzia solo partite senior (non giovanili)
        if ($line -notmatch "U21|U19|U17|U23") {
            Write-Host $line -ForegroundColor Yellow
        }
    }
    # Errori (rosso)
    elseif ($line -match "ERROR|Error|Exception") {
        Write-Host $line -ForegroundColor Red
    }
    # Confidence (ciano)
    elseif ($line -match "confidence: [6-9][0-9]%") {
        Write-Host $line -ForegroundColor Cyan
    }
    # Avvisi (magenta)
    elseif ($line -match "⚠️|WARNING") {
        Write-Host $line -ForegroundColor Magenta
    }
}


















