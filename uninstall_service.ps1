# Script PowerShell per rimuovere servizio automazione 24/7
# Esegui come Amministratore

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RIMOZIONE AUTOMAZIONE 24/7" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica privilegi amministratore
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "❌ ERRORE: Questo script richiede privilegi amministratore!" -ForegroundColor Red
    Read-Host "Premi Enter per uscire"
    exit 1
}

$taskName = "Automation24H_BettingSystem"

# Verifica se esiste
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $existingTask) {
    Write-Host "⚠️  Task '$taskName' non trovato" -ForegroundColor Yellow
    Read-Host "Premi Enter per uscire"
    exit 0
}

# Ferma task se in esecuzione
Write-Host "🛑 Arresto task..." -ForegroundColor Yellow
Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Rimuovi task
Write-Host "🗑️  Rimozione task..." -ForegroundColor Yellow
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false

Write-Host ""
Write-Host "✅ Servizio rimosso con successo!" -ForegroundColor Green
Write-Host ""
Read-Host "Premi Enter per uscire"

