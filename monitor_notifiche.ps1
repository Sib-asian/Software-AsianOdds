# Script PowerShell per Monitorare SOLO le Notifiche
# Uso: .\monitor_notifiche.ps1

Write-Host "========================================" -ForegroundColor Green
Write-Host "  MONITORAGGIO NOTIFICHE LIVE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Premi Ctrl+C per fermare" -ForegroundColor Yellow
Write-Host ""

$logPath = Join-Path $PSScriptRoot "automation_24h.log"

if (-not (Test-Path $logPath)) {
    Write-Host "❌ File di log non trovato!" -ForegroundColor Red
    exit 1
}

# Monitora solo le notifiche
Get-Content $logPath -Wait | Select-String -Pattern "opportunità|notifica|Live betting opportunity|confidence: [6-9][0-9]%" | ForEach-Object {
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
    Write-Host $_.Line -ForegroundColor Green
}


















