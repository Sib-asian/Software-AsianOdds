# Script PowerShell per avviare dashboard Streamlit in modalità remota
# Accessibile da cellulare sulla stessa rete WiFi

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AVVIO DASHBOARD STREAMLIT REMOTO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Trova IP locale
Write-Host "Trovando IP del PC..." -ForegroundColor Green
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254.*"} | Select-Object -First 1).IPAddress

if ($ipAddress) {
    Write-Host ""
    Write-Host "IP del PC: $ipAddress" -ForegroundColor Green
    Write-Host ""
    Write-Host "Dashboard accessibile su:" -ForegroundColor Cyan
    Write-Host "  http://$ipAddress:8501" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Sul cellulare (stessa WiFi):" -ForegroundColor Cyan
    Write-Host "  1. Apri browser" -ForegroundColor Gray
    Write-Host "  2. Vai a: http://$ipAddress:8501" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Premi Ctrl+C per fermare il server" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "⚠️  Impossibile trovare IP, usa localhost" -ForegroundColor Yellow
    $ipAddress = "localhost"
}

# Avvia Streamlit in modalità remota
Write-Host "Avvio Streamlit..." -ForegroundColor Green
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501

