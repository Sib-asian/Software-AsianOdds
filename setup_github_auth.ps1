#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup autenticazione GitHub con GitHub CLI
#>

# Aggiorna PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host ""
Write-Host "=== CONFIGURAZIONE AUTENTICAZIONE GITHUB ===" -ForegroundColor Cyan
Write-Host ""

# Verifica GitHub CLI
Write-Host "[1/2] Verifica GitHub CLI..." -ForegroundColor Yellow
try {
    $ghVersion = gh --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] GitHub CLI trovato" -ForegroundColor Green
        Write-Host "  $ghVersion" -ForegroundColor Gray
    } else {
        throw "GitHub CLI non trovato"
    }
} catch {
    Write-Host "  [ERRORE] GitHub CLI non trovato" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Installazione GitHub CLI..." -ForegroundColor Yellow
    winget install --id GitHub.cli --accept-package-agreements --accept-source-agreements --silent
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Verifica di nuovo
    Start-Sleep -Seconds 2
    $ghVersion = gh --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] GitHub CLI installato" -ForegroundColor Green
    } else {
        Write-Host "  [ERRORE] Installazione fallita. Riavvia PowerShell e riprova." -ForegroundColor Red
        exit 1
    }
}

# Verifica se già autenticato
Write-Host ""
Write-Host "[2/2] Verifica autenticazione..." -ForegroundColor Yellow
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -eq 0 -and $authStatus -match "Logged in") {
    Write-Host "  [OK] Già autenticato!" -ForegroundColor Green
    Write-Host ""
    gh auth status
    Write-Host ""
    Write-Host "Autenticazione già configurata. Puoi procedere con la creazione di PR!" -ForegroundColor Green
} else {
    Write-Host "  [INFO] Autenticazione richiesta" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Avvio processo di autenticazione..." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Seguirà una procedura interattiva:" -ForegroundColor White
    Write-Host "1. Scegli il metodo di autenticazione (consigliato: GitHub.com)" -ForegroundColor Gray
    Write-Host "2. Scegli il protocollo (consigliato: HTTPS)" -ForegroundColor Gray
    Write-Host "3. Autorizza GitHub CLI aprendo il browser" -ForegroundColor Gray
    Write-Host "4. Completa l'autenticazione nel browser" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Premi INVIO per continuare..." -ForegroundColor Yellow
    Read-Host
    
    # Avvia autenticazione
    gh auth login
}

Write-Host ""
Write-Host "=== AUTENTICAZIONE COMPLETATA ===" -ForegroundColor Green
Write-Host ""
Write-Host "Ora puoi creare PR con:" -ForegroundColor Cyan
Write-Host "  .\quick_pr.ps1 -Message 'Descrizione modifiche'" -ForegroundColor White
Write-Host ""

