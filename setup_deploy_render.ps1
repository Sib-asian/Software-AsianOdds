#!/usr/bin/env pwsh
# Script di Setup per Deploy su Render.com
# Questo script ti aiuta a preparare tutto per il deploy su Render.com

Write-Host "🚀 Setup Deploy su Render.com" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Funzione per verificare se un comando esiste
function Test-Command {
    param($command)
    try {
        if (Get-Command $command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

# Verifica file necessari
Write-Host "📋 Verifica file necessari..." -ForegroundColor Yellow

$requiredFiles = @(
    "automation_24h.py",
    "start_automation.py",
    "Dockerfile.automation",
    "requirements.automation.txt"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file trovato" -ForegroundColor Green
    }
    else {
        Write-Host "  ❌ $file NON trovato!" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "❌ Alcuni file necessari mancano!" -ForegroundColor Red
    Write-Host "Assicurati di essere nella cartella corretta." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "✅ Tutti i file necessari sono presenti!" -ForegroundColor Green
Write-Host ""

# Verifica Git
Write-Host "🔍 Verifica Git..." -ForegroundColor Yellow

if (Test-Command "git") {
    Write-Host "  ✅ Git installato" -ForegroundColor Green
    
    # Verifica se è un repository Git
    if (Test-Path ".git") {
        Write-Host "  ✅ Repository Git trovato" -ForegroundColor Green
        
        # Verifica se è connesso a GitHub
        $remote = git remote get-url origin 2>$null
        if ($remote) {
            Write-Host "  ✅ Remote GitHub configurato: $remote" -ForegroundColor Green
        }
        else {
            Write-Host "  ⚠️  Nessun remote GitHub configurato" -ForegroundColor Yellow
            Write-Host "     Devi configurare: git remote add origin <URL>" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  ⚠️  Questa cartella non è un repository Git" -ForegroundColor Yellow
        Write-Host "     Inizializza con: git init" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  ❌ Git non installato" -ForegroundColor Red
    Write-Host ""
    Write-Host "📥 Installa Git:" -ForegroundColor Yellow
    Write-Host "   1. Vai su: https://git-scm.com/download/win" -ForegroundColor Cyan
    Write-Host "   2. Scarica e installa Git" -ForegroundColor Cyan
    Write-Host "   3. Riavvia PowerShell" -ForegroundColor Cyan
    Write-Host "   4. Oppure usa GitHub Desktop: https://desktop.github.com/" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host ""

# Mostra prossimi passi
Write-Host "📝 PROSSIMI PASSI:" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Assicurati che il codice sia su GitHub:" -ForegroundColor Yellow
Write-Host "   - Se usi Git: git add . && git commit -m 'Ready for deploy' && git push" -ForegroundColor White
Write-Host "   - Se usi GitHub Desktop: Publish repository" -ForegroundColor White
Write-Host ""

Write-Host "2. Crea account su Render.com:" -ForegroundColor Yellow
Write-Host "   - Vai su: https://render.com" -ForegroundColor Cyan
Write-Host "   - Registrati con GitHub" -ForegroundColor White
Write-Host ""

Write-Host "3. Crea Background Worker:" -ForegroundColor Yellow
Write-Host "   - Dashboard → New + → Background Worker" -ForegroundColor White
Write-Host "   - Connetti repository GitHub" -ForegroundColor White
Write-Host "   - Configura:" -ForegroundColor White
Write-Host "     * Name: automation-24h" -ForegroundColor Gray
Write-Host "     * Environment: Docker" -ForegroundColor Gray
Write-Host "     * Dockerfile Path: ./Dockerfile.automation" -ForegroundColor Gray
Write-Host ""

Write-Host "4. Aggiungi variabili ambiente:" -ForegroundColor Yellow
Write-Host "   * TELEGRAM_BOT_TOKEN=il_tuo_token" -ForegroundColor Gray
Write-Host "   * TELEGRAM_CHAT_ID=il_tuo_chat_id" -ForegroundColor Gray
Write-Host "   * AUTOMATION_MIN_EV=8.0" -ForegroundColor Gray
Write-Host "   * AUTOMATION_MIN_CONFIDENCE=70.0" -ForegroundColor Gray
Write-Host "   * AUTOMATION_UPDATE_INTERVAL=300" -ForegroundColor Gray
Write-Host "   * PYTHONUNBUFFERED=1" -ForegroundColor Gray
Write-Host ""

Write-Host "5. Deploy!" -ForegroundColor Yellow
Write-Host "   - Clicca 'Create Background Worker'" -ForegroundColor White
Write-Host "   - Aspetta che finisca il deploy" -ForegroundColor White
Write-Host ""

Write-Host "📚 Guida completa: GUIDA_DEPLOY_SERVER_H24.md" -ForegroundColor Cyan
Write-Host ""

Write-Host "✅ Setup completato!" -ForegroundColor Green
Write-Host ""

# Verifica variabili ambiente necessarie
Write-Host "🔐 Variabili ambiente da configurare su Render:" -ForegroundColor Yellow
Write-Host ""
Write-Host "TELEGRAM_BOT_TOKEN" -ForegroundColor Cyan
Write-Host "  Come ottenerlo:" -ForegroundColor White
Write-Host "    1. Vai su Telegram → @BotFather" -ForegroundColor Gray
Write-Host "    2. /newbot → Segui istruzioni" -ForegroundColor Gray
Write-Host "    3. Copia il token" -ForegroundColor Gray
Write-Host ""

Write-Host "TELEGRAM_CHAT_ID" -ForegroundColor Cyan
Write-Host "  Come ottenerlo:" -ForegroundColor White
Write-Host "    1. Vai su Telegram → @userinfobot" -ForegroundColor Gray
Write-Host "    2. /start" -ForegroundColor Gray
Write-Host "    3. Copia il numero ID" -ForegroundColor Gray
Write-Host ""

Write-Host "💡 Suggerimento: Leggi GUIDA_DEPLOY_SERVER_H24.md per istruzioni dettagliate!" -ForegroundColor Cyan
Write-Host ""










