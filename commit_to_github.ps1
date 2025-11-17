# Script PowerShell per commit e push su GitHub
# Verifica se git è disponibile e fa commit automatico

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COMMIT E PUSH SU GITHUB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Cerca git
$gitExe = $null
$possiblePaths = @(
    "git",
    "C:\Program Files\Git\bin\git.exe",
    "C:\Program Files (x86)\Git\bin\git.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\git.exe"
)

foreach ($path in $possiblePaths) {
    try {
        $result = & $path --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $gitExe = $path
            Write-Host "✅ Git trovato: $path" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $gitExe) {
    Write-Host "❌ Git non trovato!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Opzioni:" -ForegroundColor Yellow
    Write-Host "1. Installa Git: https://git-scm.com/download/win" -ForegroundColor Gray
    Write-Host "2. Usa GitHub Desktop: https://desktop.github.com/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Oppure esegui manualmente:" -ForegroundColor Yellow
    Write-Host "  git add ." -ForegroundColor Gray
    Write-Host "  git commit -m 'feat: Sistema automazione 24/7 completo'" -ForegroundColor Gray
    Write-Host "  git push origin main" -ForegroundColor Gray
    exit 1
}

# Verifica che .env non venga committato
Write-Host ""
Write-Host "🔍 Verifica sicurezza..." -ForegroundColor Cyan
if (Test-Path ".env") {
    $envInGit = & $gitExe check-ignore ".env" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ .env è nel .gitignore (sicuro)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  ATTENZIONE: .env NON è nel .gitignore!" -ForegroundColor Red
        Write-Host "   Aggiungilo al .gitignore prima di committare!" -ForegroundColor Yellow
        Read-Host "Premi Enter per continuare comunque (NON CONSIGLIATO)"
    }
}

# Mostra stato
Write-Host ""
Write-Host "📋 Stato repository:" -ForegroundColor Cyan
& $gitExe status --short

Write-Host ""
$confirm = Read-Host "Vuoi procedere con commit e push? (s/n)"
if ($confirm -ne "s" -and $confirm -ne "S") {
    Write-Host "Operazione annullata" -ForegroundColor Yellow
    exit 0
}

# Aggiungi tutti i file
Write-Host ""
Write-Host "📦 Aggiunta file..." -ForegroundColor Cyan
& $gitExe add .

# Crea commit
Write-Host "💾 Creazione commit..." -ForegroundColor Cyan
$commitMessage = @"
feat: Sistema automazione 24/7 completo con accesso remoto

✨ Nuove funzionalità:
- Automazione 24/7 con servizio Windows auto-restart
- Fetch partite reali da TheOddsAPI
- Accesso remoto dashboard Streamlit da cellulare
- Script di installazione e gestione servizio

🔧 Miglioramenti:
- Corretto errore bankroll argument in automation_24h.py
- Migliorata gestione errori e logging
- Aggiunta documentazione completa

📚 Documentazione:
- Guida completa automazione 24/7
- Istruzioni accesso remoto
- Configurazione chiavi API
- Spiegazione funzionamento sistema

🛠️ Script:
- install_service_auto.ps1 - Installazione servizio
- start_dashboard_remote.bat - Dashboard remoto
- manage_service.bat - Gestione servizio
- verifica_tutte_chiavi.py - Verifica configurazione
"@

& $gitExe commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit creato con successo!" -ForegroundColor Green
} else {
    Write-Host "❌ Errore durante commit" -ForegroundColor Red
    exit 1
}

# Push
Write-Host ""
Write-Host "🚀 Push su GitHub..." -ForegroundColor Cyan
& $gitExe push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Push completato con successo!" -ForegroundColor Green
    Write-Host "🎉 Tutti i cambiamenti sono su GitHub!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Errore durante push" -ForegroundColor Red
    Write-Host "   Verifica connessione e credenziali GitHub" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "✅ Operazione completata!" -ForegroundColor Green

