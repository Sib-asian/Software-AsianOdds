#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup iniziale repository Git per Software-AsianOdds
    
.DESCRIPTION
    Questo script inizializza il repository Git se non esiste già,
    configura il remote e prepara tutto per creare PR.
#>

param(
    [string]$RemoteUrl = "https://github.com/Sib-asian/Software-AsianOdds.git"
)

Write-Host "=== Setup Repository Git ===" -ForegroundColor Cyan
Write-Host ""

# Verifica Git
Write-Host "[1/5] Verifica Git..." -NoNewline
try {
    $gitVersion = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " [OK]" -ForegroundColor Green
        Write-Host "   $gitVersion" -ForegroundColor Gray
    } else {
        throw "Git non trovato"
    }
} catch {
    Write-Host " [ERRORE]" -ForegroundColor Red
    Write-Host ""
    Write-Host "ATTENZIONE: Git non e' installato o non e' nel PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opzioni:" -ForegroundColor Cyan
    Write-Host "1. Installa Git: https://git-scm.com/download/win" -ForegroundColor White
    Write-Host "2. Aggiungi Git al PATH dopo l'installazione" -ForegroundColor White
    Write-Host ""
    Write-Host "Dopo l'installazione, esegui di nuovo questo script." -ForegroundColor Yellow
    exit 1
}

# Verifica se siamo già in un repo Git
Write-Host ""
Write-Host "[2/5] Verifica repository Git..." -NoNewline
if (Test-Path .git) {
    Write-Host " [OK]" -ForegroundColor Green
    Write-Host "   Repository Git gia' inizializzato" -ForegroundColor Gray
    
    # Verifica remote
    $remote = git remote get-url origin 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Remote origin: $remote" -ForegroundColor Gray
    } else {
        Write-Host "   Aggiungo remote origin..." -ForegroundColor Yellow
        git remote add origin $RemoteUrl
        Write-Host "   [OK] Remote aggiunto" -ForegroundColor Green
    }
} else {
    Write-Host " [NUOVO]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[3/5] Inizializzazione repository Git..." -ForegroundColor Cyan
    
    # Inizializza repo
    git init
    Write-Host "[OK] Repository inizializzato" -ForegroundColor Green
    
    # Aggiungi remote
    git remote add origin $RemoteUrl
    Write-Host "[OK] Remote origin configurato" -ForegroundColor Green
    
    # Crea .gitignore se non esiste
    if (-not (Test-Path .gitignore)) {
        Write-Host "[4/5] Creazione .gitignore..." -ForegroundColor Cyan
        $gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Streamlit
.streamlit/

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
desktop.ini

# Data files
*.csv
*.xlsx
*.json
!requirements.txt
!package.json
!team_profiles.json
!meta_playbook.example.json

# Models
*.pth
*.pkl
*.h5
*.model

# Cache
.cache/
*.cache

# Temporary
tmp/
temp/
*.tmp
"@
        $gitignoreContent | Out-File -FilePath .gitignore -Encoding UTF8 -NoNewline
        Write-Host "[OK] .gitignore creato" -ForegroundColor Green
    }
    
    # Prima commit se ci sono file
    $status = git status --porcelain
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        Write-Host ""
        Write-Host "[5/5] Prima commit..." -ForegroundColor Cyan
        git add .
        git commit -m "Initial commit: Setup repository"
        Write-Host "[OK] Prima commit creata" -ForegroundColor Green
    }
}

# Verifica branch
Write-Host ""
Write-Host "Verifica branch..." -ForegroundColor Cyan
$currentBranch = git branch --show-current 2>&1
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    # Crea branch main se non esiste
    git checkout -b main
    Write-Host "[OK] Branch 'main' creato" -ForegroundColor Green
} else {
    Write-Host "   Branch corrente: $currentBranch" -ForegroundColor Gray
}

# Verifica configurazione utente
Write-Host ""
Write-Host "Verifica configurazione Git..." -ForegroundColor Cyan
$userName = git config user.name
$userEmail = git config user.email

if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Host "ATTENZIONE: Configurazione utente Git non trovata" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Configura Git con:" -ForegroundColor Cyan
    Write-Host "  git config --global user.name `"Tuo Nome`"" -ForegroundColor White
    Write-Host "  git config --global user.email `"tua.email@example.com`"" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "   Nome: $userName" -ForegroundColor Gray
    Write-Host "   Email: $userEmail" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup completato! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Cyan
Write-Host "1. Configura autenticazione GitHub (se non gia' fatto)" -ForegroundColor White
Write-Host "2. Usa .\create_pr.ps1 per creare PR dopo modifiche" -ForegroundColor White
Write-Host ""
