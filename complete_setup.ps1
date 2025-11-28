#!/usr/bin/env pwsh
# Setup completo non interattivo

$ErrorActionPreference = "Continue"

# Aggiorna PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host ""
Write-Host "=== SETUP REPOSITORY GIT ===" -ForegroundColor Cyan
Write-Host ""

# 1. Verifica Git
Write-Host "[1/7] Verifica Git..." -ForegroundColor Yellow
try {
    $gitVersion = git --version 2>&1
    Write-Host "  [OK] $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERRORE] Git non trovato" -ForegroundColor Red
    exit 1
}

# 2. Configura Git (usa valori di default se non configurati)
Write-Host ""
Write-Host "[2/7] Configurazione Git..." -ForegroundColor Yellow
$userName = git config --global user.name 2>&1
$userEmail = git config --global user.email 2>&1

if ([string]::IsNullOrWhiteSpace($userName) -or $userName -match "error") {
    git config --global user.name "Software-AsianOdds User"
    Write-Host "  [OK] Nome configurato: Software-AsianOdds User" -ForegroundColor Green
    Write-Host "  [INFO] Puoi cambiarlo con: git config --global user.name 'Tuo Nome'" -ForegroundColor Gray
} else {
    Write-Host "  [OK] Nome: $userName" -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($userEmail) -or $userEmail -match "error") {
    git config --global user.email "user@asianodds.local"
    Write-Host "  [OK] Email configurata: user@asianodds.local" -ForegroundColor Green
    Write-Host "  [INFO] Puoi cambiarla con: git config --global user.email 'tua@email.com'" -ForegroundColor Gray
} else {
    Write-Host "  [OK] Email: $userEmail" -ForegroundColor Green
}

# 3. Inizializza repository
Write-Host ""
Write-Host "[3/7] Inizializzazione repository..." -ForegroundColor Yellow
if (Test-Path .git) {
    Write-Host "  [OK] Repository gia' esistente" -ForegroundColor Green
} else {
    git init | Out-Null
    Write-Host "  [OK] Repository inizializzato" -ForegroundColor Green
}

# 4. Configura remote
Write-Host ""
Write-Host "[4/7] Configurazione remote..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/Sib-asian/Software-AsianOdds.git"
$existingRemote = git remote get-url origin 2>&1

if ($LASTEXITCODE -ne 0) {
    git remote add origin $remoteUrl 2>&1 | Out-Null
    Write-Host "  [OK] Remote origin aggiunto" -ForegroundColor Green
} else {
    if ($existingRemote -ne $remoteUrl) {
        git remote set-url origin $remoteUrl 2>&1 | Out-Null
        Write-Host "  [OK] Remote origin aggiornato" -ForegroundColor Green
    } else {
        Write-Host "  [OK] Remote origin gia' configurato" -ForegroundColor Green
    }
}

# 5. Crea .gitignore
Write-Host ""
Write-Host "[5/7] Verifica .gitignore..." -ForegroundColor Yellow
if (-not (Test-Path .gitignore)) {
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
    Write-Host "  [OK] .gitignore creato" -ForegroundColor Green
} else {
    Write-Host "  [OK] .gitignore gia' esistente" -ForegroundColor Green
}

# 6. Crea branch main
Write-Host ""
Write-Host "[6/7] Verifica branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current 2>&1
if ([string]::IsNullOrWhiteSpace($currentBranch) -or $LASTEXITCODE -ne 0) {
    git checkout -b main 2>&1 | Out-Null
    Write-Host "  [OK] Branch 'main' creato" -ForegroundColor Green
} else {
    Write-Host "  [OK] Branch corrente: $currentBranch" -ForegroundColor Green
}

# 7. Commit iniziale (se ci sono file)
Write-Host ""
Write-Host "[7/7] Verifica file da committare..." -ForegroundColor Yellow
$status = git status --porcelain 2>&1
if (-not [string]::IsNullOrWhiteSpace($status) -and $status -notmatch "error") {
    git add . 2>&1 | Out-Null
    git commit -m "Setup: Repository initialization and PR automation scripts" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Commit iniziale creato" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] Nessun file da committare o commit gia' fatto" -ForegroundColor Gray
    }
} else {
    Write-Host "  [INFO] Nessun file da committare" -ForegroundColor Gray
}

# Riepilogo
Write-Host ""
Write-Host "=== SETUP COMPLETATO! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Repository Git configurato correttamente." -ForegroundColor Cyan
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Yellow
Write-Host "1. Configura autenticazione GitHub:" -ForegroundColor White
Write-Host "   - Token: https://github.com/settings/tokens" -ForegroundColor Gray
Write-Host "   - Oppure: gh auth login" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Crea PR dopo modifiche:" -ForegroundColor White
Write-Host "   .\quick_pr.ps1 -Message 'Descrizione modifiche'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Leggi documentazione:" -ForegroundColor White
Write-Host "   - README_AUTO_PR.md" -ForegroundColor Gray
Write-Host ""






