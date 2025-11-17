#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup automatico completo per repository Git e sistema PR
    
.DESCRIPTION
    Questo script automatizza tutto il setup necessario:
    1. Verifica/installa Git
    2. Configura repository
    3. Prepara tutto per creare PR
#>

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SETUP AUTOMATICO REPOSITORY GIT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Funzione per verificare Git
function Test-GitInstalled {
    try {
        $null = git --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# 1. Verifica Git
Write-Host "[1/6] Verifica Git..." -ForegroundColor Yellow
if (Test-GitInstalled) {
    $gitVersion = git --version
    Write-Host "  [OK] Git trovato: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "  [ATTENZIONE] Git non trovato" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Git deve essere installato per continuare." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Opzioni:" -ForegroundColor Cyan
    Write-Host "  1. Installa manualmente: https://git-scm.com/download/win" -ForegroundColor White
    Write-Host "  2. Oppure usa winget (se disponibile):" -ForegroundColor White
    Write-Host "     winget install --id Git.Git -e --source winget" -ForegroundColor Gray
    Write-Host ""
    
    # Prova a installare con winget
    $wingetAvailable = $false
    try {
        $null = winget --version 2>&1
        $wingetAvailable = $true
    } catch {
        $wingetAvailable = $false
    }
    
    if ($wingetAvailable) {
        Write-Host "  Rilevato winget. Vuoi installare Git automaticamente? (S/N): " -NoNewline -ForegroundColor Cyan
        $response = Read-Host
        if ($response -eq "S" -or $response -eq "s" -or $response -eq "Y" -or $response -eq "y") {
            Write-Host "  Installazione Git in corso..." -ForegroundColor Yellow
            winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
            Write-Host "  [OK] Git installato. Riavvia PowerShell e riesegui questo script." -ForegroundColor Green
            exit 0
        }
    }
    
    Write-Host "  Dopo aver installato Git, riavvia PowerShell e riesegui questo script." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# 2. Verifica configurazione Git
Write-Host ""
Write-Host "[2/6] Verifica configurazione Git..." -ForegroundColor Yellow
$userName = git config --global user.name 2>&1
$userEmail = git config --global user.email 2>&1

if ([string]::IsNullOrWhiteSpace($userName) -or $userName -match "error") {
    Write-Host "  [ATTENZIONE] Nome utente Git non configurato" -ForegroundColor Yellow
    $newName = Read-Host "  Inserisci il tuo nome"
    if (-not [string]::IsNullOrWhiteSpace($newName)) {
        git config --global user.name $newName
        Write-Host "  [OK] Nome configurato: $newName" -ForegroundColor Green
    }
} else {
    Write-Host "  [OK] Nome: $userName" -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($userEmail) -or $userEmail -match "error") {
    Write-Host "  [ATTENZIONE] Email Git non configurata" -ForegroundColor Yellow
    $newEmail = Read-Host "  Inserisci la tua email"
    if (-not [string]::IsNullOrWhiteSpace($newEmail)) {
        git config --global user.email $newEmail
        Write-Host "  [OK] Email configurata: $newEmail" -ForegroundColor Green
    }
} else {
    Write-Host "  [OK] Email: $userEmail" -ForegroundColor Green
}

# 3. Inizializza repository
Write-Host ""
Write-Host "[3/6] Inizializzazione repository Git..." -ForegroundColor Yellow
if (Test-Path .git) {
    Write-Host "  [OK] Repository Git gia' esistente" -ForegroundColor Green
} else {
    git init
    Write-Host "  [OK] Repository inizializzato" -ForegroundColor Green
}

# 4. Configura remote
Write-Host ""
Write-Host "[4/6] Configurazione remote GitHub..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/Sib-asian/Software-AsianOdds.git"
$existingRemote = git remote get-url origin 2>&1

if ($LASTEXITCODE -ne 0) {
    git remote add origin $remoteUrl
    Write-Host "  [OK] Remote origin aggiunto: $remoteUrl" -ForegroundColor Green
} else {
    if ($existingRemote -ne $remoteUrl) {
        git remote set-url origin $remoteUrl
        Write-Host "  [OK] Remote origin aggiornato: $remoteUrl" -ForegroundColor Green
    } else {
        Write-Host "  [OK] Remote origin gia' configurato correttamente" -ForegroundColor Green
    }
}

# 5. Crea .gitignore se non esiste
Write-Host ""
Write-Host "[5/6] Verifica .gitignore..." -ForegroundColor Yellow
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

# 6. Crea branch main se non esiste
Write-Host ""
Write-Host "[6/6] Verifica branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current 2>&1
if ([string]::IsNullOrWhiteSpace($currentBranch) -or $LASTEXITCODE -ne 0) {
    git checkout -b main 2>&1 | Out-Null
    Write-Host "  [OK] Branch 'main' creato" -ForegroundColor Green
} else {
    Write-Host "  [OK] Branch corrente: $currentBranch" -ForegroundColor Green
}

# Verifica se ci sono file da committare
Write-Host ""
Write-Host "Verifica file da committare..." -ForegroundColor Yellow
$status = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($status)) {
    Write-Host "  Trovati file non committati" -ForegroundColor Cyan
    $response = Read-Host "  Vuoi fare il commit iniziale? (S/N)"
    if ($response -eq "S" -or $response -eq "s" -or $response -eq "Y" -or $response -eq "y") {
        git add .
        git commit -m "Initial commit: Setup repository and PR automation"
        Write-Host "  [OK] Commit iniziale creato" -ForegroundColor Green
    }
}

# Riepilogo
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETATO CON SUCCESSO!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Cyan
Write-Host "1. Configura autenticazione GitHub:" -ForegroundColor White
Write-Host "   - Personal Access Token: https://github.com/settings/tokens" -ForegroundColor Gray
Write-Host "   - Oppure GitHub CLI: gh auth login" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Usa gli script per creare PR:" -ForegroundColor White
Write-Host "   .\quick_pr.ps1 -Message `"Descrizione modifiche`"" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Leggi la documentazione:" -ForegroundColor White
Write-Host "   - README_AUTO_PR.md (guida rapida)" -ForegroundColor Gray
Write-Host "   - ISTRUZIONI_SETUP.md (setup dettagliato)" -ForegroundColor Gray
Write-Host ""

