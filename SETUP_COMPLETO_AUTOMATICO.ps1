#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup Completo Automatico per Deploy su Render.com
    
.DESCRIPTION
    Questo script:
    1. Verifica/Installa Git (se necessario)
    2. Inizializza repository Git
    3. Prepara tutto per push su GitHub
    4. Verifica che tutti i file necessari siano presenti
#>

param(
    [string]$GitHubUsername = "",
    [string]$RepositoryName = "Software-AsianOdds"
)

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   SETUP COMPLETO AUTOMATICO - RENDER.COM DEPLOY         ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================
# STEP 1: Verifica File Necessari
# ============================================
Write-Host "[STEP 1/5] Verifica file necessari..." -ForegroundColor Yellow
Write-Host ""

$requiredFiles = @(
    "Dockerfile.automation",
    "start_automation.py",
    "automation_24h.py",
    "requirements.automation.txt",
    "api_manager.py"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file - MANCANTE!" -ForegroundColor Red
        $missingFiles += $file
    }
}

if (Test-Path "ai_system") {
    Write-Host "  ✅ ai_system/ (cartella)" -ForegroundColor Green
} else {
    Write-Host "  ❌ ai_system/ - MANCANTE!" -ForegroundColor Red
    $missingFiles += "ai_system/"
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ ERRORE: File mancanti!" -ForegroundColor Red
    Write-Host "File mancanti: $($missingFiles -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Tutti i file necessari sono presenti!" -ForegroundColor Green
Write-Host ""

# ============================================
# STEP 2: Verifica Git
# ============================================
Write-Host "[STEP 2/5] Verifica Git..." -ForegroundColor Yellow
Write-Host ""

$gitInstalled = $false
try {
    $gitVersion = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Git installato: $gitVersion" -ForegroundColor Green
        $gitInstalled = $true
    }
} catch {
    $gitInstalled = $false
}

if (-not $gitInstalled) {
    Write-Host "  ❌ Git NON installato" -ForegroundColor Red
    Write-Host ""
    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  GIT NON È INSTALLATO" -ForegroundColor Yellow
    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opzioni disponibili:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. INSTALLA GIT (Consigliato):" -ForegroundColor White
    Write-Host "   - Vai su: https://git-scm.com/download/win" -ForegroundColor Gray
    Write-Host "   - Scarica e installa Git" -ForegroundColor Gray
    Write-Host "   - Riavvia PowerShell" -ForegroundColor Gray
    Write-Host "   - Esegui di nuovo questo script" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. USA GITHUB DESKTOP (Piu facile):" -ForegroundColor White
    Write-Host "   - Vai su: https://desktop.github.com/" -ForegroundColor Gray
    Write-Host "   - Scarica e installa GitHub Desktop" -ForegroundColor Gray
    Write-Host "   - File -> Add Local Repository" -ForegroundColor Gray
    Write-Host "   - Seleziona questa cartella" -ForegroundColor Gray
    Write-Host "   - Publish repository" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. UPLOAD MANUALE (Senza installare nulla):" -ForegroundColor White
    Write-Host "   - Vai su: https://github.com/new" -ForegroundColor Gray
    Write-Host "   - Crea nuovo repository" -ForegroundColor Gray
    Write-Host "   - Clicca 'uploading an existing file'" -ForegroundColor Gray
    Write-Host "   - Trascina tutti i file di questa cartella" -ForegroundColor Gray
    Write-Host ""
    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📄 Vedi guida completa: INSTALLA_GIT_E_SETUP.md" -ForegroundColor Cyan
    Write-Host ""
    
    # Crea file con istruzioni
    $instructions = @"
# Git non installato - Istruzioni

## Opzione 1: Installa Git (CLI)
1. Vai su: https://git-scm.com/download/win
2. Scarica e installa
3. Riavvia PowerShell
4. Esegui: .\SETUP_COMPLETO_AUTOMATICO.ps1

## Opzione 2: GitHub Desktop (Piu facile)
1. Vai su: https://desktop.github.com/
2. Installa GitHub Desktop
3. File -> Add Local Repository
4. Seleziona questa cartella
5. Publish repository

## Opzione 3: Upload Manuale
1. Vai su: https://github.com/new
2. Crea repository
3. Upload file manualmente

Vedi: INSTALLA_GIT_E_SETUP.md per dettagli completi.
"@
    $instructions | Out-File -FilePath "ISTRUZIONI_GIT.txt" -Encoding UTF8
    
    Write-Host "✅ Creato file: ISTRUZIONI_GIT.txt" -ForegroundColor Green
    Write-Host ""
    exit 0
}

# ============================================
# STEP 3: Configura Git (se necessario)
# ============================================
Write-Host "[STEP 3/5] Configurazione Git..." -ForegroundColor Yellow
Write-Host ""

$userName = git config --global user.name 2>&1
$userEmail = git config --global user.email 2>&1

if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Host "  ⚠️  Configurazione Git non trovata" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Configura Git con:" -ForegroundColor Cyan
    Write-Host "    git config --global user.name `"Tuo Nome`"" -ForegroundColor White
    Write-Host "    git config --global user.email `"tua.email@example.com`"" -ForegroundColor White
    Write-Host ""
    Write-Host "  Oppure configura manualmente dopo." -ForegroundColor Gray
} else {
    Write-Host "  ✅ Nome: $userName" -ForegroundColor Green
    Write-Host "  ✅ Email: $userEmail" -ForegroundColor Green
}

Write-Host ""

# ============================================
# STEP 4: Inizializza Repository Git
# ============================================
Write-Host "[STEP 4/5] Inizializzazione repository Git..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path .git) {
    Write-Host "  ✅ Repository Git già inizializzato" -ForegroundColor Green
    
    # Verifica remote
    $remote = git remote get-url origin 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Remote origin: $remote" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Remote origin non configurato" -ForegroundColor Yellow
        if (-not [string]::IsNullOrWhiteSpace($GitHubUsername)) {
            $remoteUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"
            Write-Host "  📝 Aggiungo remote: $remoteUrl" -ForegroundColor Cyan
            git remote add origin $remoteUrl 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ Remote aggiunto!" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "  📝 Inizializzo nuovo repository Git..." -ForegroundColor Cyan
    git init | Out-Null
    Write-Host "  ✅ Repository inizializzato" -ForegroundColor Green
    
    if (-not [string]::IsNullOrWhiteSpace($GitHubUsername)) {
        $remoteUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"
        Write-Host "  📝 Aggiungo remote: $remoteUrl" -ForegroundColor Cyan
        git remote add origin $remoteUrl 2>&1 | Out-Null
        Write-Host "  ✅ Remote configurato" -ForegroundColor Green
    }
}

# Verifica branch
$currentBranch = git branch --show-current 2>&1
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    git checkout -b main 2>&1 | Out-Null
    Write-Host "  ✅ Branch 'main' creato" -ForegroundColor Green
} else {
    Write-Host "  ✅ Branch corrente: $currentBranch" -ForegroundColor Green
}

Write-Host ""

# ============================================
# STEP 5: Prepara Commit
# ============================================
Write-Host "[STEP 5/5] Preparazione commit..." -ForegroundColor Yellow
Write-Host ""

# Aggiungi tutti i file
Write-Host "  📝 Aggiungo file al repository..." -ForegroundColor Cyan
git add . 2>&1 | Out-Null

# Verifica se ci sono modifiche
$status = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($status)) {
    Write-Host "  ✅ File pronti per commit" -ForegroundColor Green
    
    # Crea commit
    Write-Host "  📝 Creo commit..." -ForegroundColor Cyan
    git commit -m "Initial commit - Ready for Render deployment" 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Commit creato!" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Errore durante commit (potrebbe essere già committato)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✅ Nessuna modifica da committare (già tutto committato)" -ForegroundColor Green
}

Write-Host ""

# ============================================
# RIEPILOGO
# ============================================
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✅ SETUP COMPLETATO!                        ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "📋 Prossimi passi:" -ForegroundColor Cyan
Write-Host ""

# Verifica se remote è configurato
$remote = git remote get-url origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "1. ✅ Repository Git configurato" -ForegroundColor Green
    Write-Host "2. Push su GitHub:" -ForegroundColor Yellow
    Write-Host "   git push -u origin main" -ForegroundColor White
    Write-Host ""
    Write-Host "   Se il repository non esiste ancora su GitHub:" -ForegroundColor Gray
    Write-Host "   - Vai su: https://github.com/new" -ForegroundColor Gray
    Write-Host "   - Crea repository: $RepositoryName" -ForegroundColor Gray
    Write-Host "   - Poi esegui: git push -u origin main" -ForegroundColor Gray
} else {
    Write-Host "1. ✅ Repository Git inizializzato" -ForegroundColor Green
    Write-Host "2. Crea repository su GitHub:" -ForegroundColor Yellow
    Write-Host "   - Vai su: https://github.com/new" -ForegroundColor White
    Write-Host "   - Nome: $RepositoryName" -ForegroundColor White
    Write-Host "   - Crea repository (NON inizializzare con README)" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Collega e push:" -ForegroundColor Yellow
    if (-not [string]::IsNullOrWhiteSpace($GitHubUsername)) {
        Write-Host "   git remote add origin https://github.com/$GitHubUsername/$RepositoryName.git" -ForegroundColor White
    } else {
        Write-Host "   git remote add origin https://github.com/TUO-USERNAME/$RepositoryName.git" -ForegroundColor White
    }
    Write-Host "   git branch -M main" -ForegroundColor White
    Write-Host "   git push -u origin main" -ForegroundColor White
}

Write-Host ""
Write-Host "4. Procedi con Step 2: Deploy su Render.com" -ForegroundColor Yellow
Write-Host "   - Vedi: GUIDA_RENDER_PASSO_PASSO.md" -ForegroundColor White
Write-Host ""

Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

