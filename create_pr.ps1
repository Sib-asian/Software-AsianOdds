#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script per creare automaticamente un Pull Request su GitHub dopo modifiche al codice
    
.DESCRIPTION
    Questo script:
    1. Verifica che Git sia installato
    2. Crea un nuovo branch
    3. Fa commit delle modifiche
    4. Pusha il branch
    5. Crea un PR su GitHub (se GitHub CLI è installato)
    
.PARAMETER BranchName
    Nome del branch da creare (default: auto-generato da timestamp)
    
.PARAMETER CommitMessage
    Messaggio del commit (default: "Update code")
    
.PARAMETER PRTitle
    Titolo del PR (default: stesso del commit message)
    
.PARAMETER PRDescription
    Descrizione del PR
    
.PARAMETER BaseBranch
    Branch base per il PR (default: "main")
    
.EXAMPLE
    .\create_pr.ps1 -CommitMessage "Fix bug in AI pipeline" -PRDescription "Risolto problema nel calcolo delle probabilità"
#>

param(
    [string]$BranchName = "",
    [string]$CommitMessage = "Update code",
    [string]$PRTitle = "",
    [string]$PRDescription = "",
    [string]$BaseBranch = "main"
)

# Colori per output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "🚀 Script Creazione PR GitHub"
Write-Output ""

# 1. Verifica Git installato
Write-Output "📋 Verifica Git..."
try {
    $gitVersion = git --version
    Write-ColorOutput Green "✅ Git trovato: $gitVersion"
} catch {
    Write-ColorOutput Red "❌ Git non trovato! Installa Git da https://git-scm.com/"
    exit 1
}

# 2. Verifica che siamo in un repository Git
Write-Output ""
Write-Output "📋 Verifica repository Git..."
if (-not (Test-Path .git)) {
    Write-ColorOutput Yellow "⚠️  Directory corrente non è un repository Git"
    Write-Output ""
    Write-Output "Esegui prima lo script di setup:" -ForegroundColor Cyan
    Write-Output "  .\setup_git_repo.ps1" -ForegroundColor White
    Write-Output ""
    exit 1
}

# 3. Verifica stato repository
Write-Output ""
Write-Output "📋 Verifica stato repository..."
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-ColorOutput Yellow "⚠️  Nessuna modifica da committare!"
    Write-Output "File modificati:"
    git status
    exit 0
}

Write-ColorOutput Green "✅ Modifiche trovate:"
git status --short

# 4. Genera nome branch se non specificato
if ([string]::IsNullOrWhiteSpace($BranchName)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BranchName = "update-$timestamp"
}
Write-Output ""
Write-ColorOutput Cyan "🌿 Branch: $BranchName"

# 5. Crea e checkout branch
Write-Output ""
Write-Output "📋 Creazione branch..."
try {
    git checkout -b $BranchName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        # Branch potrebbe esistere già, prova checkout
        git checkout $BranchName
    }
    Write-ColorOutput Green "✅ Branch '$BranchName' creato/selezionato"
} catch {
    Write-ColorOutput Red "❌ Errore creazione branch: $_"
    exit 1
}

# 6. Aggiungi tutti i file modificati
Write-Output ""
Write-Output "📋 Aggiunta file modificati..."
git add .
Write-ColorOutput Green "✅ File aggiunti allo staging"

# 7. Commit
Write-Output ""
Write-Output "📋 Creazione commit..."
try {
    git commit -m $CommitMessage
    Write-ColorOutput Green "✅ Commit creato: $CommitMessage"
} catch {
    Write-ColorOutput Red "❌ Errore commit: $_"
    exit 1
}

# 8. Push branch
Write-Output ""
Write-Output "📋 Push branch su GitHub..."
try {
    git push -u origin $BranchName
    Write-ColorOutput Green "✅ Branch pushato su GitHub"
} catch {
    Write-ColorOutput Yellow "⚠️  Errore push (potrebbe richiedere autenticazione): $_"
    Write-Output ""
    Write-Output "Esegui manualmente:"
    Write-ColorOutput Cyan "  git push -u origin $BranchName"
    Write-Output ""
}

# 9. Crea PR con GitHub CLI (se disponibile)
Write-Output ""
Write-Output "📋 Verifica GitHub CLI..."
try {
    $ghVersion = gh --version 2>&1
    Write-ColorOutput Green "✅ GitHub CLI trovato"
    
    if ([string]::IsNullOrWhiteSpace($PRTitle)) {
        $PRTitle = $CommitMessage
    }
    
    Write-Output ""
    Write-Output "📋 Creazione Pull Request..."
    $prArgs = @(
        "pr", "create",
        "--title", $PRTitle,
        "--base", $BaseBranch,
        "--head", $BranchName
    )
    
    if (-not [string]::IsNullOrWhiteSpace($PRDescription)) {
        $prArgs += "--body"
        $prArgs += $PRDescription
    }
    
    $prUrl = gh $prArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Pull Request creata!"
        Write-ColorOutput Cyan "🔗 $prUrl"
    } else {
        Write-ColorOutput Yellow "⚠️  Errore creazione PR automatica"
        Write-Output "Crea manualmente su: https://github.com/Sib-asian/Software-AsianOdds/compare/$BaseBranch...$BranchName"
    }
} catch {
    Write-ColorOutput Yellow "⚠️  GitHub CLI non trovato"
    Write-Output ""
    Write-Output "Opzioni per creare il PR:"
    Write-Output "1. Installa GitHub CLI: https://cli.github.com/"
    Write-Output "2. Crea manualmente su GitHub:"
    Write-ColorOutput Cyan "   https://github.com/Sib-asian/Software-AsianOdds/compare/$BaseBranch...$BranchName"
    Write-Output ""
    Write-Output "Oppure usa questo comando (se GitHub CLI è installato):"
    Write-ColorOutput Cyan "   gh pr create --title `"$PRTitle`" --base $BaseBranch --head $BranchName"
}

Write-Output ""
Write-ColorOutput Green "✨ Processo completato!"
Write-Output ""
Write-Output "Riepilogo:"
Write-Output "  Branch: $BranchName"
Write-Output "  Commit: $CommitMessage"
Write-Output "  Base: $BaseBranch"

