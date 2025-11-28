# Script PowerShell per fare commit e push del Market Analyzer su GitHub
# Usage: .\PUSH_MARKET_ANALYZER.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MARKET ANALYZER - PUSH SU GITHUB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica di essere nella directory corretta
if (-not (Test-Path "market_movement_analyzer_app.py")) {
    Write-Host "❌ ERRORE: File market_movement_analyzer_app.py non trovato!" -ForegroundColor Red
    Write-Host "Assicurati di eseguire lo script dalla directory Software-AsianOdds-main" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ File trovato: market_movement_analyzer_app.py" -ForegroundColor Green
Write-Host ""

# Verifica stato git
Write-Host "📊 Controllo stato repository..." -ForegroundColor Yellow
$status = git status --porcelain

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERRORE: Directory non è un repository git!" -ForegroundColor Red
    exit 1
}

# Mostra file modificati/nuovi
$newFiles = @(
    "market_movement_analyzer_app.py",
    ".streamlit/config.toml",
    "MARKET_ANALYZER_DEPLOY.md",
    "README_MARKET_ANALYZER.md",
    "AVVIA_MARKET_ANALYZER.bat"
)

Write-Host "📝 File da aggiungere:" -ForegroundColor Yellow
foreach ($file in $newFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (non trovato)" -ForegroundColor Gray
    }
}
Write-Host ""

# Chiedi conferma
$confirm = Read-Host "Vuoi procedere con il commit e push? (s/n)"
if ($confirm -notmatch "^[sS]") {
    Write-Host "❌ Operazione annullata" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "🔄 Aggiungo file a git..." -ForegroundColor Yellow

# Aggiungi i file
foreach ($file in $newFiles) {
    if (Test-Path $file) {
        git add $file
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✓ Aggiunto: $file" -ForegroundColor Green
        } else {
            Write-Host "   ✗ Errore aggiungendo: $file" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "💾 Creo commit..." -ForegroundColor Yellow
$commitMessage = "Add Market Movement Analyzer Streamlit app

- Add market_movement_analyzer_app.py (Streamlit web app)
- Add .streamlit/config.toml (Streamlit configuration)
- Add deployment guide and README
- Add Windows batch launcher"

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit creato con successo!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Nessun cambiamento da committare o errore nel commit" -ForegroundColor Yellow
    Write-Host "   Verifica lo stato con: git status" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Eseguo push su GitHub..." -ForegroundColor Yellow

# Push
$branch = git rev-parse --abbrev-ref HEAD
Write-Host "   Branch: $branch" -ForegroundColor Gray

git push origin $branch

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ SUCCESSO! Push completato!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Prossimi step:" -ForegroundColor Yellow
    Write-Host "1. Vai su https://share.streamlit.io/" -ForegroundColor White
    Write-Host "2. Clicca 'New app'" -ForegroundColor White
    Write-Host "3. Seleziona repository: Sib-asian/Software-AsianOdds" -ForegroundColor White
    Write-Host "4. Main file: market_movement_analyzer_app.py" -ForegroundColor White
    Write-Host "5. Clicca 'Deploy!'" -ForegroundColor White
    Write-Host ""
    Write-Host "Leggi la guida completa in: MARKET_ANALYZER_DEPLOY.md" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ ERRORE durante il push!" -ForegroundColor Red
    Write-Host "   Verifica la connessione e i permessi GitHub" -ForegroundColor Yellow
}

Write-Host ""

