# Monitoraggio semplificato per 6 cicli
$maxCycles = 6
$cycleMinutes = 5
$totalMinutes = $maxCycles * $cycleMinutes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📊 MONITORAGGIO 6 CICLI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "⏰ Inizio: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
Write-Host "⏱️  Durata totale: $totalMinutes minuti" -ForegroundColor Yellow
Write-Host "🔄 Cicli: $maxCycles (ogni $cycleMinutes minuti)" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date
$cycleCount = 0
$lastCheckTime = $startTime

for ($i = 1; $i -le $maxCycles; $i++) {
    Write-Host "`n⏳ Aspettando ciclo $i/$maxCycles..." -ForegroundColor Gray
    
    # Aspetta 5 minuti (300 secondi)
    $waitTime = 300
    $elapsed = 0
    while ($elapsed -lt $waitTime) {
        Start-Sleep -Seconds 60  # Controlla ogni minuto
        $elapsed += 60
        $remaining = $waitTime - $elapsed
        Write-Host "   ⏱️  $(Get-Date -Format 'HH:mm:ss') - Mancano ~$([math]::Floor($remaining/60)) minuti al prossimo ciclo..." -ForegroundColor DarkGray
    }
    
    $cycleCount++
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "📋 CICLO $cycleCount/$maxCycles - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Mostra log del ciclo
    $logs = Get-Content "logs\automation_24h.log" -Tail 1000 -ErrorAction SilentlyContinue | 
        Select-String -Pattern "(API-SPORTS ha restituito|Partite LIVE processate|Partita LIVE trovata|Found.*LIVE matches|Saltate.*pre-match|Cycle complete|opportunità|selezionate|🎯 Partite LIVE|Trovate.*partite da sistema)" | 
        Select-Object -Last 20
    
    if ($logs) {
        Write-Host "`n📊 Log rilevanti:" -ForegroundColor Green
        $logs | ForEach-Object { 
            Write-Host "   $($_.Line)" 
        }
    } else {
        Write-Host "   ⚠️  Nessun log rilevante trovato" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ MONITORAGGIO COMPLETATO!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n📊 REPORT FINALE (ultimi 60 log rilevanti):" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

$finalLogs = Get-Content "logs\automation_24h.log" -Tail 3000 -ErrorAction SilentlyContinue | 
    Select-String -Pattern "(API-SPORTS ha restituito|Partite LIVE processate|Partita LIVE trovata|Found.*LIVE matches|Saltate.*pre-match|Cycle complete|opportunità|selezionate|🎯 Partite LIVE|Trovate.*partite da sistema)" | 
    Select-Object -Last 60

if ($finalLogs) {
    $finalLogs | ForEach-Object { 
        Write-Host $_.Line 
    }
} else {
    Write-Host "   ⚠️  Nessun log trovato" -ForegroundColor Yellow
}




