Write-Host "📊 MONITORAGGIO 6 CICLI COMPLETI" -ForegroundColor Cyan
Write-Host "⏰ Ora inizio: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
Write-Host "⏱️  Durata: ~30 minuti (6 cicli x 5 minuti)" -ForegroundColor Yellow
Write-Host "🔄 Monitoraggio attivo ogni 30 secondi..." -ForegroundColor Green
Write-Host ""

$startTime = Get-Date
$cycleCount = 0
$maxCycles = 6
$cycleDuration = 300  # 5 minuti
$lastCycleTime = $startTime

while ($cycleCount -lt $maxCycles) {
    $elapsed = (Get-Date) - $lastCycleTime
    
    # Controlla se è passato un ciclo (5 minuti)
    if ($elapsed.TotalSeconds -ge $cycleDuration) {
        $cycleCount++
        $cycleTime = Get-Date
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "📋 CICLO $cycleCount/$maxCycles completato - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        
        # Mostra log rilevanti del ciclo
        $cycleLogs = Get-Content "logs\automation_24h.log" -Tail 500 -ErrorAction SilentlyContinue | 
            Select-String -Pattern "(API-SPORTS ha restituito|Partite LIVE processate|Partita LIVE trovata|Found.*LIVE matches|Saltate.*pre-match|Cycle complete|opportunità|selezionate|🎯 Partite LIVE|Trovate.*partite da sistema)" | 
            Select-Object -Last 20
        
        if ($cycleLogs) {
            Write-Host "`n📊 Log rilevanti del ciclo:" -ForegroundColor Green
            $cycleLogs | ForEach-Object { 
                Write-Host "   $($_.Line)" 
            }
        } else {
            Write-Host "   ⚠️  Nessun log rilevante trovato" -ForegroundColor Yellow
        }
        
        $lastCycleTime = $cycleTime
    }
    
    # Mostra log recenti ogni 30 secondi
    $recent = Get-Content "logs\automation_24h.log" -Tail 50 -ErrorAction SilentlyContinue | 
        Select-String -Pattern "(API-SPORTS ha restituito|Partite LIVE processate|Partita LIVE trovata|Found.*LIVE matches|Cycle complete|opportunità|selezionate)" | 
        Select-Object -Last 1
    
    if ($recent) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] $($recent[-1].Line)" -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds 30
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ MONITORAGGIO 6 CICLI COMPLETATO!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n📊 REPORT FINALE:" -ForegroundColor Yellow
Write-Host "================" -ForegroundColor Yellow

$finalLogs = Get-Content "logs\automation_24h.log" -Tail 2000 -ErrorAction SilentlyContinue | 
    Select-String -Pattern "(API-SPORTS ha restituito|Partite LIVE processate|Partita LIVE trovata|Found.*LIVE matches|Saltate.*pre-match|Cycle complete|opportunità|selezionate|🎯 Partite LIVE|Trovate.*partite da sistema)" | 
    Select-Object -Last 60

$finalLogs | ForEach-Object { 
    Write-Host $_.Line 
}




