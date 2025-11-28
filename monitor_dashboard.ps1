# Dashboard Monitoraggio Live Betting
# Si aggiorna automaticamente ogni 13 minuti

$logFile = "logs\automation_24h.log"
$updateInterval = 780  # 13 minuti in secondi

function Show-Dashboard {
    Clear-Host
    Write-Host "`n" -NoNewline
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "📊 DASHBOARD MONITORAGGIO LIVE BETTING" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "🕐 Ultimo aggiornamento: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "⏱️  Prossimo aggiornamento tra 13 minuti" -ForegroundColor Gray
    Write-Host ""
    
    if (-not (Test-Path $logFile)) {
        Write-Host "❌ File di log non trovato: $logFile" -ForegroundColor Red
        return
    }
    
    $log = Get-Content $logFile -Tail 500 -ErrorAction SilentlyContinue
    if (-not $log) {
        Write-Host "⚠️  Nessun log disponibile" -ForegroundColor Yellow
        return
    }
    
    # Statistiche generali
    Write-Host "📈 STATISTICHE GENERALI" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $totalCycles = ($log | Select-String -Pattern "Cycle complete").Count
    $totalOpportunities = ($log | Select-String -Pattern "opportunities found").Count
    $totalNotifications = ($log | Select-String -Pattern "Notified live opportunity|MESSAGGIO TELEGRAM").Count
    $totalErrors = ($log | Select-String -Pattern "ERROR|Error|❌|Exception").Count
    
    Write-Host "   Cicli completati: " -NoNewline -ForegroundColor White
    Write-Host "$totalCycles" -ForegroundColor $(if ($totalCycles -gt 0) { "Green" } else { "Yellow" })
    
    Write-Host "   Opportunità trovate: " -NoNewline -ForegroundColor White
    Write-Host "$totalOpportunities" -ForegroundColor $(if ($totalOpportunities -gt 0) { "Green" } else { "Gray" })
    
    Write-Host "   Notifiche inviate: " -NoNewline -ForegroundColor White
    Write-Host "$totalNotifications" -ForegroundColor $(if ($totalNotifications -gt 0) { "Green" } else { "Gray" })
    
    Write-Host "   Errori: " -NoNewline -ForegroundColor White
    Write-Host "$totalErrors" -ForegroundColor $(if ($totalErrors -eq 0) { "Green" } else { "Red" })
    
    Write-Host ""
    
    # Stato Telegram
    Write-Host "📱 STATO TELEGRAM" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $telegramStatus = $log | Select-String -Pattern "Telegram.*initialized|Telegram not configured" | Select-Object -Last 1
    if ($telegramStatus) {
        if ($telegramStatus.Line -match "initialized") {
            Write-Host "   ✅ Telegram configurato correttamente" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Telegram NON configurato" -ForegroundColor Red
        }
    } else {
        Write-Host "   ⚠️  Stato Telegram non trovato nei log recenti" -ForegroundColor Yellow
    }
    
    $configLoaded = $log | Select-String -Pattern "Config caricato" | Select-Object -Last 1
    if ($configLoaded) {
        Write-Host "   ✅ Config.json caricato" -ForegroundColor Green
    }
    
    Write-Host ""
    
    # Ultimo ciclo
    Write-Host "🔄 ULTIMO CICLO" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $lastCycle = $log | Select-String -Pattern "Cycle complete" | Select-Object -Last 1
    if ($lastCycle) {
        $cycleTime = if ($lastCycle.Line -match "([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})") { $matches[1] } else { "N/A" }
        $oppCount = if ($lastCycle.Line -match "([0-9]+) opportunities found") { $matches[1] } else { "0" }
        $notifCount = if ($lastCycle.Line -match "([0-9]+) notified") { $matches[1] } else { "0" }
        
        Write-Host "   Ora: $cycleTime" -ForegroundColor White
        Write-Host "   Opportunità: $oppCount" -ForegroundColor $(if ([int]$oppCount -gt 0) { "Green" } else { "Gray" })
        Write-Host "   Notifiche: $notifCount" -ForegroundColor $(if ([int]$notifCount -gt 0) { "Green" } else { "Gray" })
    } else {
        Write-Host "   ⚠️  Nessun ciclo trovato nei log recenti" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Partite LIVE
    Write-Host "⚽ PARTITE LIVE" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $liveMatches = $log | Select-String -Pattern "Found (\d+) LIVE matches|partite LIVE" | Select-Object -Last 1
    if ($liveMatches) {
        $matchCount = if ($liveMatches.Line -match "([0-9]+)") { $matches[1] } else { "0" }
        Write-Host "   Partite LIVE trovate: $matchCount" -ForegroundColor $(if ([int]$matchCount -gt 0) { "Green" } else { "Yellow" })
    } else {
        Write-Host "   ⚠️  Nessuna partita LIVE trovata recentemente" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Ultime notifiche
    Write-Host "📱 ULTIME NOTIFICHE (ultime 3)" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $notifications = $log | Select-String -Pattern "Notified live opportunity|MESSAGGIO TELEGRAM" | Select-Object -Last 3
    if ($notifications) {
        foreach ($notif in $notifications) {
            $notifTime = if ($notif.Line -match "([0-9]{2}:[0-9]{2}:[0-9]{2})") { $matches[1] } else { "" }
            $market = if ($notif.Line -match "(\w+_\d+\.?\d*|\w+)") { $matches[1] } else { "N/A" }
            Write-Host "   [$notifTime] " -NoNewline -ForegroundColor Gray
            Write-Host "$market" -ForegroundColor Cyan
        }
    } else {
        Write-Host "   ⚠️  Nessuna notifica recente" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Errori recenti
    if ($totalErrors -gt 0) {
        Write-Host "❌ ERRORI RECENTI (ultimi 3)" -ForegroundColor Yellow
        Write-Host "-" * 80 -ForegroundColor Gray
        
        $errors = $log | Select-String -Pattern "ERROR|Error|❌|Exception" | Select-Object -Last 3
        foreach ($error in $errors) {
            $errorTime = if ($error.Line -match "([0-9]{2}:[0-9]{2}:[0-9]{2})") { $matches[1] } else { "" }
            Write-Host "   [$errorTime] " -NoNewline -ForegroundColor Gray
            Write-Host $error.Line -ForegroundColor Red
        }
        Write-Host ""
    }
    
    # Processi Python
    Write-Host "🐍 PROCESSI PYTHON" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor Gray
    
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        Write-Host "   ✅ Python in esecuzione: $($pythonProcesses.Count) processo/i" -ForegroundColor Green
        foreach ($proc in $pythonProcesses) {
            $uptime = (Get-Date) - $proc.StartTime
            Write-Host "      • PID: $($proc.Id) | Uptime: $([math]::Round($uptime.TotalMinutes, 1)) minuti" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ❌ Nessun processo Python in esecuzione" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "💡 Dashboard si aggiornerà automaticamente tra 13 minuti" -ForegroundColor Yellow
    Write-Host "   Premi Ctrl+C per interrompere" -ForegroundColor Gray
    Write-Host ""
}

# Loop principale
while ($true) {
    Show-Dashboard
    Write-Host "⏳ Attendo 13 minuti per il prossimo aggiornamento..." -ForegroundColor Cyan
    Start-Sleep -Seconds $updateInterval
}

