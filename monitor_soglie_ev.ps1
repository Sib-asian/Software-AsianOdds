# Script PowerShell per monitorare le soglie EV e le opportunità
# Mostra in tempo reale se le nuove soglie (8%) funzionano correttamente

Write-Host "`n📊 MONITORAGGIO SOGLIE EV - LIVE BETTING`n" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Gray
Write-Host "`n🔍 Monitoraggio attivo per verificare soglie EV 8%`n" -ForegroundColor Yellow
Write-Host "Premi Ctrl+C per interrompere`n" -ForegroundColor Gray
Write-Host "=" * 70 -ForegroundColor Gray
Write-Host ""

$logFile = "logs\automation_24h.log"
if (-not (Test-Path $logFile)) {
    Write-Host "❌ File di log non trovato: $logFile" -ForegroundColor Red
    Write-Host "   Assicurati che il sistema sia in esecuzione`n" -ForegroundColor Yellow
    exit
}

$lastPosition = (Get-Content $logFile -ErrorAction SilentlyContinue).Length
$cycleCount = 0
$opportunityCount = 0
$notificationCount = 0
$filteredCount = 0
$acceptedCount = 0

while ($true) {
    Start-Sleep -Seconds 3
    
    try {
        $content = Get-Content $logFile -ErrorAction SilentlyContinue
        if ($content -and $content.Length -gt $lastPosition) {
            $newLines = $content[$lastPosition..($content.Length - 1)]
            
            foreach ($line in $newLines) {
                $timestamp = if ($line -match "([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})") {
                    $matches[1].Split(' ')[1]
                } else {
                    ""
                }
                
                # Soglie inizializzate
                if ($line -match "LiveBettingAdvisor.*initialized|min_ev.*8|min_ev.*6") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "✅ SOGLIE: " -ForegroundColor Green -NoNewline
                    if ($line -match "min_ev=([0-9.]+)") {
                        $ev = $matches[1]
                        Write-Host "EV = $ev%" -ForegroundColor White
                    } else {
                        Write-Host $line -ForegroundColor White
                    }
                }
                
                # Cicli completati
                elseif ($line -match "Cycle complete") {
                    $cycleCount++
                    $opp = if ($line -match "([0-9]+) opportunities found") { $matches[1] } else { "0" }
                    $notif = if ($line -match "([0-9]+) notified") { $matches[1] } else { "0" }
                    $opportunityCount += [int]$opp
                    $notificationCount += [int]$notif
                    
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "✅ CICLO #$cycleCount" -ForegroundColor Green -NoNewline
                    Write-Host " | " -ForegroundColor Gray -NoNewline
                    Write-Host "Opp: $opp" -ForegroundColor Yellow -NoNewline
                    Write-Host " | " -ForegroundColor Gray -NoNewline
                    Write-Host "Notifiche: $notif" -ForegroundColor Cyan
                }
                
                # Opportunità filtrate per EV < 8%
                elseif ($line -match "valore atteso.*< soglia.*8|EV.*<.*8\.0") {
                    $filteredCount++
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "🔍 FILTRATO (EV < 8%): " -ForegroundColor Yellow -NoNewline
                    if ($line -match "valore atteso ([0-9.]+)%") {
                        $ev = $matches[1]
                        Write-Host "EV $ev% < 8%" -ForegroundColor Gray
                    } else {
                        Write-Host $line -ForegroundColor Gray
                    }
                }
                
                # Opportunità accettate con EV ≥ 8%
                elseif ($line -match "valore atteso.*([0-9.]+)% < soglia.*([0-9.]+)%" -and $line -notmatch "valore atteso.*< soglia.*8") {
                    $ev = if ($line -match "valore atteso ([0-9.]+)%") { $matches[1] } else { "" }
                    $threshold = if ($line -match "soglia ([0-9.]+)%") { $matches[1] } else { "" }
                    if ($ev -and $threshold -and [float]$ev -ge [float]$threshold) {
                        $acceptedCount++
                        Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                        Write-Host "✅ ACCETTATA: " -ForegroundColor Green -NoNewline
                        Write-Host "EV $ev% ≥ $threshold%" -ForegroundColor White
                    }
                }
                
                # Opportunità con EV 9.2% che ora passano
                elseif ($line -match "valore atteso 9\.2|EV.*9\.2" -and $line -notmatch "Saltata") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "🎯 OPPORTUNITA CON EV 9.2%: " -ForegroundColor Cyan -NoNewline
                    Write-Host "Ora passa con soglia 8%!" -ForegroundColor Green
                }
                
                # Notifiche inviate
                elseif ($line -match "Notified live opportunity|MESSAGGIO TELEGRAM|Telegram.*sent") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "📱 NOTIFICA INVIATA" -ForegroundColor Cyan
                }
                
                # Errori
                elseif ($line -match "ERROR|Error|Exception|Traceback") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "❌ ERRORE: " -ForegroundColor Red -NoNewline
                    Write-Host $line -ForegroundColor Red
                }
                
                # Partite LIVE trovate
                elseif ($line -match "Found.*LIVE matches|partite LIVE") {
                    $matches = if ($line -match "([0-9]+)") { $matches[1] } else { "0" }
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "⚽ LIVE: " -ForegroundColor Magenta -NoNewline
                    Write-Host "$matches partite" -ForegroundColor White
                }
            }
            
            $lastPosition = $content.Length
        }
    }
    catch {
        # Ignora errori di lettura
    }
    
    # Statistiche ogni 10 cicli
    if ($cycleCount -gt 0 -and $cycleCount % 10 -eq 0) {
        Write-Host "`n📊 STATISTICHE (ultimi 10 cicli):" -ForegroundColor Cyan
        Write-Host "   • Opportunità trovate: $opportunityCount" -ForegroundColor White
        Write-Host "   • Notifiche inviate: $notificationCount" -ForegroundColor White
        Write-Host "   • Filtrate (EV < 8%): $filteredCount" -ForegroundColor Yellow
        Write-Host "   • Accettate (EV ≥ 8%): $acceptedCount" -ForegroundColor Green
        Write-Host ""
    }
}




