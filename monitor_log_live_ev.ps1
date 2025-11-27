# Script PowerShell per monitorare i log in tempo reale
# Focus su soglie EV, opportunità e notifiche

Write-Host "`n📊 MONITORAGGIO LOG LIVE BETTING - SOGLIE EV`n" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Gray
Write-Host "`n🔍 Monitoraggio attivo in tempo reale...`n" -ForegroundColor Yellow
Write-Host "Premi Ctrl+C per interrompere`n" -ForegroundColor Gray
Write-Host "=" * 70 -ForegroundColor Gray
Write-Host ""

$logFile = "logs\automation_24h.log"
if (-not (Test-Path $logFile)) {
    Write-Host "❌ File di log non trovato: $logFile" -ForegroundColor Red
    Write-Host "   Assicurati che il sistema sia in esecuzione`n" -ForegroundColor Yellow
    pause
    exit
}

$lastPosition = (Get-Content $logFile -ErrorAction SilentlyContinue).Length
$cycleCount = 0
$opportunityCount = 0
$notificationCount = 0
$filteredCount = 0
$acceptedCount = 0
$lastStatsTime = Get-Date

while ($true) {
    Start-Sleep -Seconds 2
    
    try {
        $content = Get-Content $logFile -ErrorAction SilentlyContinue
        if ($content -and $content.Length -gt $lastPosition) {
            $newLines = $content[$lastPosition..($content.Length - 1)]
            
            foreach ($line in $newLines) {
                $timestamp = if ($line -match "([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})") {
                    $Matches[1].Split(' ')[1]
                } else {
                    ""
                }
                
                # Soglie inizializzate
                if ($line -match "LiveBettingAdvisor.*initialized|min_ev.*8|min_ev.*6|min_confidence.*70") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "✅ SOGLIE ATTIVE: " -ForegroundColor Green -NoNewline
                    if ($line -match "min_ev=([0-9.]+)") {
                        $ev = $Matches[1]
                        Write-Host "EV = $ev%" -ForegroundColor White
                    } elseif ($line -match "min_confidence=([0-9.]+)") {
                        $conf = $Matches[1]
                        Write-Host "Confidence = $conf%" -ForegroundColor White
                    } else {
                        Write-Host $line -ForegroundColor White
                    }
                }
                
                # Cicli completati
                elseif ($line -match "Cycle complete") {
                    $cycleCount++
                    $opp = if ($line -match "([0-9]+) opportunities found") { $Matches[1] } else { "0" }
                    $notif = if ($line -match "([0-9]+) notified") { $Matches[1] } else { "0" }
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
                elseif ($line -match "valore atteso.*< soglia|EV.*<.*soglia") {
                    $filteredCount++
                    $evMatch = if ($line -match "valore atteso ([0-9.]+)%") { $Matches[1] } else { "" }
                    $thresholdMatch = if ($line -match "soglia ([0-9.]+)%") { $Matches[1] } else { "" }
                    
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    if ($thresholdMatch -eq "8.0" -or $thresholdMatch -eq "8") {
                        Write-Host "🔍 FILTRATO (EV minore di 8%): " -ForegroundColor Yellow -NoNewline
                        Write-Host "EV $evMatch% minore di 8%" -ForegroundColor Gray
                    } else {
                        Write-Host "🔍 FILTRATO: " -ForegroundColor Yellow -NoNewline
                        Write-Host "EV $evMatch% minore di $thresholdMatch%" -ForegroundColor Gray
                    }
                }
                
                # Opportunità con EV 9.2% che ora passano
                elseif ($line -match "valore atteso 9\.2|EV.*9\.2" -and $line -notmatch "Saltata") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "🎯 OPPORTUNITA EV 9.2%: " -ForegroundColor Cyan -NoNewline
                    Write-Host "Ora passa con soglia 8%!" -ForegroundColor Green
                    $acceptedCount++
                }
                
                # Notifiche inviate
                elseif ($line -match "Notified live opportunity|MESSAGGIO TELEGRAM|Telegram.*sent") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "📱 NOTIFICA INVIATA" -ForegroundColor Cyan
                }
                
                # Partite LIVE trovate
                elseif ($line -match "Found.*LIVE matches|partite LIVE|Trovate.*partite LIVE") {
                    $matchCount = if ($line -match "([0-9]+)") { $Matches[1] } else { "0" }
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "⚽ LIVE: " -ForegroundColor Magenta -NoNewline
                    Write-Host "$matchCount partite trovate" -ForegroundColor White
                }
                
                # Errori
                elseif ($line -match "ERROR|Error|Exception|Traceback|Failed") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "❌ ERRORE: " -ForegroundColor Red -NoNewline
                    Write-Host $line -ForegroundColor Red
                }
                
                # Opportunità trovate (dettaglio)
                elseif ($line -match "opportunities found" -and $line -notmatch "Cycle complete") {
                    Write-Host "[$timestamp] " -ForegroundColor Gray -NoNewline
                    Write-Host "🎯 OPPORTUNITA: " -ForegroundColor Yellow -NoNewline
                    Write-Host $line -ForegroundColor White
                }
            }
            
            $lastPosition = $content.Length
        }
        
        # Statistiche ogni 30 secondi
        $now = Get-Date
        if (($now - $lastStatsTime).TotalSeconds -ge 30) {
            Write-Host "`n📊 STATISTICHE (ultimi 30 secondi):" -ForegroundColor Cyan
            Write-Host "   • Cicli: $cycleCount" -ForegroundColor White
            Write-Host "   • Opportunità totali: $opportunityCount" -ForegroundColor White
            Write-Host "   • Notifiche inviate: $notificationCount" -ForegroundColor Cyan
            Write-Host "   • Filtrate (EV minore di 8%): $filteredCount" -ForegroundColor Yellow
            Write-Host "   • Accettate: $acceptedCount" -ForegroundColor Green
            Write-Host ""
            $lastStatsTime = $now
        }
    }
    catch {
        # Ignora errori di lettura
    }
}

