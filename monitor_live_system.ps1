# Monitoraggio Sistema Live Betting
# Mostra log in tempo reale con evidenziazione

$logFile = "logs\automation_24h.log"

if (-not (Test-Path $logFile)) {
    Write-Host "❌ File di log non trovato: $logFile" -ForegroundColor Red
    exit 1
}

Write-Host "`n" -NoNewline
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "📊 MONITORAGGIO SISTEMA LIVE BETTING" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "File: $logFile" -ForegroundColor Gray
Write-Host "Premi Ctrl+C per interrompere`n" -ForegroundColor Yellow

$lastPosition = 0
$stats = @{
    Cycles = 0
    Opportunities = 0
    Notifications = 0
    Errors = 0
    LiveMatches = 0
    PreMatchSkipped = 0
}

function Show-Stats {
    Write-Host "`n📈 STATISTICHE:" -ForegroundColor Cyan
    Write-Host "   Cicli: $($stats.Cycles) | Opportunità: $($stats.Opportunities) | Notifiche: $($stats.Notifications) | Errori: $($stats.Errors)" -ForegroundColor White
    Write-Host ""
}

while ($true) {
    Start-Sleep -Seconds 3
    
    if (Test-Path $logFile) {
        try {
            $content = Get-Content $logFile -ErrorAction SilentlyContinue
            if ($content -and $content.Length -gt $lastPosition) {
                $newLines = $content[$lastPosition..($content.Length - 1)]
                
                foreach ($line in $newLines) {
                    $timestamp = if ($line -match "(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})") { $matches[1] } else { "" }
                    $timeOnly = if ($timestamp) { $timestamp.Split(' ')[1] } else { "" }
                    
                    # Cicli completati
                    if ($line -match "Cycle complete") {
                        $stats.Cycles++
                        $oppMatch = if ($line -match "(\d+) opportunities found") { $matches[1] } else { "0" }
                        $notifMatch = if ($line -match "(\d+) notified") { $matches[1] } else { "0" }
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "✅ CICLO COMPLETATO" -ForegroundColor Green -NoNewline
                        Write-Host " | Opp: $oppMatch | Notifiche: $notifMatch" -ForegroundColor White
                        $stats.Opportunities += [int]$oppMatch
                        $stats.Notifications += [int]$notifMatch
                    }
                    # Partite LIVE
                    elseif ($line -match "Found (\d+) LIVE matches|partite LIVE") {
                        $matchCount = if ($line -match "(\d+)") { $matches[1] } else { "0" }
                        $stats.LiveMatches = [int]$matchCount
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "⚽ PARTITE LIVE: $matchCount" -ForegroundColor Magenta
                    }
                    # Partite pre-match saltate
                    elseif ($line -match "Saltate (\d+) partite pre-match") {
                        $skipped = if ($line -match "(\d+)") { $matches[1] } else { "0" }
                        $stats.PreMatchSkipped = [int]$skipped
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "⏭️  PRE-MATCH SALTATE: $skipped" -ForegroundColor Blue
                    }
                    # Notifiche Telegram
                    elseif ($line -match "Notified live opportunity|MESSAGGIO TELEGRAM") {
                        $market = if ($line -match "market[:\s]+(\w+)") { $matches[1] } else { "" }
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "📱 TELEGRAM INVIATO" -ForegroundColor Green -NoNewline
                        if ($market) { Write-Host " | Mercato: $market" -ForegroundColor White } else { Write-Host "" }
                    }
                    # Opportunità trovate
                    elseif ($line -match "opportunities found|opportunity.*found" -and $line -notmatch "Cycle complete") {
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "🎯 OPPORTUNITA: " -ForegroundColor Yellow -NoNewline
                        Write-Host $line -ForegroundColor White
                    }
                    # Tracking/Report
                    elseif ($line -match "Live Betting Performance Tracker|Soglie dinamiche|Report|tracker") {
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "📊 TRACKING: " -ForegroundColor Cyan -NoNewline
                        Write-Host $line -ForegroundColor White
                    }
                    # Errori
                    elseif ($line -match "ERROR|Error|❌|FAILED|Failed|Exception|Traceback") {
                        $stats.Errors++
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "❌ ERRORE: " -ForegroundColor Red -NoNewline
                        Write-Host $line -ForegroundColor Red
                    }
                    # Warning importanti
                    elseif ($line -match "WARNING.*Live|WARNING.*opportunity|WARNING.*match") {
                        Write-Host "[$timeOnly] " -ForegroundColor Gray -NoNewline
                        Write-Host "⚠️  WARNING: " -ForegroundColor Yellow -NoNewline
                        Write-Host $line -ForegroundColor Yellow
                    }
                }
                
                $lastPosition = $content.Length
                
                # Mostra statistiche ogni 10 righe nuove
                if ($newLines.Count -ge 10) {
                    Show-Stats
                }
            }
        } catch {
            # Ignora errori di lettura
        }
    }
}




