# Script PowerShell per monitorare log in tempo reale
param(
    [string]$LogFile = "logs\automation_24h.log",
    [int]$TailLines = 30,
    [int]$RefreshSeconds = 2
)

$ErrorActionPreference = "SilentlyContinue"

# Funzione per pulire la console
function Clear-Console {
    Clear-Host
}

# Funzione per mostrare header
function Show-Header {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "MONITORAGGIO LOG LIVE - SISTEMA AUTOMATION 24H" -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Nuove funzionalita attive:" -ForegroundColor Green
    Write-Host "   - 58 filtri anti-banali" -ForegroundColor White
    Write-Host "   - Traduzioni italiane mercati" -ForegroundColor White
    Write-Host "   - Statistiche live nei messaggi" -ForegroundColor White
    Write-Host "   - Champions League femminile supportata" -ForegroundColor White
    Write-Host "   - Europa Cup Women supportata" -ForegroundColor White
    Write-Host "   - Soglie: 75% conf, 10% EV" -ForegroundColor White
    Write-Host ""
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "Ultimo aggiornamento: $timestamp" -ForegroundColor Gray
    Write-Host "File log: $LogFile" -ForegroundColor Gray
    Write-Host "Aggiornamento ogni $RefreshSeconds secondi (Ctrl+C per interrompere)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
}

# Verifica se il file log esiste
if (-not (Test-Path $LogFile)) {
    Write-Host "File log non trovato: $LogFile" -ForegroundColor Yellow
    Write-Host "In attesa che il sistema generi il file log..." -ForegroundColor Yellow
    Write-Host ""
    
    # Attendi che il file venga creato
    $timeout = 60
    $elapsed = 0
    while (-not (Test-Path $LogFile) -and $elapsed -lt $timeout) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        Write-Host "." -NoNewline -ForegroundColor Gray
    }
    Write-Host ""
    
    if (-not (Test-Path $LogFile)) {
        Write-Host "File log non trovato dopo $timeout secondi" -ForegroundColor Red
        Write-Host "Verifica che il sistema sia in esecuzione" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "File log trovato! Inizio monitoraggio..." -ForegroundColor Green
    Write-Host ""
    Start-Sleep -Seconds 1
}

# Ottieni la posizione iniziale del file
$lastPosition = 0
if (Test-Path $LogFile) {
    $file = Get-Item $LogFile
    $lastPosition = $file.Length
}

# Loop principale
try {
    while ($true) {
        Clear-Console
        Show-Header
        
        if (Test-Path $LogFile) {
            # Leggi le ultime righe del file
            $lines = Get-Content $LogFile -Tail $TailLines -ErrorAction SilentlyContinue
            
            if ($lines) {
                # Mostra le righe con colori basati sul livello
                foreach ($line in $lines) {
                    if ($line -match "ERROR|Error") {
                        Write-Host $line -ForegroundColor Red
                    }
                    elseif ($line -match "WARNING|Warning") {
                        Write-Host $line -ForegroundColor Yellow
                    }
                    elseif ($line -match "INFO|Cycle complete|opportunity|notificata|Analizzando|Trovate") {
                        Write-Host $line -ForegroundColor Green
                    }
                    elseif ($line -match "DEBUG") {
                        Write-Host $line -ForegroundColor Gray
                    }
                    else {
                        Write-Host $line -ForegroundColor White
                    }
                }
            }
            else {
                Write-Host "Nessun log disponibile ancora..." -ForegroundColor Yellow
            }
            
            # Verifica se ci sono nuove righe
            $file = Get-Item $LogFile
            $currentSize = $file.Length
            
            if ($currentSize -gt $lastPosition) {
                # Ci sono nuove righe, leggi solo quelle nuove
                $stream = [System.IO.File]::OpenRead($LogFile)
                $stream.Position = $lastPosition
                $reader = New-Object System.IO.StreamReader($stream)
                
                while ($null -ne ($newLine = $reader.ReadLine())) {
                    if ($newLine -match "ERROR|Error") {
                        Write-Host $newLine -ForegroundColor Red
                    }
                    elseif ($newLine -match "WARNING|Warning") {
                        Write-Host $newLine -ForegroundColor Yellow
                    }
                    elseif ($newLine -match "INFO|Cycle complete|opportunity|notificata|Analizzando|Trovate") {
                        Write-Host $newLine -ForegroundColor Green
                    }
                    elseif ($newLine -match "DEBUG") {
                        Write-Host $newLine -ForegroundColor Gray
                    }
                    else {
                        Write-Host $newLine -ForegroundColor White
                    }
                }
                
                $reader.Close()
                $stream.Close()
                $lastPosition = $currentSize
            }
        }
        else {
            Write-Host "File log non trovato: $LogFile" -ForegroundColor Yellow
        }
        
        # Attendi prima del prossimo aggiornamento
        Start-Sleep -Seconds $RefreshSeconds
    }
}
catch {
    Write-Host ""
    Write-Host "Monitoraggio interrotto" -ForegroundColor Green
    Write-Host "Il sistema continua a funzionare in background" -ForegroundColor Gray
}
finally {
    Write-Host ""
    Write-Host "Monitoraggio terminato" -ForegroundColor Cyan
}

