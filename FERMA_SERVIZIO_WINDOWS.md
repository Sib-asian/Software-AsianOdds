# 🛑 FERMA SERVIZIO WINDOWS AUTOMATION

## 🚨 PROBLEMA IDENTIFICATO

Hai un **Servizio Windows** attivo che gira in background 24/7 sul tuo PC!

```
File: automation_service_wrapper.py
Loop: while True (infinito)
Retry: ogni 1 ORA (3600 secondi) se crashato
Chiamate API: automation_24h.py → API-SPORTS
```

**QUESTO È LA CAUSA DELLE CHIAMATE OGNI ORA!**

---

## 🛑 PASSO 1: FERMA IL SERVIZIO

### Metodo A: Task Manager (Più veloce)

1. **Apri Task Manager** (Ctrl+Shift+Esc)
2. Vai su tab **"Dettagli"** (o "Details")
3. **Cerca processi Python**:
   ```
   python.exe
   pythonw.exe
   ```
4. **Controlla la riga "Comando"**:
   - Cerca: `automation_service_wrapper.py`
   - Oppure: `automation_24h.py`
5. **Click destro** → **"Termina processo"**
6. ✅ Conferma

---

### Metodo B: Servizi Windows (Se installato come servizio)

1. **Apri Servizi** (Win+R → `services.msc`)
2. **Cerca servizi** con nome tipo:
   ```
   Automation24H
   BettingAutomation
   AsianOdds
   Python
   ```
3. **Click destro** sul servizio → **"Arresta"**
4. **Click destro** di nuovo → **"Proprietà"**
5. **Tipo di avvio**: Cambia in **"Disabilitato"**
6. ✅ Click **"OK"**

---

### Metodo C: PowerShell (Alternativo)

Apri **PowerShell come Amministratore**:

```powershell
# Lista processi Python
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Format-Table Id, ProcessName, StartTime, Path

# Ferma processo specifico (sostituisci PID)
Stop-Process -Id <PID> -Force

# Oppure ferma TUTTI i processi Python (ATTENZIONE!)
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force
```

---

### Metodo D: Script Batch

Crea un file `ferma_automation.bat`:

```batch
@echo off
echo Fermando automazione...
taskkill /F /IM python.exe
taskkill /F /IM pythonw.exe
echo Fatto!
pause
```

**Click destro** → **"Esegui come amministratore"**

---

## 🔍 PASSO 2: VERIFICA CHE SIA FERMATO

### Controlla Task Manager

1. Apri Task Manager
2. Tab "Dettagli"
3. ✅ **Nessun processo `python.exe` con `automation` nel comando**

### Controlla Servizi

1. Apri Servizi (services.msc)
2. ✅ **Nessun servizio "Automation" in esecuzione**

### Controlla API-SPORTS (Dopo 1 ora)

1. Vai su: https://dashboard.api-football.com/
2. Aspetta **1 ora**
3. ✅ **Nessuna nuova chiamata dall'IP del tuo PC**

---

## 🗑️ PASSO 3: DISINSTALLA IL SERVIZIO (Permanente)

Se era installato come servizio Windows:

### Via PowerShell (Amministratore)

```powershell
# Lista servizi con "automation" nel nome
Get-Service | Where-Object {$_.DisplayName -like "*automation*"}

# Ferma servizio (sostituisci NomeServizio)
Stop-Service -Name "NomeServizio" -Force

# Disinstalla servizio
sc.exe delete "NomeServizio"
```

### Via Script di Disinstallazione

Cerca nella directory del progetto:

```
/home/user/Software-AsianOdds/uninstall_service.ps1
/home/user/Software-AsianOdds/manage_service.bat
```

Se trovi `uninstall_service.ps1`:

```powershell
# Click destro → "Esegui con PowerShell"
# Oppure da PowerShell Amministratore:
.\uninstall_service.ps1
```

---

## 🔒 PASSO 4: PREVIENI RIAVVIO AUTOMATICO

### Disabilita Avvio Automatico

1. **Win+R** → `taskschd.msc` (Task Scheduler)
2. Cerca task con nome tipo:
   ```
   Automation
   Betting
   AsianOdds
   ```
3. **Click destro** → **"Disabilita"** o **"Elimina"**

### Controlla Startup

1. **Task Manager** → Tab **"Avvio"**
2. Cerca:
   ```
   automation_service_wrapper.py
   start_automation.py
   ```
3. Se presente: **Click destro** → **"Disabilita"**

---

## ⚠️ TROUBLESHOOTING

### "Non trovo il processo"

**Possibile causa**: Gira come servizio nascosto

**Soluzione**:
```powershell
# Mostra TUTTI i processi Python con comando completo
Get-WmiObject Win32_Process -Filter "name = 'python.exe'" | Select-Object ProcessId, CommandLine
Get-WmiObject Win32_Process -Filter "name = 'pythonw.exe'" | Select-Object ProcessId, CommandLine
```

Cerca output con `automation` nel `CommandLine`.

---

### "Si riavvia automaticamente"

**Possibile causa**: Task Scheduler attivo

**Soluzione**:
1. Apri Task Scheduler (taskschd.msc)
2. Libreria Utilità di pianificazione
3. Cerca e elimina task automation

---

### "Errore: Accesso negato"

**Soluzione**:
1. Chiudi tutti i programmi
2. Apri PowerShell **come Amministratore**
3. Riprova il comando

---

## ✅ CHECKLIST COMPLETA

- [ ] Task Manager aperto
- [ ] Processi Python trovati: _____
- [ ] Processi Python fermati
- [ ] Servizi Windows controllati
- [ ] Servizio "Automation" fermato/disabilitato
- [ ] Task Scheduler controllato
- [ ] Task automation eliminati
- [ ] Startup disabilitato
- [ ] PC riavviato (per essere sicuri)
- [ ] Dopo 1 ora: API-SPORTS nessuna nuova chiamata ✅

---

## 🎯 DOPO AVER FERMATO

Aspetta **1 ora** poi controlla:

```
Dashboard API-SPORTS
└─ Requests
   └─ Nessuna nuova chiamata ✅
   └─ Contatore fermo ✅
```

Se le chiamate continuano → C'è altro in esecuzione (vedi diagnostico completo).

---

## 📊 IDENTIFICAZIONE PROCESSO

Per capire QUALE processo sta chiamando, esegui:

```powershell
# Mostra processi Python con percorso completo
Get-Process python* | Select-Object Id, ProcessName, Path, StartTime | Format-Table -AutoSize

# Mostra comando completo
Get-WmiObject Win32_Process -Filter "name = 'python.exe'" | Format-List ProcessId, CommandLine
```

**Cerca output tipo**:
```
ProcessId  : 12345
CommandLine: python.exe automation_service_wrapper.py  ← QUESTO!
```

---

## 🆘 SE NULLA FUNZIONA

### Opzione Nucleare 1: Disinstalla Python (Temporaneo)

```
Pannello di Controllo
→ Programmi e funzionalità
→ Python 3.11
→ Disinstalla
```

**ATTENZIONE**: Elimina TUTTO Python!

---

### Opzione Nucleare 2: Rigenera Chiave API

1. https://dashboard.api-football.com/
2. Account → API Keys
3. "Regenerate Key"
4. ✅ Vecchia chiave IMMEDIATAMENTE invalidata
5. ✅ Servizio Windows non potrà più chiamare API

Poi quando vuoi riattivare Render, aggiorno io la chiave.

---

**Creato**: 2025-11-23 11:30
**Urgenza**: 🔴 CRITICA
**Versione**: 1.0
