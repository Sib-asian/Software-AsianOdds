# 🤖 Automazione 24/7 - Setup Completo Automatico

## 🎯 Cosa Fa Questo Sistema

Sistema **completamente autonomo** che:
- ✅ **Gira 24/7** senza intervento manuale
- ✅ **Si avvia automaticamente** all'avvio di Windows
- ✅ **Si riavvia automaticamente** se crasha
- ✅ **Monitora partite** e analizza value bet
- ✅ **Notifica Telegram** solo per vere opportunità
- ✅ **Gestisce errori** e log dettagliati

---

## 🚀 Setup Rapido (3 Minuti)

### Opzione 1: Installazione Automatica (CONSIGLIATA)

1. **Right-click** su `install_service.ps1`
2. Seleziona **"Run with PowerShell"** (come amministratore)
3. **Fatto!** Il sistema ora gira 24/7 automaticamente

### Opzione 2: Installazione Manuale

1. Apri PowerShell **come Amministratore**
2. Naviga nella cartella del progetto:
   ```powershell
   cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
   ```
3. Esegui installazione:
   ```powershell
   .\install_service.ps1
   ```

---

## 📋 Cosa Viene Installato

Il sistema crea un **Task Scheduler Windows** che:
- ✅ Si avvia **automaticamente all'avvio Windows**
- ✅ Si **riavvia automaticamente** se crasha (max 3 volte/minuto)
- ✅ **Verifica ogni 5 minuti** che sia in esecuzione
- ✅ **Non si ferma** anche se il PC va in sleep/hibernation

---

## 🎮 Gestione Servizio

### Metodo 1: Script Batch (Facile)

Doppio-click su `manage_service.bat` per menu interattivo:
- Avvia/Ferma/Riavvia servizio
- Visualizza stato
- Apri log
- Installa/Rimuovi servizio

### Metodo 2: PowerShell

```powershell
# Avvia servizio
Start-ScheduledTask -TaskName "Automation24H_BettingSystem"

# Ferma servizio
Stop-ScheduledTask -TaskName "Automation24H_BettingSystem"

# Riavvia servizio
Restart-ScheduledTask -TaskName "Automation24H_BettingSystem"

# Stato servizio
Get-ScheduledTask -TaskName "Automation24H_BettingSystem"

# Info dettagliate
Get-ScheduledTaskInfo -TaskName "Automation24H_BettingSystem"
```

### Metodo 3: Task Scheduler GUI

1. Apri **Task Scheduler** (cerca "Task Scheduler" nel menu Start)
2. Vai in **Task Scheduler Library**
3. Cerca **"Automation24H_BettingSystem"**
4. Right-click per opzioni (Run, End, Disable, etc.)

---

## 📝 Log e Monitoraggio

### Log File

I log sono salvati in: `logs/automation_service_YYYYMMDD.log`

**Visualizza log:**
```powershell
# Ultime 50 righe
Get-Content logs\automation_service_*.log -Tail 50

# Segui log in tempo reale
Get-Content logs\automation_service_*.log -Wait -Tail 20
```

**Apri cartella log:**
- Doppio-click su `manage_service.bat` → Opzione 5
- Oppure: `explorer logs`

### Verifica Stato

```powershell
# Verifica se è in esecuzione
$task = Get-ScheduledTask -TaskName "Automation24H_BettingSystem"
$info = Get-ScheduledTaskInfo -TaskName "Automation24H_BettingSystem"

Write-Host "Stato: $($task.State)"
Write-Host "Ultimo avvio: $($info.LastRunTime)"
Write-Host "Prossimo avvio: $($info.NextRunTime)"
```

---

## ⚙️ Configurazione

### Variabili Ambiente (Opzionale)

Crea file `.env` nella cartella principale:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
AUTOMATION_MIN_EV=8.0
AUTOMATION_MIN_CONFIDENCE=70.0
AUTOMATION_UPDATE_INTERVAL=300
```

**Se non configuri `.env`:**
- Il sistema usa valori di default hardcoded
- Funziona comunque, ma riceverai warning nei log

### Parametri Configurabili

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `AUTOMATION_MIN_EV` | 8.0% | Expected Value minimo per notificare |
| `AUTOMATION_MIN_CONFIDENCE` | 70.0% | Confidence minima per notificare |
| `AUTOMATION_UPDATE_INTERVAL` | 300s | Secondi tra aggiornamenti (5 min) |

---

## 🔧 Troubleshooting

### Servizio non si avvia

1. **Verifica Python installato:**
   ```powershell
   python --version
   ```

2. **Verifica script esiste:**
   ```powershell
   Test-Path "automation_service_wrapper.py"
   ```

3. **Verifica log per errori:**
   ```powershell
   Get-Content logs\automation_service_*.log -Tail 50
   ```

4. **Riavvia servizio:**
   ```powershell
   Restart-ScheduledTask -TaskName "Automation24H_BettingSystem"
   ```

### Servizio crasha continuamente

1. **Controlla log** per errori specifici
2. **Verifica dipendenze:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Testa manualmente:**
   ```powershell
   python automation_service_wrapper.py
   ```

### Servizio non si avvia all'avvio Windows

1. **Verifica task è abilitato:**
   ```powershell
   $task = Get-ScheduledTask -TaskName "Automation24H_BettingSystem"
   $task.State  # Deve essere "Ready"
   ```

2. **Ri-installa servizio:**
   ```powershell
   .\uninstall_service.ps1
   .\install_service.ps1
   ```

### Non ricevo notifiche Telegram

1. **Verifica token e chat_id** in `.env` o nei log
2. **Testa bot manualmente:**
   ```python
   from ai_system.telegram_notifier import TelegramNotifier
   notifier = TelegramNotifier(bot_token="...", chat_id="...")
   notifier.send_message("Test")
   ```

---

## 🗑️ Rimozione Servizio

### Metodo 1: Script Automatico

Right-click su `uninstall_service.ps1` → **"Run with PowerShell"** (come amministratore)

### Metodo 2: Manuale

```powershell
# Ferma servizio
Stop-ScheduledTask -TaskName "Automation24H_BettingSystem"

# Rimuovi servizio
Unregister-ScheduledTask -TaskName "Automation24H_BettingSystem" -Confirm:$false
```

---

## 📊 Struttura File

```
Software-AsianOdds-main/
├── automation_24h.py              # Sistema automazione principale
├── automation_service_wrapper.py  # Wrapper con auto-restart
├── start_automation.py            # Script avvio semplice
├── install_service.ps1            # Script installazione servizio
├── uninstall_service.ps1          # Script rimozione servizio
├── start_automation_service.bat   # Avvio manuale
├── manage_service.bat             # Menu gestione servizio
├── logs/                          # Cartella log (creata automaticamente)
│   └── automation_service_YYYYMMDD.log
└── .env                           # Configurazione (opzionale)
```

---

## ✅ Checklist Setup

- [ ] Python 3.8+ installato
- [ ] Dipendenze installate: `pip install -r requirements.txt`
- [ ] Telegram bot configurato (opzionale ma consigliato)
- [ ] Servizio installato: `.\install_service.ps1`
- [ ] Servizio verificato: `Get-ScheduledTask -TaskName "Automation24H_BettingSystem"`
- [ ] Log verificati: `Get-Content logs\automation_service_*.log -Tail 20`
- [ ] Test notifica Telegram ricevuta

---

## 🎉 Fatto!

**Il sistema ora gira 24/7 automaticamente!**

- ✅ Si avvia all'avvio Windows
- ✅ Si riavvia se crasha
- ✅ Monitora partite continuamente
- ✅ Notifica Telegram per opportunità
- ✅ Log dettagliati per debugging

**Non devi fare nulla!** Il sistema funziona completamente da solo. 🚀

---

## 📞 Supporto

Per problemi o domande:
1. Controlla i log in `logs/`
2. Verifica stato servizio con `manage_service.bat`
3. Testa manualmente con `python automation_service_wrapper.py`

