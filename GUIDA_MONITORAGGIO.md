# Guida Monitoraggio Log in Tempo Reale

## 📋 Metodi per Monitorare i Log

### Metodo 1: PowerShell - Monitoraggio in Tempo Reale (CONSIGLIATO)

#### Passo 1: Apri PowerShell
- Premi `Windows + X` e seleziona "Windows PowerShell" o "Terminal"
- Oppure cerca "PowerShell" nel menu Start

#### Passo 2: Vai nella cartella del progetto
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
```

#### Passo 3: Monitora i log in tempo reale
```powershell
Get-Content automation_24h.log -Wait -Tail 50
```

**Cosa fa:**
- `-Wait`: Aggiorna automaticamente quando arrivano nuovi log
- `-Tail 50`: Mostra solo le ultime 50 righe
- Premi `Ctrl+C` per fermare il monitoraggio

#### Passo 4: Filtra solo le notifiche (OPZIONALE)
Apri un **nuovo** terminale PowerShell e usa:
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
Get-Content automation_24h.log -Wait | Select-String -Pattern "opportunità|notifica|Live betting opportunity"
```

**Cosa fa:**
- Mostra solo le righe con "opportunità", "notifica" o "Live betting opportunity"
- Aggiorna in tempo reale

---

### Metodo 2: Notepad++ - Monitoraggio con Aggiornamento Automatico

#### Passo 1: Apri Notepad++
- Se non ce l'hai, scaricalo da: https://notepad-plus-plus.org/

#### Passo 2: Apri il file di log
- File → Apri
- Vai in: `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main\automation_24h.log`

#### Passo 3: Attiva l'aggiornamento automatico
- Vai in: `View` → `Monitoring (tail -f)`
- Oppure: `View` → `Always on Top` (per tenerlo sempre visibile)

#### Passo 4: Cerca le notifiche
- Premi `Ctrl+F`
- Cerca: `opportunità` o `notifica`
- Usa "Trova successivo" per navigare tra le occorrenze

---

### Metodo 3: Visual Studio Code - Monitoraggio Avanzato

#### Passo 1: Apri VS Code
- Se non ce l'hai, scaricalo da: https://code.visualstudio.com/

#### Passo 2: Apri la cartella del progetto
- File → Apri Cartella
- Seleziona: `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main`

#### Passo 3: Apri il file di log
- Clicca su `automation_24h.log` nella barra laterale

#### Passo 4: Installa estensione "Log Viewer" (OPZIONALE)
- Premi `Ctrl+Shift+X`
- Cerca "Log Viewer" e installa
- Permette di filtrare e colorare i log

#### Passo 5: Monitora in tempo reale
- Il file si aggiorna automaticamente quando viene modificato
- Usa `Ctrl+F` per cercare "opportunità" o "notifica"

---

### Metodo 4: Comando PowerShell - Ultime Notifiche

#### Per vedere solo le ultime notifiche inviate:
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
Get-Content automation_24h.log | Select-String -Pattern "opportunità|notifica" | Select-Object -Last 20
```

#### Per vedere le partite analizzate (senza giovanili):
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
Get-Content automation_24h.log -Tail 200 | Select-String -Pattern "Analizzando partita LIVE" | Where-Object { $_ -notmatch "U21|U19|U17|U23" }
```

---

## 🎯 Cosa Cercare nei Log

### ✅ Segnali di Successo
- `🎯 Live betting opportunity notificata` - Notifica inviata!
- `✅ Analizzando partita LIVE` - Partita in analisi
- `confidence: XX%` - Confidence dell'opportunità

### ⚠️ Segnali di Problema
- `ERROR` o `Error` - Errore nel sistema
- `⚠️` - Avviso
- `Saltata opportunità` - Opportunità filtrata

### 📊 Informazioni Utili
- `Min Confidence: XX%` - Confidence minima configurata
- `Update Interval: XXXs` - Frequenza di aggiornamento
- `LiveMatchAI inizializzata` - IA attiva

---

## 🔍 Esempi di Ricerche Utili

### 1. Ultime 10 notifiche inviate
```powershell
Get-Content automation_24h.log | Select-String -Pattern "opportunità notificata" | Select-Object -Last 10
```

### 2. Partite senior analizzate (ultime 2 ore)
```powershell
Get-Content automation_24h.log | Select-String -Pattern "Analizzando partita LIVE" | Where-Object { $_ -notmatch "U21|U19|U17|U23" } | Select-Object -Last 20
```

### 3. Errori recenti
```powershell
Get-Content automation_24h.log -Tail 500 | Select-String -Pattern "ERROR|Error|Exception"
```

### 4. Confidence delle opportunità trovate
```powershell
Get-Content automation_24h.log | Select-String -Pattern "confidence: [0-9]{2}%" | Select-Object -Last 20
```

---

## 💡 Suggerimenti

1. **Usa due terminali**: Uno per monitorare tutto, uno per filtrare solo le notifiche
2. **Salva i comandi**: Crea un file `.ps1` con i comandi che usi spesso
3. **Monitora in background**: Lascia PowerShell aperto mentre lavori
4. **Cerca pattern specifici**: Adatta i filtri alle tue esigenze

---

## 🚀 Comando Rapido (Copia e Incolla)

Apri PowerShell e incolla questo comando per monitorare solo le notifiche:

```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"; Get-Content automation_24h.log -Wait | Select-String -Pattern "opportunità|notifica|Live betting opportunity|confidence: [6-9][0-9]%"
```

Questo mostrerà solo:
- Le notifiche inviate
- Le opportunità trovate
- Le confidence delle opportunità (60-99%)



