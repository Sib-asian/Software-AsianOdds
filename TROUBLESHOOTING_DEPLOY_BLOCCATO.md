# 🔧 Troubleshooting: Deploy Bloccato su Render

## Problema
Il deploy si è bloccato su "Deploying..." dopo il manual deploy.

## Soluzioni

### 1. Verifica Stato Deploy
1. Vai su https://dashboard.render.com
2. Apri il tuo servizio "automation-24h"
3. Vai su **"Events"** (non "Logs")
4. Controlla lo stato dell'ultimo deploy:
   - 🟢 **Live** = Deploy completato
   - 🟡 **Building** = Build in corso (aspetta)
   - 🔴 **Failed** = Build fallito (vedi errori)
   - ⚪ **Updating** = Aggiornamento in corso

### 2. Se il Deploy è Bloccato su "Building"

**Possibili cause:**
- Build lenta (può richiedere 5-10 minuti)
- Dipendenze che impiegano tempo a installare
- Problemi di rete durante il download

**Cosa fare:**
1. **Aspetta 10-15 minuti** (build può essere lenta)
2. Se dopo 15 minuti è ancora bloccato:
   - Vai su **"Events"**
   - Clicca sul deploy bloccato
   - Controlla i log di build per errori
   - Se ci sono errori, copiali e risolvili

### 3. Se il Deploy è "Failed"

**Controlla i log di build:**
1. Vai su **"Events"**
2. Clicca sul deploy fallito
3. Cerca errori come:
   - `ModuleNotFoundError`
   - `SyntaxError`
   - `ImportError`
   - Errori Docker

**Soluzioni comuni:**
- **ModuleNotFoundError**: Aggiungi il modulo a `requirements.automation.txt`
- **SyntaxError**: Verifica che il codice Python sia corretto
- **ImportError**: Verifica che tutti i file necessari siano copiati nel Dockerfile

### 4. Se il Deploy è "Live" ma il Servizio Non Funziona

**Controlla i log runtime:**
1. Vai su **"Logs"**
2. Cerca errori come:
   - `Error in cycle`
   - `Import error`
   - `NameError`
   - `AttributeError`

**Soluzioni:**
- Se vedi errori, correggili nel codice
- Se non vedi errori ma il servizio non funziona, riavvia il servizio

### 5. Riavvia il Servizio

**Se tutto il resto fallisce:**
1. Vai su **"Settings"**
2. Scrolla fino a **"Manual Deploy"**
3. Clicca **"Clear build cache & deploy"**
4. Attendi che il deploy completi

### 6. Verifica Variabili d'Ambiente

**Assicurati che siano configurate:**
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `API_FOOTBALL_KEY`
- `AUTOMATION_MIN_EV` (opzionale)
- `AUTOMATION_MIN_CONFIDENCE` (opzionale)
- `AUTOMATION_UPDATE_INTERVAL` (opzionale)

### 7. Controlla Limiti Render

**Se sei su Free Plan:**
- Il servizio si spegne dopo 15 minuti di inattività
- Passa a Starter ($7/mese) per servizio 24/7

**Se sei su Starter/Pro:**
- Verifica che non ci siano limiti di risorse raggiunti
- Controlla "Metrics" per CPU/Memory usage

## Comandi Utili

### Verifica Logs in Tempo Reale
```bash
# Usa Render CLI (se installato)
render logs --service YOUR_SERVICE_ID --tail
```

### Verifica Stato
```bash
render services list
```

## Se Niente Funziona

1. **Cancella il servizio e ricrealo** (ultima risorsa)
2. **Contatta supporto Render**: support@render.com
3. **Verifica che il codice funzioni localmente** prima di deployare

## Prevenzione Futura

1. **Testa localmente** prima di deployare
2. **Usa Auto-Deploy** invece di Manual Deploy (più affidabile)
3. **Monitora i log** durante i primi minuti dopo il deploy
4. **Mantieni il codice pulito** e senza errori di sintassi

