# 🚀 Guida Passo-Passo: Deploy su Render.com

## 📋 Prerequisiti

Prima di iniziare, assicurati di avere:
- [ ] Codice committato su GitHub (repository pubblico o privato)
- [ ] Token Telegram Bot (da @BotFather)
- [ ] Chat ID Telegram (da @userinfobot)
- [ ] Account GitHub

---

## 🎯 STEP 1: Prepara il Codice su GitHub

### 1.1 Verifica che il codice sia su GitHub

```bash
# Se non hai ancora fatto push su GitHub:
cd Software-AsianOdds-main
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

**✅ Verifica**: Vai su GitHub e controlla che il repository contenga:
- `Dockerfile.automation`
- `start_automation.py`
- `automation_24h.py`
- `requirements.automation.txt`
- Cartella `ai_system/`

---

## 🎯 STEP 2: Crea Account Render.com

### 2.1 Vai su Render.com
1. Apri il browser e vai su: **https://render.com**
2. Clicca su **"Get Started for Free"** (in alto a destra)

### 2.2 Login con GitHub (CONSIGLIATO)
1. Clicca su **"Sign up with GitHub"**
2. Autorizza Render ad accedere al tuo GitHub
3. ✅ Account creato!

**Alternativa**: Puoi anche registrarti con email, ma GitHub è più veloce.

---

## 🎯 STEP 3: Crea Background Worker

### 3.1 Accedi al Dashboard
1. Dopo il login, vedrai il **Dashboard**
2. Clicca sul pulsante **"New +"** (in alto a sinistra, blu)
3. Seleziona **"Background Worker"**

### 3.2 Connetti Repository GitHub

**Opzione A: Repository Pubblico**
1. Seleziona **"Public Git repository"**
2. Incolla l'URL del tuo repository GitHub:
   ```
   https://github.com/TUO-USERNAME/Software-AsianOdds-main
   ```
   (Sostituisci TUO-USERNAME con il tuo username GitHub)
3. Clicca **"Continue"**

**Opzione B: Repository Privato (se hai connesso GitHub)**
1. Seleziona **"Private Git repository"**
2. Scegli il repository dalla lista: `Software-AsianOdds-main`
3. Clicca **"Continue"**

### 3.3 Configurazione Base

Compila i campi:

```
Name: automation-24h
Region: Frankfurt (o scegli la regione più vicina a te)
Branch: main (o il tuo branch principale)
Root Directory: (lascia VUOTO)
```

### 3.4 Configurazione Docker

**IMPORTANTE**: Configura così:

```
Environment: Docker
Dockerfile Path: ./Dockerfile.automation
Docker Context: .
```

**Spiegazione**:
- **Environment**: Seleziona "Docker" dal menu a tendina
- **Dockerfile Path**: Scrivi esattamente `./Dockerfile.automation`
- **Docker Context**: Scrivi `.` (punto)

### 3.5 Seleziona Plan

1. Nella sezione **"Plan"**, seleziona:
   - **"Free"** - Gratis, ma va in sleep dopo 15 min
   - **"Starter ($7/mese)"** - Sempre attivo, no sleep

**Per iniziare**: Scegli **"Free"** per testare. Puoi sempre fare upgrade dopo.

### 3.6 Configura Variabili Ambiente

**QUESTO È FONDAMENTALE!** Aggiungi queste variabili:

Clicca su **"Advanced"** → **"Add Environment Variable"** e aggiungi:

1. **TELEGRAM_BOT_TOKEN**
   - Key: `TELEGRAM_BOT_TOKEN`
   - Value: `il_tuo_token_qui` (es: `8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g`)
   - ✅ Clicca "Add"

2. **TELEGRAM_CHAT_ID**
   - Key: `TELEGRAM_CHAT_ID`
   - Value: `il_tuo_chat_id_qui` (es: `-1003278011521`)
   - ✅ Clicca "Add"

3. **AUTOMATION_MIN_EV**
   - Key: `AUTOMATION_MIN_EV`
   - Value: `8.0`
   - ✅ Clicca "Add"

4. **AUTOMATION_MIN_CONFIDENCE**
   - Key: `AUTOMATION_MIN_CONFIDENCE`
   - Value: `70.0`
   - ✅ Clicca "Add"

5. **AUTOMATION_UPDATE_INTERVAL**
   - Key: `AUTOMATION_UPDATE_INTERVAL`
   - Value: `300`
   - ✅ Clicca "Add"

6. **PYTHONUNBUFFERED**
   - Key: `PYTHONUNBUFFERED`
   - Value: `1`
   - ✅ Clicca "Add"

**✅ Verifica**: Dovresti vedere 6 variabili ambiente nella lista.

---

## 🎯 STEP 4: Avvia il Deploy

### 4.1 Crea il Worker
1. Scorri in basso
2. Clicca sul pulsante blu **"Create Background Worker"**
3. ✅ Render inizierà il deploy!

### 4.2 Attendi il Build
- Il primo deploy può richiedere **5-15 minuti** (immagine 9 GB)
- Vedrai il progresso in tempo reale
- **NON chiudere la pagina!**

**Cosa vedrai**:
```
Building Docker image...
Installing dependencies...
Starting service...
```

---

## 🎯 STEP 5: Monitora i Log

### 5.1 Accedi ai Log
1. Dopo il deploy, vedrai la pagina del tuo worker
2. Clicca sulla tab **"Logs"** (in alto)
3. Vedrai i log in tempo reale

### 5.2 Cosa Cercare nei Log

**✅ Segni di Successo**:
```
✅ Telegram configurato
🚀 Avvio automazione 24/7...
Starting automation system...
```

**❌ Segni di Problemi**:
```
ERROR: TELEGRAM_BOT_TOKEN not found
ERROR: Cannot connect to Telegram
ModuleNotFoundError: ...
```

### 5.3 Se Vedi Errori

**Errore "TELEGRAM_BOT_TOKEN not found"**:
- Vai su: Worker → "Environment" tab
- Verifica che `TELEGRAM_BOT_TOKEN` sia presente
- Se manca, aggiungila e clicca "Save Changes"
- Render riavvierà automaticamente

**Errore "ModuleNotFoundError"**:
- Verifica che `requirements.automation.txt` contenga tutte le dipendenze
- Controlla i log per vedere quale modulo manca

**Errore "Build failed"**:
- Controlla che `Dockerfile.automation` esista nel repository
- Verifica che il path sia corretto: `./Dockerfile.automation`

---

## 🎯 STEP 6: Verifica Funzionamento

### 6.1 Controlla Telegram
1. Apri Telegram
2. Cerca il tuo bot (quello che hai creato con @BotFather)
3. Dovresti ricevere messaggi tipo:
   ```
   ✅ Sistema automazione avviato
   🔍 Monitoraggio partite attivo...
   ```

### 6.2 Controlla Log Render
1. Vai su Render Dashboard → Il tuo worker → "Logs"
2. Dovresti vedere output continuo tipo:
   ```
   [INFO] Checking for new matches...
   [INFO] Analyzing odds...
   [INFO] No value bets found
   ```

### 6.3 Test Manuale
1. Se tutto funziona, vedrai log ogni 5 minuti (o l'intervallo che hai configurato)
2. Se trovi value bets, riceverai notifiche Telegram

---

## 🎯 STEP 7: Configurazione Avanzata (Opzionale)

### 7.1 Cambiare da Free a Starter (Sempre Attivo)

Se vuoi che il worker sia sempre attivo (no sleep):

1. Vai su: Worker → "Settings" tab
2. Clicca su **"Change Plan"**
3. Seleziona **"Starter"** ($7/mese)
4. Clicca **"Update Plan"**
5. ✅ Ora è sempre attivo!

### 7.2 Modificare Variabili Ambiente

1. Vai su: Worker → "Environment" tab
2. Clicca su una variabile per modificarla
3. Cambia il valore
4. Clicca **"Save Changes"**
5. Render riavvierà automaticamente

### 7.3 Riavviare Manualmente

1. Vai su: Worker → "Manual Deploy" tab
2. Clicca **"Clear build cache & deploy"**
3. Render rifarà il deploy

---

## 📊 Monitoraggio e Manutenzione

### Dashboard Render
- **Overview**: Stato generale del worker
- **Logs**: Log in tempo reale
- **Metrics**: CPU, Memory, Network usage
- **Events**: Cronologia eventi (deploy, restart, etc.)

### Quando Controllare
- **Primi giorni**: Controlla log ogni giorno per verificare che tutto funzioni
- **Dopo modifiche**: Controlla log dopo ogni modifica al codice
- **Se non ricevi notifiche**: Controlla log per errori

---

## 🆘 Troubleshooting Completo

### Problema: "Service keeps restarting"

**Causa**: Errore nel codice o variabili ambiente mancanti

**Soluzione**:
1. Vai su "Logs"
2. Cerca l'ultimo errore prima del restart
3. Controlla variabili ambiente
4. Verifica che `start_automation.py` esista

### Problema: "No Telegram notifications"

**Causa**: Token o Chat ID errati

**Soluzione**:
1. Verifica token: Vai su @BotFather → `/mybots` → Seleziona bot → "API Token"
2. Verifica Chat ID: Vai su @userinfobot → Invia `/start` → Copia ID
3. Aggiorna variabili ambiente su Render
4. Riavvia worker

### Problema: "Build takes too long"

**Causa**: Immagine 9 GB richiede tempo

**Soluzione**:
- È normale! Il primo build può richiedere 10-15 minuti
- I build successivi saranno più veloci (cache)
- Se supera 20 minuti, controlla log per errori

### Problema: "Out of memory"

**Causa**: Free tier ha solo 512 MB RAM

**Soluzione**:
1. Upgrade a Starter plan ($7/mese) - 512 MB ma più stabile
2. Oppure ottimizza codice per usare meno memoria
3. Usa `Dockerfile.optimized` invece di `Dockerfile.automation`

### Problema: "Worker goes to sleep"

**Causa**: Free tier va in sleep dopo 15 min di inattività

**Soluzione**:
- È normale per free tier
- Si risveglia automaticamente quando necessario
- Per sempre attivo: Upgrade a Starter ($7/mese)

---

## ✅ Checklist Finale

Prima di considerare il deploy completato:

- [ ] Codice su GitHub
- [ ] Account Render creato
- [ ] Background Worker creato
- [ ] Dockerfile path corretto: `./Dockerfile.automation`
- [ ] Tutte le 6 variabili ambiente configurate
- [ ] Deploy completato senza errori
- [ ] Log mostrano "Sistema avviato"
- [ ] Notifiche Telegram ricevute
- [ ] Sistema funziona correttamente

---

## 🎉 Fatto!

Il tuo sistema ora gira su Render.com 24/7!

### Prossimi Passi:
1. **Monitora** i log per i primi giorni
2. **Verifica** che le notifiche arrivino correttamente
3. **Ottimizza** se necessario (riduci dimensione immagine per usare Railway)
4. **Upgrade** a Starter se vuoi sempre attivo senza sleep

### Link Utili:
- **Render Dashboard**: https://dashboard.render.com
- **Documentazione Render**: https://render.com/docs
- **Support Render**: https://render.com/docs/support

---

## 💡 Tips Finali

1. **Salva i log**: Render mantiene solo gli ultimi log, salva quelli importanti
2. **Backup config**: Salva le variabili ambiente in un file sicuro
3. **Monitora costi**: Free tier è gratis, ma controlla se fai upgrade
4. **Test locale**: Prima di modificare, testa sempre localmente
5. **Version control**: Usa sempre git per tracciare modifiche

**Buon deploy! 🚀**






