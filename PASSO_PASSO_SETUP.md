# 📋 GUIDA PASSO-PASSO - Setup Completo

## 🎯 Cosa Faremo

1. ✅ Configurare Telegram Bot
2. ✅ Pushare codice su GitHub
3. ✅ Deployare su Railway.app (gratis, sempre attivo)
4. ✅ Testare che funzioni

**Tempo totale: ~15 minuti**

---

## 📱 PASSO 1: Configurare Telegram Bot (5 min)

### 1.1 Crea il Bot

1. Apri Telegram sul telefono/PC
2. Cerca: **@BotFather**
3. Invia: `/start`
4. Invia: `/newbot`
5. Scegli un nome per il bot (es: "AsianOdds Bot")
6. Scegli un username (deve finire con "bot", es: "asianodds_bot")
7. **COPIA IL TOKEN** che ti dà (es: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 1.2 Ottieni il Tuo Chat ID

1. Cerca su Telegram: **@userinfobot**
2. Invia: `/start`
3. **COPIA IL TUO ID** (es: `123456789`)

### 1.3 Testa il Bot

1. Cerca il tuo bot su Telegram (con il nome che hai scelto)
2. Invia: `/start`
3. Dovresti ricevere un messaggio

**✅ Fatto! Hai:**
- Token del bot: `_________________`
- Chat ID: `_________________`

---

## 💻 PASSO 2: Pushare Codice su GitHub (5 min)

### 2.1 Apri PowerShell

1. Premi `Windows + X`
2. Seleziona **"Windows PowerShell"** o **"Terminal"**

### 2.2 Vai nella Cartella del Progetto

```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
```

### 2.3 Verifica che Git Funzioni

```powershell
git --version
```

Se vedi un numero di versione (es: `git version 2.51.2`), va bene! ✅

### 2.4 Aggiungi Tutti i File

```powershell
git add .
```

### 2.5 Crea Commit

```powershell
git commit -m "Add automation 24/7 system"
```

### 2.6 Pusha su GitHub

```powershell
git push origin master
```

**Se ti chiede username/password:**
- Username: il tuo username GitHub
- Password: usa un **Personal Access Token** (non la password normale)
  - Come crearlo: https://github.com/settings/tokens
  - Clicca "Generate new token (classic)"
  - Seleziona permessi `repo`
  - Copia il token e usalo come password

**✅ Fatto! Codice su GitHub!**

---

## 🚀 PASSO 3: Deployare su Railway.app (5 min)

### 3.1 Crea Account Railway

1. Vai su: **https://railway.app**
2. Clicca **"Start a New Project"** o **"Login"**
3. Scegli **"Login with GitHub"**
4. Autorizza Railway ad accedere a GitHub

### 3.2 Crea Nuovo Progetto

1. Clicca **"New Project"**
2. Seleziona **"Deploy from GitHub repo"**
3. Se non vedi il repository, clicca **"Configure GitHub App"** e autorizza
4. Seleziona il repository: **Software-AsianOdds**

### 3.3 Railway Inizia il Deploy

Railway rileva automaticamente il `Dockerfile.automation` e inizia a deployare.

**Aspetta 2-3 minuti** mentre Railway:
- Installa dipendenze
- Costruisce il container
- Avvia il servizio

### 3.4 Configura Variabili Ambiente

1. Nella dashboard Railway, clicca sul **servizio** (dovrebbe chiamarsi "automation-24h" o simile)
2. Vai su **"Variables"** (tab in alto)
3. Clicca **"New Variable"**
4. Aggiungi queste variabili **UNA PER UNA**:

**Variabile 1:**
- Key: `TELEGRAM_BOT_TOKEN`
- Value: `[incolla il token del bot che hai copiato al passo 1.1]`
- Clicca **"Add"**

**Variabile 2:**
- Key: `TELEGRAM_CHAT_ID`
- Value: `[incolla il tuo chat ID che hai copiato al passo 1.2]`
- Clicca **"Add"**

**Variabile 3 (opzionale):**
- Key: `AUTOMATION_MIN_EV`
- Value: `8.0`
- Clicca **"Add"**

**Variabile 4 (opzionale):**
- Key: `AUTOMATION_MIN_CONFIDENCE`
- Value: `70.0`
- Clicca **"Add"**

**Variabile 5 (opzionale):**
- Key: `AUTOMATION_UPDATE_INTERVAL`
- Value: `300`
- Clicca **"Add"**

### 3.5 Riavvia il Servizio

1. Vai su **"Settings"** (tab in alto)
2. Clicca **"Restart"** (in fondo)
3. Aspetta che riavvii (30 secondi)

### 3.6 Verifica Log

1. Vai su **"Logs"** (tab in alto)
2. Dovresti vedere output tipo:
   ```
   ✅ AI Pipeline initialized
   ✅ Telegram Notifier initialized
   🚀 Starting Automation24H system...
   🔄 Running analysis cycle...
   ```

**✅ Fatto! Sistema deployato e attivo!**

---

## ✅ PASSO 4: Verificare che Funzioni (2 min)

### 4.1 Controlla Log Railway

1. Vai su Railway → **Logs**
2. Dovresti vedere messaggi ogni 5 minuti tipo:
   ```
   🔄 Running analysis cycle...
   Found X matches to monitor
   ✅ Cycle complete
   ```

### 4.2 Testa Notifica Telegram

Il sistema invierà notifiche automaticamente quando trova opportunità.

**Per testare subito**, puoi:
1. Aspettare che trovi un'opportunità (può richiedere tempo)
2. Oppure modificare temporaneamente `min_ev` a `1.0` per vedere più notifiche

### 4.3 Verifica che il Sistema Giri

1. Vai su Railway → **Metrics**
2. Dovresti vedere CPU/Memory usage
3. Se vedi attività, il sistema sta girando! ✅

---

## 🎉 FATTO!

Il sistema ora:
- ✅ Gira 24/7 su Railway (gratis)
- ✅ Analizza partite automaticamente
- ✅ Ti invia notifiche Telegram quando trova opportunità
- ✅ Non consiglia basandosi su score (solo value bet reali)

---

## ❓ Problemi Comuni

### "Deploy failed" su Railway

**Soluzione:**
1. Vai su **Logs** e leggi l'errore
2. Verifica che `requirements.txt` esista
3. Verifica che `Dockerfile.automation` esista
4. Prova a riavviare il deploy

### "No notifications" su Telegram

**Soluzione:**
1. Verifica che il token sia corretto (senza spazi)
2. Verifica che il chat ID sia corretto
3. Invia `/start` al bot su Telegram
4. Controlla log Railway per errori

### "Service keeps restarting"

**Soluzione:**
1. Vai su **Logs** e leggi l'errore
2. Verifica che tutte le variabili ambiente siano configurate
3. Verifica che il token Telegram sia valido

### "Git push failed"

**Soluzione:**
1. Verifica di essere loggato: `git config user.name`
2. Se non sei loggato:
   ```powershell
   git config --global user.name "Tuo Nome"
   git config --global user.email "tua@email.com"
   ```
3. Per la password, usa un Personal Access Token (vedi passo 2.6)

---

## 📞 Supporto

Se hai problemi:
1. Controlla i **Log** su Railway
2. Controlla i **Log** su GitHub Actions (se usi quello)
3. Verifica che tutte le variabili siano configurate
4. Rileggi questa guida passo-passo

---

## 🎯 Checklist Finale

- [ ] Telegram bot creato e token copiato
- [ ] Chat ID ottenuto
- [ ] Codice pushato su GitHub
- [ ] Account Railway creato
- [ ] Progetto deployato su Railway
- [ ] Variabili ambiente configurate
- [ ] Servizio riavviato
- [ ] Log verificati (nessun errore)
- [ ] Sistema attivo e funzionante

**🎊 Se tutti i checkbox sono ✅, sei pronto! Il sistema gira 24/7!**

