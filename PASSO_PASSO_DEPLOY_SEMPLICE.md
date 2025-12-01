# 🚀 Guida Passo-Passo: Sposta il Software su Server Cloud

## 📋 Cosa Farà Questa Guida

Ti insegno come spostare il tuo software `automation_24h.py` su un server cloud (Render.com) in modo che:
- ✅ Il software giri 24/7 anche con il PC spento
- ✅ Continui a ricevere notifiche Telegram
- ✅ Non devi più tenere acceso il PC

---

## 🎯 PREREQUISITI (Cosa ti serve)

Prima di iniziare, assicurati di avere:
1. ✅ Un account GitHub (gratis) - [https://github.com/signup](https://github.com/signup)
2. ✅ Un account Telegram (ce l'hai già)
3. ✅ Il tuo Token Telegram Bot (se non ce l'hai, vedi sotto come ottenerlo)

---

## 📝 STEP 1: Prepara il Codice su GitHub

### Opzione A: Usa GitHub Desktop (PIÙ FACILE) ⭐

1. **Scarica GitHub Desktop:**
   - Vai su: https://desktop.github.com/
   - Clicca "Download for Windows"
   - Installa il programma

2. **Apri GitHub Desktop:**
   - Clicca "Sign in to GitHub.com"
   - Accedi con il tuo account GitHub

3. **Aggiungi il tuo progetto:**
   - File → Add Local Repository
   - Clicca "Choose..." e seleziona la cartella:
     `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main`
   - Clicca "Add repository"

4. **Pubblica su GitHub:**
   - In alto, clicca "Publish repository"
   - Scegli un nome (es: `automation-24h`)
   - Spunta "Keep this code private" (se vuoi che sia privato)
   - Clicca "Publish repository"

5. ✅ **Fatto!** Il tuo codice è ora su GitHub

### Opzione B: Usa Git da Terminale

Se preferisci usare il terminale:

```powershell
# Apri PowerShell nella cartella del progetto
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"

# Se non hai ancora Git installato, installalo da:
# https://git-scm.com/download/win

# Inizializza repository Git
git init

# Aggiungi tutti i file
git add .

# Crea commit
git commit -m "Ready for deployment"

# Crea un nuovo repository su GitHub:
# 1. Vai su https://github.com/new
# 2. Scegli un nome (es: automation-24h)
# 3. NON inizializzare con README
# 4. Clicca "Create repository"

# Collega il repository (sostituisci TUO_USERNAME e NOME_REPO)
git remote add origin https://github.com/TUO_USERNAME/NOME_REPO.git
git branch -M main
git push -u origin main
```

---

## 📝 STEP 2: Ottieni Token e Chat ID Telegram

### Ottieni Token Telegram Bot:

1. Apri Telegram
2. Cerca **@BotFather**
3. Clicca "Start"
4. Scrivi: `/newbot`
5. Scegli un nome per il bot (es: "My Automation Bot")
6. Scegli un username (deve finire con "bot", es: "my_automation_bot")
7. BotFather ti darà un **TOKEN** (un codice lungo tipo: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
8. **COPIA E SALVA QUESTO TOKEN** (ti serve dopo!)

### Ottieni Chat ID:

1. Apri Telegram
2. Cerca **@userinfobot**
3. Clicca "Start"
4. Ti darà un numero (il tuo Chat ID)
5. **COPIA E SALVA QUESTO NUMERO** (ti serve dopo!)

---

## 📝 STEP 3: Crea Account Render.com

1. Vai su: **https://render.com**
2. Clicca **"Get Started for Free"** (in alto a destra)
3. Scegli **"Log in with GitHub"**
4. Autorizza Render.com ad accedere al tuo GitHub
5. ✅ **Fatto!** Hai creato l'account

---

## 📝 STEP 4: Crea Background Worker su Render

1. **Vai sulla Dashboard Render:**
   - Dopo il login, vedrai la dashboard

2. **Crea nuovo Background Worker:**
   - Clicca il pulsante **"New +"** (in alto a destra, blu)
   - Seleziona **"Background Worker"**

3. **Connetti il repository GitHub:**
   - Se non vedi il tuo repository, clicca **"Configure account"**
   - Autorizza Render.com ad accedere ai tuoi repository GitHub
   - Seleziona il repository che hai creato prima (es: `automation-24h`)

4. **Configura il Worker:**
   - **Name**: `automation-24h` (o un nome che preferisci)
   - **Region**: Scegli quello più vicino (es: Frankfurt per Europa)
   - **Branch**: `main` (o il branch principale)
   - **Root Directory**: (lascia vuoto)
   - **Environment**: Seleziona **"Docker"**
   - **Dockerfile Path**: `./Dockerfile.automation`
   - **Docker Context**: `.` (punto)

5. **IMPORTANTE: Aggiungi Variabili Ambiente**

   Clicca su **"Advanced"** per vedere tutte le opzioni.
   
   Nella sezione **"Environment Variables"**, clicca **"Add Environment Variable"** e aggiungi queste **6 variabili** una per una:

   ```
   Nome: TELEGRAM_BOT_TOKEN
   Valore: [incolla qui il TOKEN che hai copiato da BotFather]
   
   Nome: TELEGRAM_CHAT_ID
   Valore: [incolla qui il CHAT ID che hai copiato da @userinfobot]
   
   Nome: AUTOMATION_MIN_EV
   Valore: 8.0
   
   Nome: AUTOMATION_MIN_CONFIDENCE
   Valore: 70.0
   
   Nome: AUTOMATION_UPDATE_INTERVAL
   Valore: 300
   
   Nome: PYTHONUNBUFFERED
   Valore: 1
   ```

   ⚠️ **ATTENZIONE**: Assicurati di copiare esattamente i nomi delle variabili (maiuscole/minuscole importanti!)

6. **Scegli il Piano:**
   - **Free** - Gratis ma va in sleep dopo 15 minuti (si risveglia automaticamente)
   - **Starter ($7/mese)** - Sempre attivo 24/7
   
   Per iniziare, scegli **Free**. Puoi sempre cambiare dopo.

7. **Clicca "Create Background Worker"**

8. ✅ **Render inizia il deploy automaticamente!**

---

## 📝 STEP 5: Aspetta che Finisca il Deploy

1. Render mostrerà una pagina con i log del deploy
2. Aspetta che finisca (può richiedere 5-10 minuti)
3. Vedrai messaggi come:
   - "Building..." (sta costruendo l'immagine Docker)
   - "Deploying..." (sta deployando)
   - "Running" (è in esecuzione!)

4. **Controlla i Log:**
   - Clicca sulla tab **"Logs"** in alto
   - Dovresti vedere messaggi come:
     - ✅ "Sistema avviato"
     - ✅ "Automation24H started"
     - ✅ "Telegram configurato"

5. ✅ **Se vedi questi messaggi, funziona!**

---

## 📝 STEP 6: Verifica che Funzioni

1. **Controlla Telegram:**
   - Il sistema dovrebbe inviare una notifica di test all'avvio
   - Se ricevi una notifica su Telegram, ✅ **funziona perfettamente!**

2. **Controlla i Log Render:**
   - Vai su Dashboard → Il tuo worker → "Logs"
   - Dovresti vedere attività continue
   - Se vedi errori, vedi la sezione "Problemi?" sotto

3. **Prova a Spegnere il PC:**
   - Una volta verificato che funziona, spegni tranquillamente il PC
   - Il software continua a girare su Render! 🎉

---

## ✅ RIEPILOGO: Checklist Finale

Prima di considerare tutto fatto, verifica:

- [ ] Codice pubblicato su GitHub
- [ ] Account Render.com creato
- [ ] Background Worker creato su Render
- [ ] 6 variabili ambiente configurate correttamente
- [ ] Deploy completato (stato "Running")
- [ ] Log mostrano "Sistema avviato"
- [ ] Notifica Telegram ricevuta
- [ ] PC spento e software continua a girare!

---

## 🆘 PROBLEMI COMUNI E SOLUZIONI

### ❌ "Build Failed" / "Docker Error"

**Cosa significa:** Il deploy non è riuscito

**Soluzione:**
1. Vai su Dashboard → Il tuo worker → "Logs"
2. Guarda gli ultimi messaggi di errore
3. Controlla che `Dockerfile.automation` esista nel repository GitHub
4. Verifica che tutti i file siano stati pushati su GitHub

---

### ❌ "Worker Keeps Restarting" / "Worker Crashed"

**Cosa significa:** Il worker continua a riavviarsi

**Soluzione:**
1. Controlla i log per vedere l'errore esatto
2. Verifica che tutte le **6 variabili ambiente** siano configurate:
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - AUTOMATION_MIN_EV
   - AUTOMATION_MIN_CONFIDENCE
   - AUTOMATION_UPDATE_INTERVAL
   - PYTHONUNBUFFERED
3. Assicurati che i valori non abbiano spazi extra o virgolette

---

### ❌ "No Telegram Notifications"

**Cosa significa:** Non ricevi notifiche Telegram

**Soluzione:**
1. Verifica `TELEGRAM_BOT_TOKEN` su Render:
   - Dashboard → Worker → Environment → TELEGRAM_BOT_TOKEN
   - Assicurati che sia corretto (senza spazi, senza virgolette)
2. Verifica `TELEGRAM_CHAT_ID`:
   - Deve essere un numero (es: `123456789`)
   - Se è negativo (es: `-1001234567890`), è corretto per i canali
3. Testa il bot manualmente:
   - Vai su Telegram → Cerca il tuo bot → Scrivi `/start`
   - Se risponde, il token è corretto
4. Controlla i log Render per errori di connessione

---

### ❌ "Worker Goes to Sleep" (Free Tier)

**Cosa significa:** Il worker va in sleep dopo 15 minuti

**Questo è normale con il piano Free!**

**Cosa fare:**
- **Opzione 1:** Accetta il sleep (si risveglia automaticamente quando necessario)
- **Opzione 2:** Passa a piano Starter ($7/mese) per sempre attivo:
  - Dashboard → Worker → Settings → Plan → Change to Starter

---

## 💰 COSTI

### Render.com Free Tier:
- ✅ **Gratis per sempre**
- ⚠️ Va in sleep dopo 15 minuti di inattività
- ✅ Si risveglia automaticamente

### Render.com Starter ($7/mese):
- ✅ Sempre attivo 24/7
- ✅ Nessun sleep
- ✅ 512 MB RAM
- ✅ Monitoraggio completo

**Consiglio:** Inizia con **Free**, poi passa a **Starter** se vedi che serve sempre attivo.

---

## 🔄 Come Aggiornare il Codice

Quando modifichi il codice e vuoi aggiornare il server:

1. **Salva le modifiche su GitHub:**
   - Se usi GitHub Desktop: Clicca "Commit to main" → "Push origin"
   - Se usi Git: `git add . && git commit -m "Update" && git push`

2. **Render rileva automaticamente il push** e fa il deploy automatico!
   - Vai su Dashboard → Worker → "Events"
   - Vedrai "Deploy in progress"
   - Aspetta che finisca

3. ✅ Il nuovo codice è live!

---

## 📱 Come Monitorare da Remoto

Dopo il deploy, puoi:

- ✅ **Vedere i Log:** Dashboard Render → Worker → Logs
- ✅ **Ricevere Notifiche:** Telegram (come sempre)
- ✅ **Modificare Impostazioni:** Dashboard Render → Worker → Environment
- ✅ **Riavviare:** Dashboard Render → Worker → Manual Deploy
- ✅ **Vedere Statistiche:** Dashboard Render → Worker → Metrics

---

## 🎉 FINE!

Dopo aver seguito questi passi:

1. ✅ Il tuo software gira su Render.com 24/7
2. ✅ Puoi spegnere il PC tranquillamente
3. ✅ Continui a ricevere notifiche Telegram
4. ✅ Puoi monitorare tutto da remoto

**Non serve più tenere acceso il PC!** 🚀

---

## 🆘 Hai Ancora Problemi?

1. Controlla i log Render per errori specifici
2. Verifica che tutte le variabili ambiente siano corrette
3. Assicurati che tutti i file siano su GitHub
4. Controlla che `Dockerfile.automation` esista

Se hai ancora problemi, descrivi l'errore che vedi nei log Render.

---

**Buon deploy! 🎉**










