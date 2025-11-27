# 🚀 Guida Completa: Deploy su Server Cloud 24/7

## 📋 Obiettivo

Spostare il software `automation_24h.py` su un server cloud che gira 24/7, così puoi:
- ✅ Spegnere il PC senza problemi
- ✅ Il software continua a funzionare
- ✅ Ricevere notifiche Telegram normalmente
- ✅ Monitorare da qualsiasi dispositivo

---

## 🎯 Opzioni Disponibili

### Opzione 1: Render.com (⭐ CONSIGLIATO)

**Vantaggi:**
- ✅ Gratuito (con piano free)
- ✅ Supporta immagini fino a 10 GB
- ✅ Background Worker sempre attivo (gira 24/7)
- ✅ Facile da configurare
- ✅ Deploy automatico da GitHub
- ✅ Monitoraggio integrato

**Svantaggi:**
- ⚠️ Con piano gratuito, va in sleep dopo 15 min di inattività (ma si risveglia automaticamente)

**Costo:** Gratis (con limitazioni) o $7/mese (sempre attivo)

---

### Opzione 2: Railway.app

**Vantaggi:**
- ✅ Sempre attivo con piano gratuito
- ✅ Facile da usare
- ✅ Deploy automatico da GitHub

**Svantaggi:**
- ⚠️ Limite 4 GB per immagini Docker
- ⚠️ $5/mese dopo uso gratuito iniziale

**Costo:** Gratis per primi giorni, poi $5/mese

---

### Opzione 3: Fly.io

**Vantaggi:**
- ✅ Free tier generoso
- ✅ Sempre attivo

**Svantaggi:**
- ⚠️ 256 MB RAM (potrebbe essere poco)
- ⚠️ Più complesso da configurare

**Costo:** Gratis (con limitazioni)

---

## 🚀 Deploy su Render.com (Guida Passo-Passo)

### PREREQUISITI

Prima di iniziare, devi avere:
1. ✅ Codice su GitHub (se non ce l'hai, vedi "Setup GitHub" sotto)
2. ✅ Account Render.com (gratuito)
3. ✅ Token Telegram Bot
4. ✅ Chat ID Telegram

---

### STEP 1: Prepara il Codice su GitHub

Se il tuo codice NON è ancora su GitHub:

#### Opzione A: Usa GitHub Desktop (PIÙ FACILE)

1. **Installa GitHub Desktop:**
   - Vai su: https://desktop.github.com/
   - Scarica e installa GitHub Desktop

2. **Pubblica su GitHub:**
   - Apri GitHub Desktop
   - File → Add Local Repository
   - Seleziona la cartella `Software-AsianOdds-main`
   - Clicca "Publish repository" in alto
   - Scegli nome (es: `automation-24h`)
   - Spunta "Keep this code private" (se vuoi privato)
   - Clicca "Publish repository"

3. ✅ Fatto! Il codice è su GitHub

#### Opzione B: Usa Git da Terminale

```powershell
# Apri PowerShell nella cartella Software-AsianOdds-main
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"

# Inizializza repository Git (se non già fatto)
git init

# Aggiungi tutti i file
git add .

# Crea commit
git commit -m "Ready for deployment"

# Crea repository su GitHub (vai su https://github.com/new)
# Poi collega:
git remote add origin https://github.com/TUO_USERNAME/nome-repository.git
git branch -M main
git push -u origin main
```

---

### STEP 2: Crea Account Render.com

1. Vai su: **https://render.com**
2. Clicca **"Get Started for Free"**
3. Registrati con **GitHub** (più facile, usa lo stesso account)
4. ✅ Account creato!

---

### STEP 3: Crea Background Worker su Render

1. **Vai su Dashboard Render.com**
2. Clicca **"New +"** (in alto a destra)
3. Seleziona **"Background Worker"**
4. Connetti il tuo repository GitHub:
   - Se non vedi il repository, clicca **"Configure account"**
   - Autorizza Render.com ad accedere ai tuoi repository
   - Seleziona il repository che hai creato

5. **Configura il Worker:**
   ```
   Name: automation-24h
   Region: Frankfurt (o quello più vicino a te)
   Branch: main (o il tuo branch principale)
   Root Directory: (lascia vuoto)
   Environment: Docker
   Dockerfile Path: ./Dockerfile.automation
   Docker Context: .
   ```

6. **IMPORTANTE: Variabili Ambiente**

   Clicca su **"Advanced"** e aggiungi queste variabili:

   ```
   TELEGRAM_BOT_TOKEN=il_tuo_token_telegram_qui
   TELEGRAM_CHAT_ID=il_tuo_chat_id_qui
   AUTOMATION_MIN_EV=8.0
   AUTOMATION_MIN_CONFIDENCE=70.0
   AUTOMATION_UPDATE_INTERVAL=300
   PYTHONUNBUFFERED=1
   ```

   **Come ottenere Token e Chat ID:**
   - **Token**: Vai su Telegram → @BotFather → `/newbot` → Segui istruzioni → Copia il token
   - **Chat ID**: Vai su Telegram → @userinfobot → `/start` → Copia il numero ID

7. **Plan:**
   - Seleziona **"Free"** (per iniziare, puoi sempre cambiare dopo)
   - ⚠️ Il free tier va in sleep dopo 15 min, ma si risveglia automaticamente
   - Per sempre attivo: Seleziona **"Starter" ($7/mese)**

8. **Clicca "Create Background Worker"**

9. ✅ Render inizia a fare il deploy automaticamente!

---

### STEP 4: Monitora il Deploy

1. Vai su **"Logs"** nella pagina del worker
2. Vedrai i log in tempo reale del build e dell'avvio
3. Aspetta che finisca il build (può richiedere 5-10 minuti)
4. Cerca messaggi come:
   - ✅ "Sistema avviato"
   - ✅ "Automation24H started"
   - ❌ Se vedi errori, vedi "Troubleshooting" sotto

---

### STEP 5: Verifica che Funzioni

1. **Controlla Log Render:**
   - Vai su Dashboard → Il tuo worker → "Logs"
   - Dovresti vedere messaggi di avvio e attività

2. **Ricevi Notifica Telegram:**
   - Il sistema dovrebbe inviare una notifica di test all'avvio
   - Se ricevi una notifica, ✅ funziona!

3. **Spegni il PC:**
   - Una volta verificato che funziona, puoi spegnere tranquillamente il PC
   - Il software continua a girare su Render! 🎉

---

## 🔧 Configurazioni Opzionali

### Modificare Impostazioni

Puoi modificare le impostazioni modificando le variabili ambiente su Render:

1. Vai su Dashboard → Il tuo worker → "Environment"
2. Modifica le variabili:
   ```
   AUTOMATION_MIN_EV=10.0          # EV minimo (default: 8.0)
   AUTOMATION_MIN_CONFIDENCE=75.0  # Confidence minima (default: 70.0)
   AUTOMATION_UPDATE_INTERVAL=600  # Intervallo in secondi (default: 300 = 5 min)
   ```
3. Clicca "Save Changes" → Render riavvia automaticamente

---

## 📊 Monitoraggio

### Log Render

- Vai su Dashboard → Worker → "Logs"
- Vedi tutto in tempo reale
- Puoi scaricare i log

### Notifiche Telegram

- Ricevi notifiche quando trova opportunità
- Ricevi notifiche di errore (se configurate)

---

## 🆘 Troubleshooting

### "Build Failed" / "Docker Error"

**Problema:** Il build fallisce

**Soluzione:**
1. Controlla che `Dockerfile.automation` esista nella root del repository
2. Controlla i log per errori specifici
3. Verifica che tutti i file necessari siano committati su GitHub

---

### "Worker Keeps Restarting"

**Problema:** Il worker continua a riavviarsi

**Soluzione:**
1. Controlla i log per errori
2. Verifica che tutte le variabili ambiente siano configurate:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Verifica che i valori siano corretti (no spazi, no virgolette extra)

---

### "No Telegram Notifications"

**Problema:** Non ricevi notifiche Telegram

**Soluzione:**
1. Verifica `TELEGRAM_BOT_TOKEN` su Render
2. Verifica `TELEGRAM_CHAT_ID` su Render
3. Testa il bot manualmente:
   - Vai su Telegram → Il tuo bot → Scrivi `/start`
   - Se risponde, il token è corretto
4. Controlla i log Render per errori di connessione Telegram

---

### "Worker Goes to Sleep" (Free Tier)

**Problema:** Il worker va in sleep dopo 15 min

**Soluzione:**
1. **Opzione 1:** Accetta il sleep (si risveglia automaticamente)
2. **Opzione 2:** Passa a piano Starter ($7/mese) per sempre attivo
3. **Opzione 3:** Configura un "Health Check" che tiene sveglio il worker:
   - Render → Worker → Settings → Health Check Path: `/health`
   - Ma questo richiede modifiche al codice

---

### "Out of Memory" / "Crashed"

**Problema:** Il worker va in crash per memoria

**Soluzione:**
1. Passa a piano più grande su Render (Starter o Standard)
2. Oppure ottimizza il Dockerfile (vedi sotto)

---

## 🔄 Aggiornare il Codice

Quando modifichi il codice e vuoi aggiornare il server:

1. **Committa le modifiche su GitHub:**
   ```powershell
   git add .
   git commit -m "Aggiornamenti"
   git push
   ```

2. **Render rileva automaticamente il push** e fa il deploy automatico!
   - Vai su Dashboard → Worker → "Events"
   - Vedrai "Deploy in progress"
   - Aspetta che finisca

3. ✅ Il nuovo codice è live!

---

## 💰 Costi

### Render.com Free Tier:
- ✅ Gratis
- ⚠️ Va in sleep dopo 15 min di inattività
- ✅ Si risveglia automaticamente

### Render.com Starter ($7/mese):
- ✅ Sempre attivo 24/7
- ✅ Nessun sleep
- ✅ 512 MB RAM
- ✅ 0.1 CPU

### Render.com Standard ($25/mese):
- ✅ Sempre attivo
- ✅ 2 GB RAM
- ✅ 0.5 CPU
- ✅ Più veloce

**Raccomandazione:** Inizia con Free, poi passa a Starter se serve sempre attivo.

---

## 🎯 Alternative: Railway.app

Se preferisci Railway (più semplice ma limite 4 GB):

### Setup Railway:

1. Vai su: **https://railway.app**
2. Crea account con GitHub
3. **New Project** → **Deploy from GitHub repo**
4. Seleziona il tuo repository
5. Railway rileva automaticamente il `Dockerfile.automation`
6. Aggiungi variabili ambiente (stesso di Render)
7. ✅ Deploy automatico!

**Costo:** Gratis per primi giorni, poi $5/mese

---

## ✅ Checklist Finale

Prima di considerare tutto completato:

- [ ] Codice su GitHub
- [ ] Account Render.com creato
- [ ] Background Worker creato
- [ ] Variabili ambiente configurate
- [ ] Deploy completato
- [ ] Log mostrano "Sistema avviato"
- [ ] Notifica Telegram ricevuta
- [ ] PC spento e software continua a girare! 🎉

---

## 📱 Accesso Remoto

Dopo il deploy, puoi:

- ✅ **Monitorare Log:** Dashboard Render → Worker → Logs
- ✅ **Ricevere Notifiche:** Telegram (come sempre)
- ✅ **Modificare Impostazioni:** Dashboard Render → Worker → Environment
- ✅ **Riavviare:** Dashboard Render → Worker → Manual Deploy

---

## 🎉 Conclusioni

Dopo aver seguito questa guida:

1. ✅ Il tuo software gira su Render.com 24/7
2. ✅ Puoi spegnere il PC tranquillamente
3. ✅ Continui a ricevere notifiche Telegram
4. ✅ Puoi monitorare tutto da remoto

**Non serve più tenere acceso il PC!** 🚀

---

## 🆘 Supporto

Se hai problemi:

1. Controlla i log Render per errori
2. Verifica le variabili ambiente
3. Controlla che tutti i file siano su GitHub
4. Assicurati che `Dockerfile.automation` esista

---

**Buon deploy! 🚀**










