# 🚀 GUIDA COMPLETA DEPLOY SU RENDER.COM

## 📋 INDICE
1. [Preparazione Account](#1-preparazione-account)
2. [Preparazione Repository GitHub](#2-preparazione-repository-github)
3. [Creazione Service su Render](#3-creazione-service-su-render)
4. [Configurazione Variabili d'Ambiente](#4-configurazione-variabili-dambiente)
5. [Deploy e Verifica](#5-deploy-e-verifica)
6. [Monitoraggio e Troubleshooting](#6-monitoraggio-e-troubleshooting)

---

## 1. PREPARAZIONE ACCOUNT

### 1.1 Crea Account Render
1. Vai su **https://render.com**
2. Clicca su **"Get Started for Free"** o **"Sign Up"**
3. Scegli **"Sign up with GitHub"** (consigliato) oppure usa email
4. Completa la registrazione

### 1.2 Scegli Piano
1. Vai su **Dashboard** → **Account Settings** → **Billing**
2. Scegli piano:
   - **Starter Plan ($7/mese)**: Consigliato per iniziare
   - **Standard Plan ($25/mese)**: Se hai bisogno di più risorse
3. Inserisci metodo di pagamento (carta di credito)

⚠️ **IMPORTANTE**: Il Free Tier si spegne dopo 15 minuti di inattività, NON va bene per 24/7!

---

## 2. PREPARAZIONE REPOSITORY GITHUB

### 2.1 Verifica Repository
Assicurati che il codice sia su GitHub:
1. Vai su **https://github.com**
2. Verifica che il repository esista e sia pubblico (o privato se hai Render Pro)
3. Verifica che il branch principale sia aggiornato

### 2.2 File Necessari (già presenti)
✅ `render.yaml` - Configurazione Render
✅ `Dockerfile.automation` - Immagine Docker
✅ `requirements.automation.txt` - Dipendenze Python
✅ `start_automation.py` - Script di avvio

---

## 3. CREAZIONE SERVICE SU RENDER

### 3.1 Accedi al Dashboard
1. Vai su **https://dashboard.render.com**
2. Clicca su **"New +"** in alto a destra
3. Seleziona **"Background Worker"** (NON Web Service!)

### 3.2 Connetti Repository GitHub

**PASSO 1: Connect Repository**
- **Name**: `automation-24h` (o qualsiasi nome tu preferisca)
- **Repository**: Clicca su **"Connect account"** se non hai ancora connesso GitHub
- Seleziona il repository: `Sib-asian/Software-AsianOdds` (o il tuo)
- **Branch**: `main` (o il branch che vuoi usare)

**PASSO 2: Build & Deploy Settings**

**Environment**: 
- Seleziona **"Docker"** (NON "Python"!)

**Dockerfile Path**:
- Inserisci: `./Dockerfile.automation`
- ⚠️ IMPORTANTE: Deve essere esattamente così, con il punto e la barra

**Docker Context**:
- Inserisci: `.` (solo un punto)
- Questo significa "root del repository"

**Root Directory** (opzionale):
- Lascia vuoto (usa root del repository)

**Build Command** (opzionale):
- Lascia vuoto (Render usa automaticamente Docker)

**Start Command** (opzionale):
- Lascia vuoto (già definito nel Dockerfile: `CMD ["python", "start_automation.py"]`)

### 3.3 Plan & Region

**Plan**:
- Seleziona **"Starter"** ($7/mese) o **"Standard"** ($25/mese)
- ⚠️ NON selezionare "Free" - si spegne dopo 15 minuti!

**Region**:
- Scegli la regione più vicina a te (es. "Frankfurt" per Europa)
- Per l'Italia: **"Frankfurt"** o **"London"**

### 3.4 Advanced Settings (opzionale)

**Auto-Deploy**:
- ✅ **"Yes"** - Deploy automatico quando fai push su GitHub
- Oppure **"No"** se vuoi deploy manuale

**Health Check Path** (opzionale):
- Lascia vuoto (non necessario per worker)

**Dockerfile Path** (già inserito sopra):
- Verifica che sia: `./Dockerfile.automation`

---

## 4. CONFIGURAZIONE VARIABILI D'AMBIENTE

⚠️ **CRITICO**: Queste variabili DEVONO essere configurate prima del deploy!

### 4.1 Accedi alle Environment Variables

1. Nel service che hai creato, vai su **"Environment"** nel menu laterale
2. Clicca su **"Add Environment Variable"**

### 4.2 Variabili Obbligatorie

#### 🔑 TELEGRAM_BOT_TOKEN
- **Key**: `TELEGRAM_BOT_TOKEN`
- **Value**: Il tuo token del bot Telegram
  - Esempio: `8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g`
  - ⚠️ NON condividere questo token pubblicamente!
- **Sync**: Lascia deselezionato (non sincronizza tra servizi)

#### 🔑 TELEGRAM_CHAT_ID
- **Key**: `TELEGRAM_CHAT_ID`
- **Value**: Il tuo Chat ID Telegram
  - Esempio: `-1003278011521`
  - ⚠️ Deve essere il Chat ID del canale/gruppo dove vuoi ricevere notifiche
- **Sync**: Lascia deselezionato

#### 🔑 API_FOOTBALL_KEY (API-SPORTS)
- **Key**: `API_FOOTBALL_KEY`
- **Value**: La tua chiave API di API-SPORTS
  - Esempio: `abc123def456ghi789...`
  - ⚠️ Questa è la chiave per API-SPORTS (v3.football.api-sports.io)
- **Sync**: Lascia deselezionato

### 4.3 Variabili Opzionali (ma Consigliate)

#### ⚙️ AUTOMATION_MIN_EV
- **Key**: `AUTOMATION_MIN_EV`
- **Value**: `8.0`
- **Descrizione**: Valore minimo di Expected Value per inviare segnali
- **Default**: Se non impostato, usa 8.0

#### ⚙️ AUTOMATION_MIN_CONFIDENCE
- **Key**: `AUTOMATION_MIN_CONFIDENCE`
- **Value**: `70.0`
- **Descrizione**: Confidence minima (in percentuale) per inviare segnali
- **Default**: Se non impostato, usa 70.0

#### ⚙️ AUTOMATION_UPDATE_INTERVAL
- **Key**: `AUTOMATION_UPDATE_INTERVAL`
- **Value**: `600`
- **Descrizione**: Intervallo tra cicli in secondi (600 = 10 minuti)
- **Default**: Se non impostato, usa 300 (5 minuti)

#### ⚙️ THEODDS_API_KEY (Opzionale)
- **Key**: `THEODDS_API_KEY`
- **Value**: La tua chiave API di TheOddsAPI (se la usi)
- **Descrizione**: Per ottenere quote da TheOddsAPI
- **Default**: Se non impostato, usa mock data

#### ⚙️ NEWSAPI_KEY (Opzionale)
- **Key**: `NEWSAPI_KEY`
- **Value**: La tua chiave API di NewsAPI (se la usi)
- **Descrizione**: Per analisi sentiment delle news
- **Default**: Se non impostato, disabilita analisi news

### 4.4 Variabili di Sistema (Automatiche)

Queste vengono impostate automaticamente da Render, NON devi aggiungerle:
- `PYTHONUNBUFFERED=1` (già nel render.yaml)
- `PORT` (non necessario per worker)
- `RENDER` (variabile di sistema Render)

---

## 5. DEPLOY E VERIFICA

### 5.1 Avvia Deploy

1. Dopo aver configurato tutte le variabili, clicca su **"Create Background Worker"**
2. Render inizierà automaticamente il build:
   - **Build Log**: Puoi vedere il progresso in tempo reale
   - **Tempo stimato**: 5-10 minuti per la prima build

### 5.2 Monitora Build

**Cosa guardare nei log**:
- ✅ `Successfully built` - Build completato
- ✅ `Successfully tagged` - Immagine Docker creata
- ✅ `Starting container` - Container avviato
- ❌ Se vedi errori, controlla:
  - Variabili d'ambiente mancanti
  - File mancanti nel repository
  - Errori nel Dockerfile

### 5.3 Verifica Funzionamento

**Dopo il deploy**:
1. Vai su **"Logs"** nel menu del service
2. Cerca questi messaggi:
   - ✅ `✅ Telegram Notifier initialized`
   - ✅ `✅ Signal Quality Learner inizializzato`
   - ✅ `🚀 Avvio sistema automazione 24/7...`
   - ✅ `Running analysis cycle...`

**Test Telegram**:
- Dovresti ricevere una notifica su Telegram quando il sistema si avvia
- Se non ricevi notifiche, controlla:
  - `TELEGRAM_BOT_TOKEN` corretto
  - `TELEGRAM_CHAT_ID` corretto
  - Bot aggiunto al canale/gruppo

---

## 6. MONITORAGGIO E TROUBLESHOOTING

### 6.1 Logs

**Come vedere i log**:
1. Vai su **"Logs"** nel menu del service
2. I log sono in tempo reale
3. Puoi filtrare per livello (INFO, WARNING, ERROR)

**Cosa cercare**:
- ✅ `Cycle complete` - Ciclo completato con successo
- ✅ `Notified live opportunity` - Segnale inviato
- ⚠️ `API quota exhausted` - Quota API esaurita
- ❌ `Error in cycle` - Errore nel ciclo

### 6.2 Metrics

**Monitoraggio risorse**:
- **CPU Usage**: Dovrebbe essere bassa (< 20%)
- **Memory Usage**: Dovrebbe essere ~200-500 MB
- **Network**: Dovrebbe essere minima

### 6.3 Troubleshooting Comune

#### ❌ Problema: Build Fallisce
**Causa**: File mancanti o Dockerfile errato
**Soluzione**:
- Verifica che tutti i file siano nel repository
- Controlla il Dockerfile path: `./Dockerfile.automation`
- Verifica che il Dockerfile sia corretto

#### ❌ Problema: Service si Spegne
**Causa**: Piano Free (si spegne dopo 15 min)
**Soluzione**:
- Passa a Starter Plan ($7/mese)
- Verifica che il worker sia sempre attivo

#### ❌ Problema: Nessuna Notifica Telegram
**Causa**: Variabili d'ambiente errate
**Soluzione**:
- Verifica `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
- Controlla i log per errori Telegram
- Verifica che il bot sia aggiunto al canale

#### ❌ Problema: Errori API
**Causa**: Chiave API mancante o errata
**Soluzione**:
- Verifica `API_FOOTBALL_KEY` nelle variabili d'ambiente
- Controlla quota API su API-SPORTS
- Verifica formato chiave API

#### ❌ Problema: Database Non Persiste
**Causa**: Volume non montato
**Soluzione**:
- Render gestisce automaticamente la persistenza
- Se perdi dati, verifica che il volume sia montato
- Considera backup periodici del database

### 6.4 Backup Database

**Come fare backup**:
1. Vai su **"Shell"** nel menu del service
2. Esegui: `cp signal_quality_learning.db /tmp/backup.db`
3. Scarica il file via SSH o usa un servizio di backup

**Backup automatico** (opzionale):
- Configura un cron job per backup periodici
- Usa un servizio di storage esterno (S3, etc.)

---

## 7. AGGIORNAMENTI E MANUTENZIONE

### 7.1 Deploy Automatico

**Se hai abilitato Auto-Deploy**:
- Ogni push su GitHub triggera automaticamente un nuovo deploy
- Verifica i log dopo ogni push

**Deploy Manuale**:
1. Vai su **"Manual Deploy"** nel menu
2. Seleziona branch e commit
3. Clicca **"Deploy"**

### 7.2 Aggiornare Variabili d'Ambiente

1. Vai su **"Environment"**
2. Modifica la variabile
3. Clicca **"Save Changes"**
4. Il service si riavvierà automaticamente

### 7.3 Riavviare Service

**Riavvio Manuale**:
1. Vai su **"Manual Deploy"**
2. Clicca **"Clear build cache & deploy"**
3. Oppure clicca **"Restart"** nel menu

---

## 8. COSTI E OTTIMIZZAZIONE

### 8.1 Costi Stimati

**Starter Plan ($7/mese)**:
- ✅ Sufficiente per il software
- ✅ 512 MB RAM
- ✅ 0.5 CPU
- ✅ Storage illimitato

**Standard Plan ($25/mese)**:
- ✅ Più risorse se necessario
- ✅ 2 GB RAM
- ✅ 1 CPU
- ✅ Migliori performance

### 8.2 Ottimizzazione Costi

**Ridurre costi**:
- Usa Starter Plan se sufficiente
- Monitora uso risorse
- Ottimizza intervallo cicli (più lungo = meno chiamate API)

**Aumentare performance**:
- Passa a Standard Plan
- Aumenta CPU/RAM se necessario
- Ottimizza codice per usare meno risorse

---

## 9. CHECKLIST FINALE

Prima di considerare il deploy completato, verifica:

- [ ] Account Render creato e piano selezionato
- [ ] Repository GitHub connesso
- [ ] Service "Background Worker" creato
- [ ] Dockerfile path corretto: `./Dockerfile.automation`
- [ ] Variabili d'ambiente configurate:
  - [ ] `TELEGRAM_BOT_TOKEN`
  - [ ] `TELEGRAM_CHAT_ID`
  - [ ] `API_FOOTBALL_KEY`
  - [ ] `AUTOMATION_MIN_EV` (opzionale)
  - [ ] `AUTOMATION_MIN_CONFIDENCE` (opzionale)
  - [ ] `AUTOMATION_UPDATE_INTERVAL` (opzionale)
- [ ] Build completato con successo
- [ ] Logs mostrano sistema avviato
- [ ] Notifica Telegram ricevuta
- [ ] Service rimane attivo (non si spegne)

---

## 10. SUPPORTO

**Se hai problemi**:
1. Controlla i log per errori
2. Verifica variabili d'ambiente
3. Consulta documentazione Render: https://render.com/docs
4. Supporto Render: support@render.com

**Domande comuni**:
- **Q**: Il service si spegne dopo 15 minuti?
  - **A**: Stai usando Free Plan, passa a Starter ($7/mese)
- **Q**: Come vedo i log?
  - **A**: Vai su "Logs" nel menu del service
- **Q**: Come aggiorno il codice?
  - **A**: Fai push su GitHub, se Auto-Deploy è attivo si aggiorna automaticamente

---

## ✅ CONCLUSIONE

Dopo aver completato tutti i passi, il tuo software dovrebbe girare 24/7 su Render!

**Prossimi passi**:
1. Monitora i log per 24-48 ore
2. Verifica che i segnali vengano inviati correttamente
3. Controlla i costi dopo il primo mese
4. Ottimizza se necessario

**Buon deploy! 🚀**

