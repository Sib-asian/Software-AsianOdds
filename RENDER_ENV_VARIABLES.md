# 🔑 VARIABILI D'AMBIENTE PER RENDER

## 📋 GUIDA RAPIDA

Quando crei il service su Render, devi aggiungere queste variabili d'ambiente nella sezione **"Environment"**.

---

## ✅ VARIABILI OBBLIGATORIE

### 1. TELEGRAM_BOT_TOKEN
**Key**: `TELEGRAM_BOT_TOKEN`  
**Value**: Il tuo token del bot Telegram  
**Esempio**: `8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g`  
**Dove trovarlo**: 
- Vai su Telegram → Cerca "@BotFather"
- Invia `/mybots`
- Seleziona il tuo bot
- Clicca "API Token"
- Copia il token

**⚠️ IMPORTANTE**: 
- NON condividere questo token pubblicamente
- Se il token viene compromesso, rigeneralo da BotFather

---

### 2. TELEGRAM_CHAT_ID
**Key**: `TELEGRAM_CHAT_ID`  
**Value**: Il Chat ID del canale/gruppo dove vuoi ricevere notifiche  
**Esempio**: `-1003278011521`  
**Dove trovarlo**:
- Aggiungi il bot al canale/gruppo
- Invia un messaggio nel canale/gruppo
- Vai su: https://api.telegram.org/bot<TOKEN>/getUpdates
- Cerca `"chat":{"id":-1003278011521}` nel JSON
- Il numero dopo `"id":` è il Chat ID

**⚠️ IMPORTANTE**: 
- Per canali: inizia con `-100`
- Per gruppi: può essere negativo
- Per chat private: è un numero positivo

---

### 3. API_FOOTBALL_KEY
**Key**: `API_FOOTBALL_KEY`  
**Value**: La tua chiave API di API-SPORTS  
**Esempio**: `95c43f936816cd4389a747fd2cfe061a`  
**Dove trovarlo**:
- Vai su: https://dashboard.api-sports.io/
- Accedi al tuo account
- Vai su "API Keys"
- Copia la chiave (RapidAPI Key)

**⚠️ IMPORTANTE**: 
- Questa è la chiave per API-SPORTS (v3.football.api-sports.io)
- Verifica che il piano sia attivo (Pro: 7500 calls/day)
- NON condividere questa chiave pubblicamente

---

## ⚙️ VARIABILI OPZIONALI (Hanno Default)

### 4. AUTOMATION_MIN_EV
**Key**: `AUTOMATION_MIN_EV`  
**Value**: `8.0`  
**Descrizione**: Valore minimo di Expected Value (in percentuale) per inviare segnali  
**Default**: Se non impostato, usa `8.0`  
**Esempio**: 
- `8.0` = minimo 8% di EV
- `10.0` = minimo 10% di EV (più conservativo)
- `5.0` = minimo 5% di EV (più aggressivo)

---

### 5. AUTOMATION_MIN_CONFIDENCE
**Key**: `AUTOMATION_MIN_CONFIDENCE`  
**Value**: `70.0`  
**Descrizione**: Confidence minima (in percentuale) per inviare segnali  
**Default**: Se non impostato, usa `70.0`  
**Esempio**:
- `70.0` = minimo 70% di confidence
- `80.0` = minimo 80% di confidence (più conservativo)
- `60.0` = minimo 60% di confidence (più aggressivo)

---

### 6. AUTOMATION_UPDATE_INTERVAL
**Key**: `AUTOMATION_UPDATE_INTERVAL`  
**Value**: `600`  
**Descrizione**: Intervallo tra cicli di analisi in secondi  
**Default**: Se non impostato, usa `300` (5 minuti)  
**Esempio**:
- `300` = 5 minuti (più frequente, più chiamate API)
- `600` = 10 minuti (consigliato, bilanciato)
- `900` = 15 minuti (meno frequente, meno chiamate API)

**⚠️ NOTA**: 
- Intervalli più brevi = più chiamate API = più costi
- Intervalli più lunghi = meno segnali in tempo reale

---

## 🔧 VARIABILI API OPZIONALI

### 7. THEODDS_API_KEY
**Key**: `THEODDS_API_KEY`  
**Value**: La tua chiave API di TheOddsAPI (opzionale)  
**Dove trovarlo**: https://the-odds-api.com/  
**Default**: Se non impostato, usa mock data  
**Quando usarla**: Se vuoi quote da TheOddsAPI invece di API-SPORTS

---

### 8. FOOTBALL_DATA_KEY
**Key**: `FOOTBALL_DATA_KEY`  
**Value**: La tua chiave API di Football-Data.org (opzionale)  
**Dove trovarlo**: https://www.football-data.org/  
**Default**: Se non impostato, non usa questo provider  
**Quando usarla**: Come fallback per dati partite

---

### 9. NEWSAPI_KEY
**Key**: `NEWSAPI_KEY`  
**Value**: La tua chiave API di NewsAPI (opzionale)  
**Dove trovarlo**: https://newsapi.org/  
**Default**: Se non impostato, disabilita analisi news  
**Quando usarla**: Se vuoi analisi sentiment delle news

---

## 📝 ESEMPIO COMPLETO

Ecco come dovrebbero essere configurate le variabili su Render:

```
TELEGRAM_BOT_TOKEN = 8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g
TELEGRAM_CHAT_ID = -1003278011521
API_FOOTBALL_KEY = 95c43f936816cd4389a747fd2cfe061a
AUTOMATION_MIN_EV = 8.0
AUTOMATION_MIN_CONFIDENCE = 70.0
AUTOMATION_UPDATE_INTERVAL = 600
```

---

## ⚠️ SICUREZZA

**NON fare mai**:
- ❌ Committare variabili d'ambiente nel codice
- ❌ Condividere chiavi API pubblicamente
- ❌ Includere token in screenshot o documentazione pubblica

**Fai sempre**:
- ✅ Usa variabili d'ambiente su Render
- ✅ Mantieni le chiavi segrete
- ✅ Rigenera le chiavi se compromesse
- ✅ Usa `.env` locale solo per sviluppo (non committare!)

---

## 🔍 VERIFICA CONFIGURAZIONE

Dopo aver configurato le variabili, verifica nei log:

1. ✅ `✅ Telegram Notifier initialized` - Telegram configurato
2. ✅ `✅ API Manager initialized` - API configurate
3. ✅ `✅ Signal Quality Learner inizializzato` - Sistema IA attivo
4. ✅ `🚀 Avvio sistema automazione 24/7...` - Sistema avviato

Se vedi errori:
- ⚠️ `⚠️ Telegram not configured` - Controlla `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
- ⚠️ `⚠️ API_FOOTBALL_KEY non configurata` - Controlla `API_FOOTBALL_KEY`

---

## 📞 SUPPORTO

Se hai problemi con le variabili:
1. Controlla i log su Render
2. Verifica che tutte le variabili siano scritte correttamente (case-sensitive!)
3. Assicurati che non ci siano spazi extra
4. Verifica che i valori siano corretti (token, chat ID, etc.)

