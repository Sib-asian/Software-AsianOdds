# 🚀 Deployment Centralizzato su Render

Tutte le configurazioni di deployment sono ora **centralizzate su Render** per evitare istanze multiple e consumo eccessivo di API.

## ⚙️ Architettura

```
Render (render.yaml)
├── 🤖 Worker: automation-24h
│   ├── Docker: Dockerfile.automation
│   ├── Script: start_automation.py
│   ├── Ciclo: ogni 20 minuti (72 cicli/giorno)
│   ├── Piano: Starter ($7/mese)
│   └── Disco: /data (1GB, condiviso)
│
└── 🌐 Web Service: streamlit-dashboard
    ├── Docker: Dockerfile
    ├── App: Frontendcloud.py
    ├── Porta: 8501
    ├── Piano: Starter ($7/mese)
    └── Disco: /data (1GB, condiviso)
```

## 📝 Configurazioni Disabilitate

Le seguenti configurazioni sono state **disabilitate** per evitare istanze multiple:

- ❌ `fly.toml` → `fly.toml.disabled` (Fly.io)
- ❌ `railway.json` → `railway.json.disabled` (Railway)

**IMPORTANTE**: Se hai servizi attivi su queste piattaforme, **disabilitali manualmente**:

### Disabilitare Fly.io
```bash
flyctl apps list
flyctl scale count 0 -a automation-24h
# Oppure elimina definitivamente:
flyctl apps destroy automation-24h
```

### Disabilitare Railway
```bash
# Vai su https://railway.app/dashboard
# Seleziona il progetto e clicca "Delete Service"
```

## 🔧 Deploy su Render

### 1. Prerequisiti

- Account Render: https://dashboard.render.com
- Repository GitHub collegato
- Variabili d'ambiente configurate

### 2. Deploy Automatico via render.yaml

1. **Vai su Render Dashboard**
   ```
   https://dashboard.render.com
   ```

2. **Crea nuovo Blueprint**
   - Click su "New" → "Blueprint"
   - Seleziona il repository GitHub
   - Render rileverà automaticamente `render.yaml`

3. **Configura variabili d'ambiente**

   **OBBLIGATORIE** (da configurare manualmente nel dashboard):
   ```
   TELEGRAM_BOT_TOKEN=il_tuo_bot_token
   TELEGRAM_CHAT_ID=il_tuo_chat_id
   API_FOOTBALL_KEY=la_tua_api_key
   ```

   **OPZIONALI** (già con default in render.yaml):
   ```
   AUTOMATION_MIN_EV=8.0
   AUTOMATION_MIN_CONFIDENCE=70.0
   AUTOMATION_UPDATE_INTERVAL=1200
   ```

   **OPZIONALI** (API aggiuntive):
   ```
   THEODDS_API_KEY=... (solo se usi TheOddsAPI)
   FOOTBALL_DATA_KEY=... (solo se usi Football-Data)
   NEWSAPI_KEY=... (solo se usi NewsAPI)
   ```

4. **Deploy**
   - Click "Apply" per creare entrambi i servizi
   - Render creerà:
     - `automation-24h` (worker)
     - `streamlit-dashboard` (web)

### 3. Verifica Deployment

Dopo il deploy, verifica che:

✅ **Worker attivo**:
```
Logs → automation-24h
Cerca: "✅ Sistema 24/7 AVVIATO con successo!"
```

✅ **Dashboard accessibile**:
```
https://streamlit-dashboard.onrender.com
```

✅ **Consumo API sotto controllo**:
```
Logs → automation-24h
Cerca: "📊 Consumo API nel ciclo"
Dovrebbe mostrare: ~2 chiamate/ciclo (con cache attiva)
```

## 🛑 Sospendere i Servizi

Per sospendere temporaneamente (es. durante manutenzione):

### Via Dashboard
1. Vai su https://dashboard.render.com
2. Seleziona il servizio
3. Click "Suspend"

### Via CLI
```bash
# Installa Render CLI
npm install -g render-cli

# Sospendi worker
render services suspend automation-24h

# Sospendi dashboard
render services suspend streamlit-dashboard
```

## 💰 Costi Stimati

```
Worker (automation-24h):    $7/mese  (Starter Plan)
Web (streamlit-dashboard):  $7/mese  (Starter Plan)
Persistent Disk (1GB):      $0.25/mese
────────────────────────────────────────
TOTALE:                     ~$14.25/mese
```

**Risparmi rispetto a istanze multiple**:
- ❌ Fly.io duplicato: $7/mese eliminato
- ❌ Railway duplicato: $5/mese eliminato
- ✅ **Risparmio totale**: ~$12/mese

## 📊 Monitoraggio

### Logs in tempo reale
```bash
# Worker logs
render logs automation-24h -f

# Dashboard logs
render logs streamlit-dashboard -f
```

### Metriche Chiave

**Consumo API giornaliero** (target: <500 chiamate/giorno):
```
Logs filtrati: "📊 Consumo API nel ciclo"
Formula: chiamate_per_ciclo × 72 cicli/giorno
Target: 2 × 72 = 144 chiamate/giorno ✅
```

**Uptime servizi**:
```
Dashboard → Services → Metrics
Target: >99.5% uptime
```

## 🔒 Sicurezza

- ✅ Variabili sensibili: `sync: false` (non sincronizzate tra servizi)
- ✅ Auto-deploy: **disabilitato** (`autoDeploy: false`)
- ✅ Credenziali: Solo via dashboard Render (mai in git)
- ✅ Logs: Filtrati automaticamente (no API keys nei logs)

## 🆘 Troubleshooting

### "Chiamate API ancora attive con servizio sospeso"

**Causa**: Istanze su altre piattaforme ancora attive

**Soluzione**:
1. Controlla Fly.io: `flyctl apps list`
2. Controlla Railway: https://railway.app/dashboard
3. Disabilita tutte le istanze trovate

### "Dashboard non accessibile"

**Causa**: Porta Streamlit non configurata

**Soluzione**:
```yaml
# Verifica in render.yaml:
envVars:
  - key: PORT
    value: 8501
```

### "Worker crasha all'avvio"

**Causa**: Variabili d'ambiente mancanti

**Soluzione**:
```bash
# Controlla logs:
render logs automation-24h

# Cerca errori tipo:
# "TELEGRAM_BOT_TOKEN not found"
# "API_FOOTBALL_KEY not found"
```

Configura le variabili mancanti nel dashboard.

---

**Ultimo aggiornamento**: 2025-11-23
**Configurazione**: Render centralizzato
**Versione**: 1.0
