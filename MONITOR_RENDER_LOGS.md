# 📊 Monitoraggio Log Render.com

Guida per monitorare i log del tuo servizio Render.com in tempo reale.

## 🎯 Metodi Disponibili

### 1. Dashboard Web (Più Semplice) ⭐

**Passi:**
1. Vai su https://dashboard.render.com
2. Accedi al tuo account
3. Clicca sul tuo servizio (Background Worker)
4. Vai alla tab **"Logs"**
5. Clicca su **"Live tail"** per vedere i log in tempo reale

**Vantaggi:**
- ✅ Nessuna installazione necessaria
- ✅ Interfaccia grafica
- ✅ Filtri e ricerca integrata
- ✅ Supporto per regex e caratteri jolly

---

### 2. Script Python (Monitoraggio da Terminale)

**Installazione:**
```bash
pip install requests colorama
```

**Configurazione:**
1. Ottieni la tua **API Key**:
   - Vai su https://dashboard.render.com/account/api-keys
   - Crea una nuova API Key
   - Copia la chiave

2. Ottieni il **Service ID**:
   - Vai sul tuo servizio su Render
   - L'URL sarà: `https://dashboard.render.com/web/.../services/YOUR_SERVICE_ID`
   - Copia l'ID dal URL

**Uso:**
```bash
# Metodo 1: Variabili d'ambiente
export RENDER_API_KEY=your_api_key
export RENDER_SERVICE_ID=your_service_id
python monitor_render_logs.py

# Metodo 2: Parametri diretti
python monitor_render_logs.py --api-key YOUR_API_KEY --service-id YOUR_SERVICE_ID

# Con filtri
python monitor_render_logs.py --api-key YOUR_API_KEY --service-id YOUR_SERVICE_ID --level ERROR
python monitor_render_logs.py --api-key YOUR_API_KEY --service-id YOUR_SERVICE_ID --filter "segnali"
```

**Opzioni:**
- `--interval`: Intervallo aggiornamento in secondi (default: 2.0)
- `--level`: Filtra per livello (ERROR, WARN, INFO, DEBUG)
- `--filter`: Filtra per testo nel messaggio

---

### 3. Render CLI (Opzionale)

**Installazione:**
```bash
# Windows (PowerShell)
scoop install render

# macOS
brew install render

# Linux
curl -fsSL https://render.com/install.sh | bash
```

**Uso:**
```bash
# Login
render login

# Monitora log
render logs --service YOUR_SERVICE_ID --tail

# Filtra per livello
render logs --service YOUR_SERVICE_ID --tail --level error
```

---

## 🔍 Cosa Monitorare

### Log Importanti da Cercare:

1. **Avvio Sistema:**
   ```
   ✅ Automation24H initialized
   ✅ Signal Quality Learner inizializzato
   ```

2. **Segnali:**
   ```
   📝 Segnale registrato nel database
   ✅ Notified live opportunity
   ```

3. **Errori:**
   ```
   ❌ Error
   ⚠️  Warning
   ```

4. **Apprendimento IA:**
   ```
   🧠 Eseguendo apprendimento automatico
   ✅ Aggiornati X segnali per partita
   ```

5. **API Calls:**
   ```
   📊 API calls remaining
   ⚠️  API quota limit
   ```

---

## 🛠️ Troubleshooting

### Problema: Script Python non funziona

**Soluzione:**
- Verifica che `requests` e `colorama` siano installati: `pip install requests colorama`
- Controlla che API Key e Service ID siano corretti
- Verifica che il servizio sia attivo su Render

### Problema: Log non aggiornati

**Soluzione:**
- Il servizio potrebbe essere spento, verifica su dashboard
- Controlla che Auto-Deploy sia attivo
- Verifica che il codice sia stato pushato su GitHub

### Problema: Troppi log

**Soluzione:**
- Usa filtri: `--level ERROR` per solo errori
- Usa `--filter "testo"` per cercare specifici messaggi
- Aumenta `--interval` per aggiornamenti meno frequenti

---

## 📝 Note

- I log su Render sono conservati per un periodo limitato (dipende dal piano)
- Per log storici più lunghi, considera l'integrazione con provider esterni (Datadog, Sumo Logic)
- Il monitoraggio via API ha limiti di rate, non fare polling troppo frequente

---

## 🔗 Link Utili

- Dashboard Render: https://dashboard.render.com
- API Keys: https://dashboard.render.com/account/api-keys
- Documentazione API: https://render.com/docs/api
- Documentazione Log: https://render.com/docs/logging

