# 🚀 Deploy Gratuito 24/7 - Guida Completa

## 🎯 Opzioni Gratuite per Hosting 24/7

Hai diverse opzioni **GRATUITE** per far girare il sistema 24/7 senza server dedicato:

---

## 🥇 Opzione 1: Railway.app (CONSIGLIATO)

**✅ Vantaggi:**
- Free tier generoso ($5 crediti/mese)
- Deploy automatico da GitHub
- Nessun sleep (sempre attivo)
- Facile da configurare

**📋 Setup:**

1. **Crea account**: https://railway.app
2. **Connetti GitHub** al tuo repository
3. **Crea nuovo progetto** → "Deploy from GitHub repo"
4. **Seleziona repository** Software-AsianOdds
5. **Configura variabili ambiente**:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   AUTOMATION_MIN_EV=8.0
   AUTOMATION_MIN_CONFIDENCE=70.0
   AUTOMATION_UPDATE_INTERVAL=300
   ```
6. **Deploy automatico!** ✅

**💰 Costo:** GRATIS (con $5 crediti/mese)

---

## 🥈 Opzione 2: Render.com

**✅ Vantaggi:**
- Free tier disponibile
- Deploy automatico
- Facile configurazione

**⚠️ Limitazioni:**
- Worker free tier va in sleep dopo 15 min di inattività
- Per 24/7 serve piano a pagamento (ma puoi usare cron jobs)

**📋 Setup:**

1. **Crea account**: https://render.com
2. **New** → **Background Worker**
3. **Connetti GitHub** repository
4. **Configura**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python start_automation.py`
5. **Aggiungi variabili ambiente** (come Railway)
6. **Deploy!**

**💰 Costo:** GRATIS (con limitazioni) o $7/mese per sempre attivo

---

## 🥉 Opzione 3: Fly.io

**✅ Vantaggi:**
- Free tier generoso
- Deploy veloce
- Buona documentazione

**📋 Setup:**

1. **Installa Fly CLI**:
   ```bash
   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   
   # Mac/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Deploy**:
   ```bash
   fly launch
   # Segui le istruzioni
   ```

4. **Configura variabili**:
   ```bash
   fly secrets set TELEGRAM_BOT_TOKEN=your_token
   fly secrets set TELEGRAM_CHAT_ID=your_chat_id
   ```

**💰 Costo:** GRATIS (3 VM shared-cpu-1x)

---

## 🆓 Opzione 4: GitHub Actions (Cron Job)

**✅ Vantaggi:**
- Completamente gratuito
- Integrato con GitHub
- 2000 minuti/mese gratis

**⚠️ Limitazioni:**
- Non sempre attivo (solo quando esegue)
- Esegue ogni X minuti (non continuo)

**📋 Setup:**

Crea `.github/workflows/automation.yml`:

```yaml
name: Automation 24/7

on:
  schedule:
    - cron: '*/5 * * * *'  # Ogni 5 minuti
  workflow_dispatch:  # Esecuzione manuale

jobs:
  automation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run automation
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          AUTOMATION_MIN_EV: 8.0
          AUTOMATION_MIN_CONFIDENCE: 70.0
        run: |
          python automation_24h.py \
            --telegram-token $TELEGRAM_BOT_TOKEN \
            --telegram-chat-id $TELEGRAM_CHAT_ID
```

**Configura secrets su GitHub:**
- Settings → Secrets → Actions
- Aggiungi `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`

**💰 Costo:** GRATIS (2000 min/mese)

---

## 🆓 Opzione 5: PythonAnywhere

**✅ Vantaggi:**
- Free tier disponibile
- Sempre attivo
- Facile da usare

**⚠️ Limitazioni:**
- Free tier limitato (1 task sempre attivo)
- Interfaccia web-based

**📋 Setup:**

1. **Crea account**: https://www.pythonanywhere.com
2. **Upload codice** via web interface
3. **Crea scheduled task**:
   - Tasks → Create new task
   - Command: `python3 /home/username/Software-AsianOdds-main/start_automation.py`
   - Schedule: Always (runs continuously)

**💰 Costo:** GRATIS (con limitazioni)

---

## 🎯 RACCOMANDAZIONE

**Per uso 24/7 continuo:**
1. **Railway.app** (migliore, sempre attivo, gratis)
2. **Fly.io** (buona alternativa)
3. **Render.com** (se accetti sleep)

**Per uso periodico:**
- **GitHub Actions** (gratis, ogni X minuti)

---

## 📋 Checklist Deploy

### Prima del Deploy:

- [ ] Codice committato su GitHub
- [ ] Telegram bot creato e token ottenuto
- [ ] Chat ID ottenuto
- [ ] Variabili ambiente preparate
- [ ] Dockerfile testato localmente (opzionale)

### Durante Deploy:

- [ ] Account creato su servizio scelto
- [ ] Repository connesso
- [ ] Variabili ambiente configurate
- [ ] Deploy avviato
- [ ] Log verificati

### Dopo Deploy:

- [ ] Sistema attivo e funzionante
- [ ] Notifiche Telegram ricevute
- [ ] Log verificati per errori
- [ ] Monitoraggio configurato

---

## 🔧 Configurazione Variabili Ambiente

Tutte le piattaforme richiedono queste variabili:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
AUTOMATION_MIN_EV=8.0
AUTOMATION_MIN_CONFIDENCE=70.0
AUTOMATION_UPDATE_INTERVAL=300
```

**Come ottenerle:**

1. **Telegram Bot Token:**
   - Vai su https://t.me/BotFather
   - `/newbot` → Segui istruzioni
   - Copia token

2. **Telegram Chat ID:**
   - Vai su https://t.me/userinfobot
   - Invia `/start`
   - Copia il tuo ID

---

## 🐳 Deploy con Docker (Tutte le Piattaforme)

Se usi Docker, tutte le piattaforme supportano il `Dockerfile.automation` incluso.

**Test locale:**
```bash
docker build -f Dockerfile.automation -t automation-24h .
docker run -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=xxx automation-24h
```

---

## 📊 Monitoraggio

### Railway.app
- Dashboard → Logs (in tempo reale)
- Metrics → CPU/Memory usage

### Render.com
- Dashboard → Logs
- Metrics disponibili

### Fly.io
```bash
fly logs
fly status
```

### GitHub Actions
- Actions tab → Vedi esecuzioni
- Log per ogni run

---

## 🛠️ Troubleshooting

### "Service keeps restarting"
- Verifica variabili ambiente
- Controlla log per errori
- Verifica che `start_automation.py` esista

### "No notifications received"
- Verifica token Telegram
- Verifica chat ID
- Controlla log per errori API

### "Out of memory"
- Riduci `update_interval`
- Ottimizza codice
- Upgrade piano (se necessario)

---

## 💡 Tips

1. **Inizia con Railway.app** - Più semplice e sempre attivo
2. **Monitora i log** - Primi giorni per verificare funzionamento
3. **Testa localmente** - Prima di deployare
4. **Backup config** - Salva variabili ambiente in file sicuro

---

## ✅ Quick Start (Railway.app)

```bash
# 1. Push codice su GitHub
git add .
git commit -m "Add automation 24/7"
git push

# 2. Vai su railway.app
# 3. New Project → Deploy from GitHub
# 4. Seleziona repository
# 5. Aggiungi variabili ambiente
# 6. Deploy! ✅
```

**🎉 Fatto! Il sistema gira 24/7 gratis!**

