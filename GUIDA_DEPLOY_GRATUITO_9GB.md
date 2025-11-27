# 🚀 Guida Deploy Gratuito per Immagini 8.5-9 GB

## 📊 Situazione
- **Dimensione immagine**: 8.5-9 GB
- **Obiettivo**: Automazione 24/7 gratuita
- **Limitazioni**: Molti servizi gratuiti hanno limiti < 5 GB

---

## 🥇 OPZIONE 1: Render.com (CONSIGLIATO per 9 GB)

### ✅ Perché Render.com?
- **Limite 10 GB** - Perfetto per la tua immagine!
- Free tier disponibile (con sleep dopo 15 min)
- Background Worker sempre attivo con piano gratuito (con limitazioni)
- Deploy automatico da GitHub
- Facile configurazione

### 📋 Setup Passo-Passo

1. **Crea account**: https://render.com (puoi usare GitHub login)

2. **Crea nuovo Background Worker**:
   - Dashboard → "New +" → "Background Worker"
   - Connetti il tuo repository GitHub
   - Seleziona branch (es. `main`)

3. **Configurazione**:
   ```
   Name: automation-24h
   Environment: Docker
   Dockerfile Path: ./Dockerfile.automation
   Docker Context: .
   ```

4. **Variabili Ambiente** (Environment Variables):
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   AUTOMATION_MIN_EV=8.0
   AUTOMATION_MIN_CONFIDENCE=70.0
   AUTOMATION_UPDATE_INTERVAL=300
   PYTHONUNBUFFERED=1
   ```

5. **Plan**: Seleziona "Free" (può andare in sleep, ma si risveglia automaticamente)

6. **Deploy!** ✅

### 💰 Costo
- **GRATIS** per Background Worker (con sleep dopo 15 min di inattività)
- Per sempre attivo: $7/mese (ma puoi provare prima il free tier)

### ⚠️ Nota sul Sleep
Il free tier va in sleep dopo 15 min di inattività, ma:
- Si risveglia automaticamente quando riceve richieste
- Per automazione continua, considera l'opzione 2 o 3

---

## 🥈 OPZIONE 2: GitHub Actions (GRATIS, Nessun Limite Dimensione)

### ✅ Perché GitHub Actions?
- **Nessun limite dimensione immagine**
- 2000 minuti/mese GRATIS
- Esegue ogni X minuti (configurabile)
- Completamente gratuito
- Integrato con GitHub

### 📋 Setup

1. **Crea workflow file**: `.github/workflows/automation.yml`

```yaml
name: Automation 24/7

on:
  schedule:
    - cron: '*/5 * * * *'  # Ogni 5 minuti
  workflow_dispatch:  # Esecuzione manuale

jobs:
  automation:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.automation.txt
      
      - name: Run automation
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          AUTOMATION_MIN_EV: 8.0
          AUTOMATION_MIN_CONFIDENCE: 70.0
          AUTOMATION_UPDATE_INTERVAL: 300
        run: |
          python start_automation.py
```

2. **Configura Secrets su GitHub**:
   - Vai su: Repository → Settings → Secrets and variables → Actions
   - Aggiungi:
     - `TELEGRAM_BOT_TOKEN`
     - `TELEGRAM_CHAT_ID`

3. **Push e attiva**:
   ```bash
   git add .github/workflows/automation.yml
   git commit -m "Add GitHub Actions automation"
   git push
   ```

### 💰 Costo
- **GRATIS** (2000 minuti/mese)
- Calcolo: 5 minuti ogni 5 minuti = ~1440 minuti/giorno = ~43,200 minuti/mese
- ⚠️ **Supera il limite gratuito!**

### 🔧 Soluzione: Riduci frequenza
Cambia cron a `*/30 * * * *` (ogni 30 minuti):
- 30 minuti ogni 30 minuti = ~1440 minuti/giorno = ~43,200 minuti/mese
- Ancora troppo!

**Migliore**: `0 */2 * * *` (ogni 2 ore):
- 30 minuti ogni 2 ore = 360 minuti/giorno = ~10,800 minuti/mese
- ⚠️ Ancora troppo!

**Ottimale**: `0 */6 * * *` (ogni 6 ore):
- 30 minuti ogni 6 ore = 120 minuti/giorno = ~3,600 minuti/mese
- ✅ Entro il limite!

**O usa**: Esecuzione una volta al giorno:
- `0 0 * * *` (ogni giorno a mezzanotte)
- 30 minuti/giorno = ~900 minuti/mese
- ✅ Perfetto!

---

## 🥉 OPZIONE 3: Fly.io (Free Tier Generoso)

### ✅ Perché Fly.io?
- Free tier: 3 VM shared-cpu-1x (256 MB RAM)
- Nessun limite esplicito dimensione immagine
- Sempre attivo
- Deploy veloce

### ⚠️ Limitazioni
- 256 MB RAM potrebbe non essere sufficiente per 9 GB di dipendenze
- Potrebbe essere lento

### 📋 Setup

1. **Installa Fly CLI**:
   ```powershell
   # Windows PowerShell
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Deploy**:
   ```bash
   cd Software-AsianOdds-main
   fly launch --dockerfile Dockerfile.automation
   ```

4. **Configura secrets**:
   ```bash
   fly secrets set TELEGRAM_BOT_TOKEN=your_token
   fly secrets set TELEGRAM_CHAT_ID=your_chat_id
   fly secrets set AUTOMATION_MIN_EV=8.0
   fly secrets set AUTOMATION_MIN_CONFIDENCE=70.0
   fly secrets set AUTOMATION_UPDATE_INTERVAL=300
   ```

5. **Deploy!**:
   ```bash
   fly deploy
   ```

### 💰 Costo
- **GRATIS** (3 VM shared-cpu-1x)
- Potrebbe essere lento con 9 GB

---

## 🆓 OPZIONE 4: Ottimizza Immagine + Railway (MIGLIORE se riduci dimensione)

### 🎯 Obiettivo: Ridurre da 9 GB a < 4 GB

Railway ha limite 4 GB, ma è il servizio più semplice. Se riesci a ridurre l'immagine, è perfetto!

### 📋 Strategia Ottimizzazione

1. **Usa multi-stage build** (vedi Dockerfile.optimized qui sotto)
2. **Rimuovi dipendenze non necessarie**
3. **Usa immagini base più piccole**

---

## 🔧 Dockerfile Ottimizzato (Multi-Stage Build)

Ho creato un `Dockerfile.optimized` che riduce drasticamente la dimensione:

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Installa solo build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.automation.txt .

# Installa dipendenze in virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.automation.txt

# Stage 2: Runtime (minimale)
FROM python:3.11-slim

WORKDIR /app

# Copia solo virtualenv da builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Variabili ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copia SOLO file essenziali
COPY automation_24h.py .
COPY start_automation.py .
COPY api_manager.py .

# Copia ai_system (solo file Python, no modelli)
COPY ai_system/__init__.py ./ai_system/
COPY ai_system/config.py ./ai_system/
COPY ai_system/pipeline.py ./ai_system/
COPY ai_system/telegram_notifier.py ./ai_system/
COPY ai_system/blocco_*.py ./ai_system/ 2>/dev/null || true
COPY ai_system/live_*.py ./ai_system/ 2>/dev/null || true
COPY ai_system/backtesting.py ./ai_system/ 2>/dev/null || true
COPY ai_system/background_services.py ./ai_system/ 2>/dev/null || true

# Crea directory utils se necessario
RUN mkdir -p ./ai_system/utils

# Pulisci tutto
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

CMD ["python", "start_automation.py"]
```

**Dimensione attesa**: ~500 MB - 1.5 GB (invece di 9 GB!)

---

## 🎯 RACCOMANDAZIONE FINALE

### Per Immagine 9 GB (senza ottimizzare):
1. **Render.com** - Limite 10 GB, perfetto! ✅
2. **GitHub Actions** - Se accetti esecuzione ogni 6+ ore
3. **Fly.io** - Prova, ma potrebbe essere lento

### Dopo Ottimizzazione (< 2 GB):
1. **Railway.app** - Sempre attivo, gratis, facile! ⭐
2. **Render.com** - Sempre valida
3. **Fly.io** - Buona alternativa

---

## 📋 Checklist Deploy

### Prima:
- [ ] Codice committato su GitHub
- [ ] Telegram bot creato (https://t.me/BotFather)
- [ ] Chat ID ottenuto (https://t.me/userinfobot)
- [ ] Variabili ambiente preparate

### Durante:
- [ ] Account creato su servizio scelto
- [ ] Repository connesso
- [ ] Variabili ambiente configurate
- [ ] Deploy avviato
- [ ] Log verificati

### Dopo:
- [ ] Sistema attivo
- [ ] Notifiche Telegram ricevute
- [ ] Log senza errori
- [ ] Monitoraggio configurato

---

## 🚀 Quick Start (Render.com - Consigliato)

```bash
# 1. Assicurati che il codice sia su GitHub
git add .
git commit -m "Ready for deployment"
git push

# 2. Vai su https://render.com
# 3. New + → Background Worker
# 4. Connetti GitHub repository
# 5. Configura:
#    - Name: automation-24h
#    - Environment: Docker
#    - Dockerfile Path: ./Dockerfile.automation
# 6. Aggiungi Environment Variables (vedi sopra)
# 7. Deploy! ✅
```

---

## 💡 Tips

1. **Inizia con Render.com** - Più semplice per immagini grandi
2. **Monitora i log** - Primi giorni per verificare
3. **Testa localmente** - Prima di deployare:
   ```bash
   docker build -f Dockerfile.automation -t automation-test .
   docker run -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=xxx automation-test
   ```
4. **Ottimizza dopo** - Se vuoi usare Railway, riduci dimensione prima

---

## 🆘 Troubleshooting

### "Build failed - out of memory"
- Usa Dockerfile.optimized (multi-stage)
- Riduci dipendenze in requirements.automation.txt

### "Service keeps restarting"
- Verifica variabili ambiente
- Controlla log per errori
- Verifica che start_automation.py esista

### "No notifications"
- Verifica token Telegram
- Verifica chat ID
- Controlla log per errori API

---

## ✅ Conclusione

**Per la tua situazione (9 GB):**
- **Render.com** è la scelta migliore (limite 10 GB) ✅
- **GitHub Actions** se accetti esecuzione periodica
- **Ottimizza immagine** se vuoi usare Railway (migliore UX)

**🎉 Buon deploy!**






