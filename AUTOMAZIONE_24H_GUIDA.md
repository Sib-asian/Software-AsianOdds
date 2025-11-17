# 🤖 Sistema Automazione 24/7 - Guida Completa

## 🎯 Cosa Fa Questo Sistema

Sistema completamente autonomo che:
1. ✅ **Monitora partite 24/7** - Analizza partite pre-match e live
2. ✅ **Notifica solo VALUE BET reali** - Non consiglia basandosi su score
3. ✅ **Telegram intelligente** - Alert solo per vere opportunità
4. ✅ **Aggiorna dati automaticamente** - Mantiene dati sempre freschi
5. ✅ **Gestisce API quota** - Ottimizza uso API per rimanere nei limiti

---

## 🚫 Problema Risolto

**PRIMA (Sbagliato):**
- Sistema consigliava "1-0 quindi gioca 1" ❌
- Non aveva senso logico
- Basato solo su score, non su valore reale

**ADESSO (Corretto):**
- Analizza solo VALUE BET reali ✅
- Verifica probabilità vs quote
- Considera EV, confidence, e vero valore
- **NON** consiglia basandosi solo su score

---

## 📋 Requisiti

1. **Python 3.8+**
2. **Dipendenze installate**: `pip install -r requirements.txt`
3. **Telegram Bot** (opzionale ma consigliato)
4. **API Keys** configurate (API-Football, TheOddsAPI, etc.)

---

## 🔧 Setup

### 1. Configura Telegram Bot

1. Crea bot su Telegram: https://t.me/BotFather
2. Ottieni token del bot
3. Ottieni chat_id: https://t.me/userinfobot

### 2. Configura Variabili Ambiente

Crea file `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
AUTOMATION_MIN_EV=8.0
AUTOMATION_MIN_CONFIDENCE=70.0
AUTOMATION_UPDATE_INTERVAL=300
```

### 3. Avvia Sistema

**Opzione A: Script Python**
```bash
python start_automation.py
```

**Opzione B: Diretto**
```bash
python automation_24h.py \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID \
  --min-ev 8.0 \
  --min-confidence 70.0
```

**Opzione C: Con Config File**
```bash
python automation_24h.py --config automation_config.json
```

---

## 🎛️ Configurazione

### Parametri Principali

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `min_ev` | 8.0% | Expected Value minimo per notificare |
| `min_confidence` | 70.0% | Confidence minima per notificare |
| `update_interval` | 300s | Secondi tra aggiornamenti (5 min) |
| `api_budget_per_day` | 100 | Chiamate API massime al giorno |

### Filtri Intelligenti

Il sistema applica automaticamente questi filtri:

1. **Evita consigli basati su score**
   - Non consiglia "1-0 quindi gioca 1"
   - Verifica che ci sia altro reasoning oltre allo score

2. **Richiede vero valore**
   - Probabilità reale > Probabilità implicita + margine
   - Margine minimo: 5%

3. **Verifica EV e Confidence**
   - EV > soglia minima
   - Confidence > soglia minima

---

## 📱 Notifiche Telegram

### Formato Notifica

```
⚽ AUTO-24H BETTING OPPORTUNITY ⚽

📅 Match
Team A vs Team B
🏆 Serie A

💰 Recommendation
Market: 1X2_HOME
Stake: €133.68
Odds: 1.90

📊 Analysis
Expected Value: +25.4%
Win Probability: 66.0%
Confidence: 🟢 VERY HIGH (82%)

🤖 AI Ensemble
Dixon-Coles: 65.0% (weight: 30%)
XGBoost: 71.0% (weight: 40%)
LSTM: 68.0% (weight: 30%)

⏰ 14:30:25
```

### Quando Ricevi Notifiche

✅ **Ricevi notifica quando:**
- EV > soglia minima (default 8%)
- Confidence > soglia minima (default 70%)
- C'è vero valore (probabilità > probabilità implicita)
- **NON** è basato solo su score

❌ **NON ricevi notifica quando:**
- EV troppo basso
- Confidence troppo bassa
- Raccomandazione basata solo su score
- Nessun vero valore rilevato

---

## 🔄 Funzionamento

### Ciclo di Analisi

1. **Ogni 5 minuti** (configurabile):
   - Ottiene partite da monitorare
   - Analizza ogni partita con AI Pipeline
   - Verifica se è vera opportunità VALUE BET
   - Notifica se merita

2. **Filtri Applicati**:
   - ✅ Action = BET (non WATCH/SKIP)
   - ✅ EV > soglia
   - ✅ Confidence > soglia
   - ✅ NON basato su score
   - ✅ Vero valore rilevato

3. **Gestione Duplicati**:
   - Evita notifiche duplicate per stessa partita
   - Reset ogni giorno

### Gestione API

- **Quota giornaliera**: 100 chiamate (configurabile)
- **Reset automatico**: Ogni giorno a mezzanotte
- **Ottimizzazione**: Usa cache quando possibile

---

## 🚀 Avvio come Servizio (Linux)

### Systemd Service

Crea `/etc/systemd/system/automation24h.service`:

```ini
[Unit]
Description=Automation 24/7 Betting System
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Software-AsianOdds-main
ExecStart=/usr/bin/python3 /path/to/Software-AsianOdds-main/start_automation.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Comandi:**
```bash
sudo systemctl enable automation24h
sudo systemctl start automation24h
sudo systemctl status automation24h
```

---

## 🚀 Avvio come Servizio (Windows)

### Task Scheduler

1. Apri Task Scheduler
2. Crea nuova attività
3. Trigger: All'avvio
4. Azione: Avvia programma
   - Programma: `python.exe`
   - Argomenti: `C:\path\to\start_automation.py`
   - Directory: `C:\path\to\Software-AsianOdds-main`

### NSSM (Alternative)

```bash
nssm install Automation24H "C:\Python\python.exe" "C:\path\to\start_automation.py"
nssm start Automation24H
```

---

## 📊 Monitoraggio

### Log File

Il sistema scrive log in `automation_24h.log`:

```
2025-01-15 14:30:00 - INFO - 🚀 Starting Automation24H system...
2025-01-15 14:30:00 - INFO - ✅ AI Pipeline initialized
2025-01-15 14:30:00 - INFO - ✅ Telegram Notifier initialized
2025-01-15 14:35:00 - INFO - 🔄 Running analysis cycle...
2025-01-15 14:35:01 - INFO -    Found 5 matches to monitor
2025-01-15 14:35:05 - INFO - ✅ Notified opportunity: match_123
2025-01-15 14:35:10 - INFO - ✅ Cycle complete: 1 opportunities found
```

### Verifica Stato

```bash
# Linux
tail -f automation_24h.log

# Windows
Get-Content automation_24h.log -Wait
```

---

## 🛠️ Troubleshooting

### "Telegram notifier not available"
- Verifica token e chat_id in `.env`
- Testa bot: `python -c "from ai_system.telegram_notifier import TelegramNotifier; ..."`

### "API quota exhausted"
- Aumenta `api_budget_per_day` in config
- Oppure riduci `update_interval`

### "No matches to monitor"
- Verifica connessione API
- Controlla che ci siano partite nelle prossime 24h

### Sistema si blocca
- Verifica log per errori
- Riavvia sistema
- Controlla memoria/CPU

---

## 📈 Miglioramenti Futuri

- [ ] Integrazione con API reali (API-Football, TheOddsAPI)
- [ ] Dashboard web per monitoraggio
- [ ] Notifiche push su mobile
- [ ] Machine learning per ottimizzare filtri
- [ ] Backtesting automatico

---

## ✅ Checklist Setup

- [ ] Telegram bot creato e configurato
- [ ] Variabili ambiente configurate
- [ ] Sistema testato manualmente
- [ ] Servizio configurato (opzionale)
- [ ] Log verificati
- [ ] Notifiche ricevute correttamente

---

**🎉 Sistema pronto! Ora gira 24/7 e ti notifica solo le vere opportunità!**

