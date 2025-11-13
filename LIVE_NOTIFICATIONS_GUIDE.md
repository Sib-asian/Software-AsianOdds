# Live Betting Notifications Guide

Sistema di notifiche Telegram automatiche per opportunità betting live.

## 🚀 Quick Start

### 1. Configura Telegram Bot

**Crea un Bot:**
1. Apri Telegram e cerca `@BotFather`
2. Invia `/newbot`
3. Segui le istruzioni e salva il **Bot Token**

**Ottieni Chat ID:**
1. Scrivi al bot `@userinfobot`
2. Copia il tuo **Chat ID** (numero con il segno meno per gruppi)

### 2. Configura Variabili d'Ambiente

```bash
# Linux/Mac
export TELEGRAM_BOT_TOKEN="8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
export TELEGRAM_CHAT_ID="-1003278011521"

# Windows (PowerShell)
$env:TELEGRAM_BOT_TOKEN="8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
$env:TELEGRAM_CHAT_ID="-1003278011521"

# Oppure crea file .env nella root del progetto
echo 'TELEGRAM_BOT_TOKEN="your_token"' >> .env
echo 'TELEGRAM_CHAT_ID="your_chat_id"' >> .env
```

### 3. Avvia Monitoring

```bash
# Basic usage
python start_live_monitoring.py

# Con intervallo personalizzato (ogni 30 secondi)
python start_live_monitoring.py --interval 30

# Test senza inviare notifiche (dry-run)
python start_live_monitoring.py --dry-run

# Debug mode
python start_live_monitoring.py --log-level DEBUG
```

---

## 📱 Tipi di Notifiche

### 1. **Pre-Match Opportunities**
Inviata quando l'AI rileva opportunità di valore prima della partita.

```
⚽ PRE-MATCH BETTING OPPORTUNITY ⚽

📅 Match
Manchester City vs Arsenal
🏆 Premier League

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

📉 Uncertainty: 2.5%

⏰ 14:32:15
```

### 2. **Live Value Shifts**
Alert quando la probabilità cambia significativamente durante la partita.

```
📈 VALUE OPPORTUNITY DETECTED

🔴 LIVE
Manchester City vs Arsenal
35' - Score: 1-0

📊 Updated Probability
Home Win: 72.1%

⏱️ Timing
Recommendation: WATCH

🔄 Live Adjustments
Score impact: 1.15x
xG impact: 1.08x
Momentum: 1.05x
```

### 3. **Bet Now Alerts**
Notifica quando è il momento ottimale per puntare.

```
🚨 BET NOW - OPTIMAL TIMING

🔴 LIVE
Liverpool vs Chelsea
78' - Score: 2-1

📊 Updated Probability
Home Win: 89.2%

⏱️ Timing
Recommendation: NOW
```

### 4. **Match Events**
Alert per eventi importanti (goal, espulsioni).

```
⚽ GOAL SCORED

🔴 LIVE
Real Madrid vs Barcelona
67' - Score: 3-2

📊 Updated Probability
Home Win: 78.5%
```

---

## ⚙️ Configurazione Avanzata

### File: `ai_system/config.py`

```python
# Telegram notifications
telegram_enabled: bool = True
telegram_bot_token: str = ""  # O usa env var
telegram_chat_id: str = ""    # O usa env var

# Notification thresholds
telegram_min_ev: float = 5.0           # Min EV% per inviare (default: 5%)
telegram_min_confidence: float = 60.0  # Min confidence (default: 60%)
telegram_rate_limit_seconds: int = 3   # Secondi tra messaggi

# Live monitoring
live_monitoring_enabled: bool = True
live_update_interval: int = 60         # Aggiorna ogni 60s
live_min_ev_alert: float = 8.0         # Alert solo se EV > 8%

# Daily reports
telegram_daily_report_enabled: bool = True
telegram_daily_report_time: str = "22:00"
```

---

## 🔧 Integrazione con Sistema Esistente

### Opzione 1: Usa il Monitor Standalone

```python
from ai_system.telegram_notifier import TelegramNotifier
from ai_system.live_monitor import LiveMonitor

# Setup
notifier = TelegramNotifier(
    bot_token="YOUR_TOKEN",
    chat_id="YOUR_CHAT_ID",
    min_ev=5.0
)

monitor = LiveMonitor(
    telegram_notifier=notifier,
    update_interval=60,
    min_ev_alert=8.0
)

# Aggiungi partita
monitor.add_match(
    match_id="12345",
    home_team="Man City",
    away_team="Arsenal",
    league="Premier League",
    pre_match_prob=0.65,
    odds=1.90
)

# Start
monitor.start()
```

### Opzione 2: Integra in Pipeline Esistente

```python
from ai_system.pipeline import quick_analyze
from ai_system.telegram_notifier import TelegramNotifier

# Setup notifier
notifier = TelegramNotifier(
    bot_token="YOUR_TOKEN",
    chat_id="YOUR_CHAT_ID"
)

# Analizza partita
result = quick_analyze(
    home_team="Man City",
    away_team="Arsenal",
    league="Premier League",
    prob_dixon_coles=0.65,
    odds=1.90,
    bankroll=1000.0
)

# Invia notifica se opportunità
if result['action'] == 'BET':
    match_data = {
        'home': 'Man City',
        'away': 'Arsenal',
        'league': 'Premier League'
    }

    notifier.send_betting_opportunity(
        match_data,
        result,
        opportunity_type="PRE-MATCH"
    )
```

### Opzione 3: Integra in Frontendcloud.py

Nel tuo file `Frontendcloud.py` circa alla riga 16827:

```python
# Existing code
ai_result = quick_analyze(
    match=match_data,
    prob_dixon_coles=prob,
    odds_data=odds,
    bankroll=bankroll
)

# ADD THIS: Send notification if opportunity found
if TELEGRAM_ENABLED and ai_result.get('action') == 'BET':
    from ai_system.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier(
        bot_token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID
    )

    notifier.send_betting_opportunity(
        match_data=match_data,
        analysis_result=ai_result,
        opportunity_type="PRE-MATCH"
    )
```

---

## 🔍 Customizzare le Notifiche

### Cambia Template Messaggio

Modifica `ai_system/telegram_notifier.py`:

```python
def _format_betting_message(self, match_data, analysis_result, opportunity_type):
    # Personalizza qui il formato del messaggio
    message = f"""
🎯 NUOVA OPPORTUNITÀ

Match: {match_data['home']} vs {match_data['away']}
Bet: {analysis_result['market']}
Stake: €{analysis_result['stake_amount']:.2f}
"""
    return message
```

### Aggiungi Nuovi Tipi di Alert

```python
# In telegram_notifier.py
def send_custom_alert(self, title: str, message: str) -> bool:
    """Invia alert personalizzato"""
    formatted = f"🔔 <b>{title}</b>\n\n{message}"
    return self._send_message(formatted)
```

---

## 📊 Monitoraggio Stato

### Verifica Status Monitor

```python
from ai_system.live_monitor import LiveMonitor

monitor = LiveMonitor(...)
status = monitor.get_status()

print(status)
# {
#     'running': True,
#     'matches_monitored': 3,
#     'matches': [
#         {
#             'match_id': '001',
#             'home': 'Man City',
#             'away': 'Arsenal',
#             'minute': 67,
#             'score': '2-1',
#             'probability': 0.78
#         },
#         ...
#     ]
# }
```

---

## 🐛 Troubleshooting

### Notifiche Non Arrivano

**1. Verifica credentials:**
```bash
# Test manuale
curl -X POST https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage \
  -H "Content-Type: application/json" \
  -d '{"chat_id":"<YOUR_CHAT_ID>","text":"Test"}'
```

**2. Check bot avviato:**
- Cerca il bot su Telegram
- Invia `/start` al bot
- Prova a inviare un messaggio

**3. Check gruppo/canale:**
- Bot deve essere membro del gruppo
- Bot deve avere permessi di scrittura

### Rate Limiting

Se ricevi errore 429:
```python
# Aumenta rate_limit_seconds in config
telegram_rate_limit_seconds: int = 5  # Da 3 a 5 secondi
```

### Troppe Notifiche

```python
# Aumenta thresholds
telegram_min_ev: float = 10.0          # Da 5% a 10%
telegram_min_confidence: float = 70.0  # Da 60% a 70%
live_min_ev_alert: float = 12.0        # Da 8% a 12%
```

---

## 📝 TODO - Integrazioni Future

Per rendere il sistema completamente automatico, devi:

1. **Implementa `fetch_live_data_from_api()` in `start_live_monitoring.py`**
   - Integra con API-Football, Sofascore, o altra API live
   - Esempio: https://www.api-football.com/documentation-v3

2. **Implementa `get_matches_to_monitor()` in `start_live_monitoring.py`**
   - Query al tuo database per partite con opportunità
   - Oppure scheduler che seleziona partite automaticamente

3. **Scheduler Automatico**
   - Cron job che avvia monitoring 1h prima del kickoff
   - Systemd service per monitoring continuo

**Esempio Cron:**
```bash
# Avvia monitoring ogni ora
0 * * * * /usr/bin/python3 /path/to/start_live_monitoring.py >> /var/log/betting_monitor.log 2>&1
```

**Esempio Systemd Service:**
```ini
# /etc/systemd/system/betting-monitor.service
[Unit]
Description=Live Betting Monitor
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/Software-AsianOdds
ExecStart=/usr/bin/python3 start_live_monitoring.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 💡 Best Practices

1. **Test in Dry-Run**: Sempre testa con `--dry-run` prima di andare live
2. **Monitor Logs**: Controlla `live_monitoring.log` per errori
3. **Rate Limiting**: Non abbassare sotto 3 secondi (rischio ban Telegram)
4. **Backup Credentials**: Salva token in vault sicuro
5. **Alert Thresholds**: Inizia conservativo (EV > 8%) e aggiusta

---

## 📚 Riferimenti

- **Telegram Bot API**: https://core.telegram.org/bots/api
- **API-Football**: https://www.api-football.com/
- **AI System Guide**: `AI_SYSTEM_COMPLETE_GUIDE.md`
