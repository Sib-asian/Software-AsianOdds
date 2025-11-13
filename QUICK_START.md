# 🚀 Quick Start - 24/7 Live Betting Monitor

Sistema completamente automatico che gira **24/7** anche con computer spento.

## ⚡ Setup in 2 Minuti

### Opzione A: Linux VPS (Raccomandato) ⭐

```bash
# 1. Connetti al tuo VPS
ssh root@your-vps-ip

# 2. Clone repository
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds
git checkout claude/implement-new-ai-011eNsNiUbswNBeHfeTxAHpA

# 3. Setup automatico
./deploy/setup.sh systemd

# 4. Verifica
sudo systemctl status betting-monitor
sudo journalctl -u betting-monitor -f
```

**✅ FATTO!** Il sistema ora gira 24/7 e invia notifiche Telegram automaticamente.

### Opzione B: Docker

```bash
# Clone repo
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds/deploy

# Build & run
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## 📱 Cosa Riceverai su Telegram

### 1. Opportunità Pre-Match
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
Dixon-Coles: 65.0% (30%)
XGBoost: 71.0% (40%)
LSTM: 68.0% (30%)
```

### 2. Alert Live Durante Partita
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

---

## 🎯 Come Funziona

1. **Monitoring Automatico**
   - Cerca partite live + partite nelle prossime 6h
   - Filtra solo leghe top (Premier, Serie A, Champions, ecc.)
   - Max 10 partite contemporaneamente

2. **Analisi AI**
   - Usa Dixon-Coles + XGBoost + LSTM ensemble
   - Calcola probabilità, EV, confidence
   - Identifica opportunità di valore

3. **Notifiche Telegram**
   - Invia solo se EV > 5% e Confidence > 60%
   - Rate limiting per evitare spam
   - Formattazione HTML professionale

4. **24/7 Operation**
   - Auto-restart se crashato
   - Auto-start al boot del server
   - Log automatici

---

## ⚙️ Comandi Utili

### Systemd (Linux)

```bash
# Status
sudo systemctl status betting-monitor

# Logs in tempo reale
sudo journalctl -u betting-monitor -f

# Start/Stop/Restart
sudo systemctl start betting-monitor
sudo systemctl stop betting-monitor
sudo systemctl restart betting-monitor

# Disable auto-start
sudo systemctl disable betting-monitor
```

### Docker

```bash
cd deploy

# Logs
docker-compose logs -f

# Stop/Start
docker-compose stop
docker-compose start

# Restart
docker-compose restart

# Remove
docker-compose down
```

---

## 🔧 Configurazione

### Cambia Intervallo Aggiornamento

**Systemd:**
```bash
sudo nano /etc/systemd/system/betting-monitor.service
# Cambia --interval 60 a --interval 120 (2 minuti)
sudo systemctl daemon-reload
sudo systemctl restart betting-monitor
```

**Docker:**
```bash
nano deploy/docker-compose.yml
# Cambia UPDATE_INTERVAL=60 a UPDATE_INTERVAL=120
docker-compose up -d
```

### Cambia Threshold Notifiche

```bash
nano ai_system/config.py

# Modifica:
telegram_min_ev: float = 10.0  # Da 5% a 10%
telegram_min_confidence: float = 75.0  # Da 60% a 75%
```

Restart servizio dopo la modifica.

---

## 🌐 VPS Raccomandati

Per far girare 24/7 senza PC acceso:

| Provider | Costo | RAM | Note |
|----------|-------|-----|------|
| **Oracle Cloud Free** | €0 | 1GB | Free forever! |
| **Hetzner** | €4/mo | 2GB | Best value |
| **DigitalOcean** | $6/mo | 512MB | Più facile |
| **Raspberry Pi** | €3-5/mo | 1-4GB | A casa, bassissimo consumo |

**Raccomandazione**: Oracle Cloud (gratis) o Hetzner (€4/mo)

---

## 📊 Monitoring Performance

### CPU e RAM Usage

```bash
# Real-time monitoring
htop

# Service resource usage
systemctl status betting-monitor
```

### API Quota

```bash
# Check API usage
sudo journalctl -u betting-monitor | grep "API"

# Sistema usa cache per ridurre chiamate
# Limite: 100 calls/day
# Cache: 24h per dati statici, 60s per dati live
```

### Database Stats

```bash
cd /home/Software-AsianOdds
sqlite3 api_cache.db

# View selection history
SELECT * FROM match_selections ORDER BY timestamp DESC LIMIT 10;

# View cache stats
SELECT * FROM cache_stats ORDER BY date DESC LIMIT 7;
```

---

## 🐛 Troubleshooting

### Notifiche non arrivano

```bash
# Test Telegram credentials
curl -X POST https://api.telegram.org/bot8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g/sendMessage \
  -d "chat_id=-1003278011521&text=Test"

# Check logs
sudo journalctl -u betting-monitor -n 100 | grep -i telegram
```

### Service non si avvia

```bash
# Check errors
sudo journalctl -u betting-monitor -n 50 --no-pager

# Test manual
cd /home/Software-AsianOdds
python3 start_live_monitoring.py --dry-run
```

### Nessuna partita monitorata

```bash
# Normale se:
# - È notte (3-10 AM)
# - Off-season (Giugno-Luglio)
# - Nessuna partita top leagues

# Check logs
sudo journalctl -u betting-monitor | grep "matches"
```

---

## 📚 Guide Complete

- **Deployment**: `deploy/DEPLOYMENT_GUIDE.md`
- **Notifiche**: `LIVE_NOTIFICATIONS_GUIDE.md`
- **AI System**: `AI_SYSTEM_COMPLETE_GUIDE.md`

---

## ❓ FAQ

**Q: Devo lasciare il PC acceso?**
A: NO! Sistema gira su VPS/server remoto 24/7.

**Q: Quanto costa?**
A: €0 (Oracle Free) o €4-6/mese (VPS).

**Q: Ricevo troppe notifiche?**
A: Aumenta threshold (telegram_min_ev da 5% a 10%).

**Q: Posso aggiungere altre leghe?**
A: Sì, modifica `ai_system/auto_match_selector.py`.

**Q: E se crashato?**
A: Auto-restart automatico entro 10 secondi.

---

## ✅ Checklist Post-Setup

- [ ] Service attivo: `sudo systemctl status betting-monitor`
- [ ] Logs senza errori: `sudo journalctl -u betting-monitor -f`
- [ ] Test notifica ricevuta su Telegram
- [ ] Auto-start abilitato: `sudo systemctl is-enabled betting-monitor`
- [ ] Bookmarked comandi utili

---

## 🎉 Congratulazioni!

Il tuo sistema è ora **completamente automatico** e gira 24/7!

**Prossimi passi**:
1. Monitora logs per 24h
2. Aspetta prima partita e verifica notifica
3. (Opzionale) Setup auto-update con cron

**Supporto**:
- Leggi guide complete in `deploy/`
- Check logs per troubleshooting
- Verifica Telegram bot con `/start`

---

**Enjoy le notifiche automatiche! 🚀**
