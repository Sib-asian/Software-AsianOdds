# 🚀 Deploy Directory

Script e configurazioni per deployment 24/7 del sistema di betting automatico.

---

## 📁 Files

| File | Descrizione |
|------|-------------|
| `setup.sh` | **Script principale** - Setup automatico one-command |
| `betting-monitor.service.template` | Template systemd service |
| `test_setup.sh` | Test configurazione VPS |
| `README.md` | Questa guida |

---

## 🎯 Quick Start

### 1. Setup Systemd (Raccomandato)

```bash
chmod +x deploy/setup.sh
./deploy/setup.sh systemd
```

**Cosa fa:**
- ✅ Installa dipendenze sistema
- ✅ Crea virtual environment Python
- ✅ Installa dipendenze Python
- ✅ Configura .env (chiede token Telegram)
- ✅ Crea systemd service
- ✅ Abilita auto-start al boot
- ✅ Avvia monitoring

**Tempo:** 2-5 minuti

---

### 2. Setup Docker (Alternativa)

```bash
./deploy/setup.sh docker
```

**Cosa fa:**
- ✅ Installa Docker + docker-compose
- ✅ Configura .env
- ✅ Builda immagine Docker
- ✅ Avvia container con auto-restart

**Tempo:** 5-10 minuti (build image)

---

### 3. Setup Manual (Solo dipendenze)

```bash
./deploy/setup.sh manual
```

Utile per testing o se vuoi controllo manuale.

---

## 🔧 Configurazione

### Variabili d'Ambiente (.env)

Creato automaticamente da `setup.sh`:

```env
# Telegram (OBBLIGATORIO)
TELEGRAM_BOT_TOKEN="your_bot_token"
TELEGRAM_CHAT_ID="your_chat_id"

# Monitoring
LIVE_UPDATE_INTERVAL=60        # Controlla ogni 60s
MIN_EV_ALERT=5.0               # Alert se EV > 5%
MIN_CONFIDENCE=60.0            # Alert se confidence > 60%

# API (OPZIONALE)
API_FOOTBALL_KEY=""            # Lascia vuoto per solo API gratis

# Logging
LOG_LEVEL=INFO
```

**Modifica dopo setup:**
```bash
nano .env
sudo systemctl restart betting-monitor  # Se systemd
# O
docker-compose restart                  # Se docker
```

---

## 📊 Gestione Service

### Systemd

```bash
# Status
sudo systemctl status betting-monitor

# Start/Stop
sudo systemctl start betting-monitor
sudo systemctl stop betting-monitor

# Restart
sudo systemctl restart betting-monitor

# Logs
sudo journalctl -u betting-monitor -f

# Disable (non parte al boot)
sudo systemctl disable betting-monitor

# Enable (parte al boot)
sudo systemctl enable betting-monitor
```

### Docker

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Logs
docker-compose logs -f

# Rebuild
docker-compose build
docker-compose up -d
```

---

## 🧪 Testing

### Test Configurazione VPS

```bash
./deploy/test_setup.sh
```

Verifica:
- ✅ Connettività Telegram API
- ✅ Python version
- ✅ Dipendenze installate
- ✅ RAM disponibile
- ✅ Disk space
- ✅ Firewall configurato

### Test Dry-Run

```bash
# Test senza inviare notifiche
source venv/bin/activate
python start_live_monitoring.py --dry-run
```

---

## 🐛 Troubleshooting

### Service non si avvia

```bash
# Check logs
sudo journalctl -u betting-monitor -n 50

# Test manuale
source venv/bin/activate
python start_live_monitoring.py --dry-run

# Verifica permessi
ls -la .env
chmod 600 .env
```

### Notifiche non arrivano

```bash
# Test bot
curl -X POST https://api.telegram.org/bot<TOKEN>/sendMessage \
  -d "chat_id=<CHAT_ID>&text=Test"

# Verifica .env
cat .env | grep TELEGRAM

# Check firewall
sudo ufw status
# Deve permettere 443 outbound
```

### Out of Memory

```bash
# Check RAM
free -h

# Aggiungi swap se necessario
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📈 Update Sistema

### Update codice

```bash
git pull
sudo systemctl restart betting-monitor  # Systemd
# O
docker-compose build && docker-compose up -d  # Docker
```

### Update dipendenze

```bash
# Systemd
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart betting-monitor

# Docker
docker-compose build --no-cache
docker-compose up -d
```

---

## 🔒 Security

### File Permissions

```bash
# .env deve essere readable solo da owner
chmod 600 .env

# Script devono essere executable
chmod +x deploy/*.sh
```

### Firewall

```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Backup .env

**IMPORTANTE:** Non committare .env su GitHub!

```bash
# Backup locale
scp user@vps:~/Software-AsianOdds/.env ./backup-env-$(date +%Y%m%d)

# .gitignore già include .env
```

---

## 📚 Documentazione Completa

- **DEPLOYMENT_GUIDE.md** - Guida deployment completa
- **VPS_PROVIDERS_GUIDE.md** - Scegliere VPS gratis
- **LIVE_NOTIFICATIONS_GUIDE.md** - Setup Telegram
- **API_SETUP_GUIDE.md** - Setup API (opzionale)

---

## ✅ Checklist Post-Deploy

- [ ] Service running (`sudo systemctl status betting-monitor`)
- [ ] No errors in logs (`sudo journalctl -u betting-monitor -n 20`)
- [ ] Ricevuto messaggio Telegram di avvio
- [ ] Test restart OK (`sudo systemctl restart betting-monitor`)
- [ ] Auto-start abilitato (`sudo systemctl is-enabled betting-monitor`)
- [ ] Backup .env fatto
- [ ] Firewall configurato

**Se tutti ✅ → Sistema 24/7 LIVE! 🎉**

---

## 🆘 Support

**Problemi?**
1. Check logs: `sudo journalctl -u betting-monitor -f`
2. Test manuale: `python start_live_monitoring.py --dry-run`
3. Run test: `./deploy/test_setup.sh`
4. Vedi DEPLOYMENT_GUIDE.md troubleshooting section

**Issues:** https://github.com/Sib-asian/Software-AsianOdds/issues
