# 🚀 Deployment Guide - Sistema 24/7 GRATIS

Guida completa per deployare il sistema di betting automatico su VPS **gratis** o low-cost.

---

## 📋 Indice

1. [Quick Start (2 minuti)](#quick-start)
2. [Provider VPS Gratuiti](#provider-vps-gratuiti)
3. [Setup VPS](#setup-vps)
4. [Deployment Methods](#deployment-methods)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring](#monitoring)

---

## 🎯 Quick Start

### Prerequisiti

- ✅ Account VPS (vedi sotto per opzioni gratuite)
- ✅ Telegram Bot Token (da @BotFather)
- ✅ Telegram Chat ID (da @userinfobot)

### 1. Scegli VPS Gratis

**Opzione A: Oracle Cloud (CONSIGLIATO - Forever Free)**
- RAM: 1GB
- CPU: 1 core
- Storage: 50GB
- Costo: **€0/mese FOREVER**
- Sign up: https://www.oracle.com/cloud/free/

**Opzione B: Google Cloud (300$ credito per 90 giorni)**
- RAM: 0.6GB (e2-micro)
- CPU: 0.25-2 cores
- Storage: 30GB
- Costo: **€0 per 3 mesi**, poi ~€5/mese
- Sign up: https://cloud.google.com/free

**Opzione C: AWS Free Tier (12 mesi gratis)**
- RAM: 1GB (t2.micro)
- CPU: 1 core
- Storage: 30GB
- Costo: **€0 per 12 mesi**, poi ~€8/mese
- Sign up: https://aws.amazon.com/free/

### 2. Crea VPS

**Oracle Cloud (esempio):**

1. Vai su https://cloud.oracle.com/
2. Sign up (serve carta di credito, NON verrà addebitato nulla)
3. Crea Compute Instance:
   - Shape: VM.Standard.E2.1.Micro (Always Free)
   - Image: Ubuntu 22.04
   - Boot volume: 50GB
4. Salva la chiave SSH privata (scarica .pem)
5. Annota IP pubblico

### 3. Connettiti a VPS

```bash
# Linux/Mac
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_VPS_IP

# Windows (usa PuTTY o Windows Terminal)
ssh ubuntu@YOUR_VPS_IP
```

### 4. Deploy ONE-COMMAND

```bash
# Su VPS, esegui:
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Setup automatico (scelta consigliata: systemd)
chmod +x deploy/setup.sh
./deploy/setup.sh systemd
```

**Durante il setup ti chiederà:**
- Telegram Bot Token → Incolla token da @BotFather
- Telegram Chat ID → Incolla ID da @userinfobot

**In 2-5 minuti è tutto fatto!** ✅

---

## 🏗️ Deployment Methods

### Metodo 1: Systemd (CONSIGLIATO)

**Vantaggi:**
- ✅ Auto-start al boot
- ✅ Auto-restart su crash
- ✅ Log centralizati
- ✅ Controllo facile

**Setup:**
```bash
./deploy/setup.sh systemd
```

**Comandi utili:**
```bash
# Status
sudo systemctl status betting-monitor

# Start/Stop
sudo systemctl start betting-monitor
sudo systemctl stop betting-monitor

# Restart
sudo systemctl restart betting-monitor

# Logs live
sudo journalctl -u betting-monitor -f

# Logs ultimi 100 righe
sudo journalctl -u betting-monitor -n 100

# Disabilita (non parte più al boot)
sudo systemctl disable betting-monitor
```

---

### Metodo 2: Docker

**Vantaggi:**
- ✅ Isolamento completo
- ✅ Facile update
- ✅ Portable
- ✅ Resource limits

**Setup:**
```bash
./deploy/setup.sh docker
```

**Comandi utili:**
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Logs live
docker-compose logs -f

# Status
docker-compose ps

# Update
git pull
docker-compose build
docker-compose up -d
```

---

### Metodo 3: Manual (Per testing)

**Setup:**
```bash
./deploy/setup.sh manual

# Poi avvia manualmente
source venv/bin/activate
python start_live_monitoring.py
```

**Mantenerlo running:**
```bash
# Usa screen o tmux
sudo apt install screen
screen -S betting
python start_live_monitoring.py

# Detach: Ctrl+A, poi D
# Riattacca: screen -r betting
```

---

## ⚙️ Configurazione

### File .env

Dopo il setup, modifica `.env` per personalizzare:

```bash
nano .env
```

```env
# Telegram (OBBLIGATORIO)
TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
TELEGRAM_CHAT_ID="-1001234567890"

# Monitoring
LIVE_UPDATE_INTERVAL=60        # Controlla ogni 60 secondi
MIN_EV_ALERT=5.0               # Alert solo se EV > 5%
MIN_CONFIDENCE=60.0            # Alert solo se confidence > 60%

# API (OPZIONALE - TheSportsDB è gratis)
API_FOOTBALL_KEY=""            # Lascia vuoto per usare solo API gratis

# Logging
LOG_LEVEL=INFO                 # DEBUG per più dettagli
```

**Dopo modifica .env:**
```bash
# Systemd
sudo systemctl restart betting-monitor

# Docker
docker-compose restart
```

---

## 📊 Monitoring & Logs

### Check Status

**Systemd:**
```bash
# Status completo
sudo systemctl status betting-monitor

# Output:
# ● betting-monitor.service - Asian Odds Betting Monitor
#    Loaded: loaded
#    Active: active (running) since ...
#    ...
```

**Docker:**
```bash
docker-compose ps

# Output:
# NAME                  STATUS    PORTS
# asian-odds-monitor    Up 2 hours
```

### Logs

**Systemd:**
```bash
# Live logs (Ctrl+C per uscire)
sudo journalctl -u betting-monitor -f

# Ultimi 100 log
sudo journalctl -u betting-monitor -n 100

# Logs di oggi
sudo journalctl -u betting-monitor --since today

# Logs con errori
sudo journalctl -u betting-monitor -p err
```

**Docker:**
```bash
# Live logs
docker-compose logs -f

# Ultimi 100
docker-compose logs --tail=100

# Solo errori
docker-compose logs | grep ERROR
```

**Log file diretto:**
```bash
# File log principale
tail -f monitor.log

# O
tail -f live_monitoring.log
```

---

## 🐛 Troubleshooting

### Service non si avvia

**1. Check logs:**
```bash
sudo journalctl -u betting-monitor -n 50
```

**2. Verifica configurazione:**
```bash
# .env esiste?
ls -la .env

# Token Telegram validi?
cat .env | grep TELEGRAM

# Test manuale
source venv/bin/activate
python start_live_monitoring.py --dry-run
```

**3. Permessi:**
```bash
# Fix permessi
chmod +x deploy/setup.sh
chmod 600 .env
```

### Notifiche non arrivano

**1. Test bot manualmente:**
```bash
curl -X POST https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage \
  -H "Content-Type: application/json" \
  -d '{"chat_id":"<YOUR_CHAT_ID>","text":"Test"}'
```

**2. Verifica bot settings:**
- Bot deve essere avviato (invia `/start` al bot)
- Se gruppo: bot deve essere admin

**3. Check firewall:**
```bash
# Oracle Cloud: apri outbound port 443
# Vedi VPS_PROVIDERS_GUIDE.md per dettagli
```

### Troppo CPU/RAM

**1. Riduci frequenza update:**
```bash
nano .env
# Cambia LIVE_UPDATE_INTERVAL=60 a 120
```

**2. Limita risorse (Docker):**
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 256M
```

**3. Disabilita features:**
```python
# ai_system/config.py
ensemble_enabled = False  # Usa solo un modello
cache_predictions = True   # Abilita cache
```

### Out of Memory

**1. Check RAM usage:**
```bash
free -h
top
```

**2. Aggiungi swap (solo se necessario):**
```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Rendi permanente
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 🔄 Update Sistema

### Git pull + restart

```bash
# Systemd
cd Software-AsianOdds
git pull
sudo systemctl restart betting-monitor

# Docker
cd Software-AsianOdds
git pull
docker-compose build
docker-compose up -d
```

### Update dipendenze

```bash
# Systemd
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart betting-monitor

# Docker
# Rebuild automatically updates dependencies
docker-compose build --no-cache
docker-compose up -d
```

---

## 🔒 Security Best Practices

### 1. Firewall

```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS (Telegram API)
sudo ufw enable

# Check
sudo ufw status
```

### 2. SSH Key Only (no password)

```bash
sudo nano /etc/ssh/sshd_config

# Cambia:
PasswordAuthentication no
PubkeyAuthentication yes

# Restart SSH
sudo systemctl restart sshd
```

### 3. Auto-updates

```bash
# Ubuntu
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 4. Backup .env

```bash
# Backup locale sicuro (NON committare su GitHub!)
scp ubuntu@YOUR_VPS_IP:~/Software-AsianOdds/.env ./backup-env
```

---

## 📈 Monitoring Avanzato

### 1. Check Resource Usage

```bash
# CPU, RAM, Disk
htop

# Network
sudo apt install nethogs
sudo nethogs

# Disk space
df -h
du -sh Software-AsianOdds/
```

### 2. Alert su Crash

Aggiungi in `betting-monitor.service`:

```ini
[Service]
# ... existing config ...

# Send email on failure
OnFailure=status-email@%n.service
```

### 3. Uptime monitoring

Usa servizi gratuiti tipo:
- UptimeRobot (https://uptimerobot.com/) - 50 monitor gratis
- Healthchecks.io (https://healthchecks.io/) - 20 checks gratis

```python
# Aggiungi in start_live_monitoring.py
import requests

# Ogni 5 minuti
requests.get("https://hc-ping.com/YOUR-UUID")
```

---

## 💰 Costi Previsti

| Setup | VPS | APIs | Totale/mese |
|-------|-----|------|-------------|
| **Oracle Free** | €0 | €0 (TheSportsDB) | **€0** ✅ |
| **Google Cloud (primi 3 mesi)** | €0 | €0 | **€0** ✅ |
| **AWS Free Tier (primi 12 mesi)** | €0 | €0 | **€0** ✅ |
| **Hetzner CX11** | €4 | €0 | €4 |
| **DigitalOcean** | €6 | €0 | €6 |
| **Contabo VPS S** | €5 | €0 | €5 |

**API Costs (opzionale):**
- TheSportsDB: **Gratis** ✅ (unlimited)
- API-Football Free: **Gratis** (100 calls/day)
- API-Football Pro: €12/mese (unlimited, non necessario)

---

## 🎓 Best Practices

1. **Inizia con Oracle Free** → Nessun costo, nessun rischio
2. **Usa Systemd** → Più stabile e semplice di Docker per VPS small
3. **Abilita auto-updates** → Sicurezza sempre aggiornata
4. **Monitor logs** → Controlla prima settimana ogni giorno
5. **Backup .env** → Token Telegram al sicuro
6. **Start conservativo** → MIN_EV_ALERT=8.0 all'inizio, poi abbassa

---

## 🆘 Support

**Problemi comuni:**
- [GitHub Issues](https://github.com/Sib-asian/Software-AsianOdds/issues)
- Logs: `sudo journalctl -u betting-monitor -f`
- Test manuale: `python start_live_monitoring.py --dry-run`

**Documentazione:**
- `LIVE_NOTIFICATIONS_GUIDE.md` - Setup Telegram
- `VPS_PROVIDERS_GUIDE.md` - Dettagli provider VPS
- `API_SETUP_GUIDE.md` - Setup API

---

## ✅ Checklist Post-Deploy

- [ ] VPS accessibile via SSH
- [ ] Service running (`sudo systemctl status betting-monitor`)
- [ ] Logs non mostrano errori (`sudo journalctl -u betting-monitor -n 20`)
- [ ] Ricevuto messaggio Telegram di avvio
- [ ] Testato riavvio (`sudo systemctl restart betting-monitor`)
- [ ] Backup .env fatto
- [ ] Firewall configurato
- [ ] Auto-updates abilitati

**Se tutti ✅ → Sei LIVE 24/7! 🚀**

---

## 🎉 Next Steps

1. **Monitora primi giorni** → Check logs ogni giorno
2. **Adjust thresholds** → Modifica MIN_EV_ALERT in .env
3. **Add matches** → Implementa `get_matches_to_monitor()` con tua logica
4. **Connect API live** → Implementa `fetch_live_data_from_api()` per dati real-time

Enjoy your automated 24/7 betting system! 🎰
