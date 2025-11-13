# 🚀 Deployment Guide - 24/7 Live Betting Monitor

Guida completa per far girare il sistema di notifiche Telegram **24/7** anche con computer/cellulare spento.

## 📋 Table of Contents

1. [Prerequisiti](#prerequisiti)
2. [Metodo 1: Systemd (Linux VPS)](#metodo-1-systemd-linux-vps) ⭐ **RACCOMANDATO**
3. [Metodo 2: Docker](#metodo-2-docker)
4. [Metodo 3: Cloud Hosting](#metodo-3-cloud-hosting)
5. [Metodo 4: Raspberry Pi](#metodo-4-raspberry-pi)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisiti

### 1. Server/VPS Linux (raccomandato)

**Opzioni economiche per 24/7:**
- **DigitalOcean Droplet** - $6/mese (512MB RAM, sufficiente)
- **Hetzner Cloud** - €4/mese (2GB RAM)
- **AWS Free Tier** - Gratis primo anno (t2.micro)
- **Oracle Cloud Free Tier** - Gratis forever (2x VM)
- **Contabo** - €5/mese (4GB RAM)

### 2. Requisiti Software

- **Linux**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **Python**: 3.8+
- **RAM**: Minimo 512MB, raccomandato 1GB
- **Storage**: 2GB disponibili
- **Network**: Connessione stabile

### 3. Telegram Bot Token

Già configurato:
```
TELEGRAM_BOT_TOKEN="8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
TELEGRAM_CHAT_ID="-1003278011521"
```

---

## Metodo 1: Systemd (Linux VPS) ⭐

**Vantaggi:**
- ✅ Avvio automatico al boot
- ✅ Riavvio automatico se crashato
- ✅ Log integrati con journald
- ✅ Resource limits
- ✅ Gestione nativa Linux

### Step 1: Preparazione VPS

```bash
# Connetti al VPS
ssh root@your-vps-ip

# Update sistema
apt update && apt upgrade -y

# Installa dipendenze
apt install -y python3 python3-pip git

# Clone repository
cd /home
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Checkout branch
git checkout claude/implement-new-ai-011eNsNiUbswNBeHfeTxAHpA
```

### Step 2: Setup Automatico

```bash
# Esegui setup script
cd /home/Software-AsianOdds
chmod +x deploy/setup.sh
./deploy/setup.sh systemd
```

Lo script:
1. ✅ Installa dipendenze Python
2. ✅ Configura systemd service
3. ✅ Abilita avvio automatico
4. ✅ Avvia il monitor

### Step 3: Verifica

```bash
# Check status
sudo systemctl status betting-monitor

# Dovrebbe mostrare:
# ● betting-monitor.service - AsianOdds Live Betting Monitor
#    Loaded: loaded
#    Active: active (running)

# View live logs
sudo journalctl -u betting-monitor -f
```

### Step 4: Comandi Utili

```bash
# Start/Stop/Restart
sudo systemctl start betting-monitor
sudo systemctl stop betting-monitor
sudo systemctl restart betting-monitor

# Enable/Disable auto-start
sudo systemctl enable betting-monitor
sudo systemctl disable betting-monitor

# View logs (last 100 lines)
sudo journalctl -u betting-monitor -n 100

# View logs since today
sudo journalctl -u betting-monitor --since today

# Follow logs in real-time
sudo journalctl -u betting-monitor -f
```

---

## Metodo 2: Docker

**Vantaggi:**
- ✅ Isolato dal sistema
- ✅ Portable
- ✅ Facile da aggiornare
- ✅ Resource limits

### Step 1: Installa Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install -y docker-compose

# Verifica
docker --version
docker-compose --version
```

### Step 2: Setup Automatico

```bash
cd /home/Software-AsianOdds
./deploy/setup.sh docker
```

### Step 3: Comandi Docker

```bash
cd /home/Software-AsianOdds/deploy

# View logs
docker-compose logs -f

# Stop
docker-compose stop

# Start
docker-compose start

# Restart
docker-compose restart

# Rebuild (after code changes)
docker-compose up -d --build

# Remove container
docker-compose down
```

### Step 4: Auto-Start al Boot

```bash
# Docker si avvia automaticamente con --restart=unless-stopped
# Configurato in docker-compose.yml

# Verifica
docker ps
# Dovrebbe mostrare container "asianodds-monitor" running
```

---

## Metodo 3: Cloud Hosting

### AWS EC2

1. **Launch Instance**
   - AMI: Ubuntu 22.04 LTS
   - Type: t2.micro (free tier)
   - Security Group: Allow SSH (22), HTTPS (443)

2. **Connect & Setup**
   ```bash
   ssh -i your-key.pem ubuntu@ec2-instance-ip
   sudo su
   cd /home
   # Follow Systemd steps above
   ```

3. **Enable Auto-Start**
   - Systemd service runs automatically on boot

### Google Cloud Platform

```bash
# Create VM
gcloud compute instances create betting-monitor \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# Connect
gcloud compute ssh betting-monitor

# Follow setup
```

### Oracle Cloud (Free Forever)

1. Create Always Free VM (ARM or AMD)
2. SSH into instance
3. Follow Systemd setup

---

## Metodo 4: Raspberry Pi

**Perfetto per 24/7 a basso costo (consumo ~3W)**

### Setup Raspberry Pi

```bash
# Update Raspberry Pi OS
sudo apt update && sudo apt upgrade -y

# Clone repo
cd /home/pi
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Setup
./deploy/setup.sh systemd
```

### Enable Auto-Start on Boot

```bash
# Systemd service già configurato per auto-start
sudo systemctl enable betting-monitor
```

---

## Monitoring & Maintenance

### 1. Health Check

Create cron job per verificare che il servizio sia attivo:

```bash
# Edit crontab
crontab -e

# Add this line (check every 5 minutes)
*/5 * * * * systemctl is-active --quiet betting-monitor || systemctl restart betting-monitor
```

### 2. Log Rotation

Evita che i log riempiano il disco:

```bash
# Create log rotation config
sudo cat > /etc/logrotate.d/betting-monitor << EOF
/var/log/betting-monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

### 3. Monitoring Dashboard (Opzionale)

```bash
# Install htop for process monitoring
apt install -y htop

# View processes
htop
# Look for "python3 start_live_monitoring.py"
```

### 4. Telegram Bot Monitoring

Il bot stesso ti notificherà:
- ✅ Quando si avvia
- ✅ Quando trova opportunità
- ❌ Errori critici (opzionale)

### 5. Resource Usage

```bash
# Check memory usage
free -h

# Check CPU usage
top

# Check disk space
df -h

# Check service resource usage
systemctl status betting-monitor
```

---

## Auto-Update System

### Script di Auto-Update

Create `/home/Software-AsianOdds/auto-update.sh`:

```bash
#!/bin/bash
cd /home/Software-AsianOdds
git pull origin claude/implement-new-ai-011eNsNiUbswNBeHfeTxAHpA
pip3 install -r requirements.txt --upgrade
systemctl restart betting-monitor
```

```bash
chmod +x auto-update.sh

# Add to cron (update daily at 3 AM)
crontab -e
# Add: 0 3 * * * /home/Software-AsianOdds/auto-update.sh >> /var/log/auto-update.log 2>&1
```

---

## Troubleshooting

### Service non si avvia

```bash
# Check logs
sudo journalctl -u betting-monitor -n 50 --no-pager

# Check status
sudo systemctl status betting-monitor

# Test manual run
cd /home/Software-AsianOdds
python3 start_live_monitoring.py --dry-run
```

### Telegram notifiche non arrivano

```bash
# Test credentials
curl -X POST https://api.telegram.org/bot8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g/sendMessage \
  -d "chat_id=-1003278011521&text=Test from VPS"

# If fails, check:
# 1. Bot token is correct
# 2. Chat ID is correct
# 3. Bot has been started (/start)
# 4. Network connectivity (ping api.telegram.org)
```

### Out of Memory

```bash
# Check memory
free -h

# If low, add swap
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### API Quota Exceeded

```bash
# Check API usage in logs
sudo journalctl -u betting-monitor | grep "API"

# System automatically uses cache to reduce API calls
# Default: 100 calls/day limit

# Adjust update interval if needed
sudo nano /etc/systemd/system/betting-monitor.service
# Change --interval 60 to --interval 120 (2 minutes)
sudo systemctl daemon-reload
sudo systemctl restart betting-monitor
```

### Permission Denied

```bash
# Fix ownership
sudo chown -R user:user /home/Software-AsianOdds

# Fix service user
sudo nano /etc/systemd/system/betting-monitor.service
# Change User=user to your actual username
sudo systemctl daemon-reload
sudo systemctl restart betting-monitor
```

---

## Performance Optimization

### 1. Reduce API Calls

```bash
# Edit config
nano /home/Software-AsianOdds/ai_system/config.py

# Increase cache TTL
api_cache_ttl: int = 86400  # 24 hours

# Increase update interval
live_update_interval: int = 120  # 2 minutes instead of 1
```

### 2. Prioritize Matches

```bash
# Only monitor top leagues
nano /home/Software-AsianOdds/ai_system/auto_match_selector.py

# Edit priority_leagues to only include your favorites:
self.priority_leagues = [
    'Premier League',
    'Serie A',
    'Champions League'
]
```

---

## Security Best Practices

### 1. Firewall Setup

```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS (if adding web dashboard)
sudo ufw enable

# Check status
sudo ufw status
```

### 2. SSH Key Authentication

```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

### 3. Environment Variables

```bash
# Never commit .env with real tokens to git
echo ".env" >> .gitignore

# Store secrets in systemd service (already configured)
sudo nano /etc/systemd/system/betting-monitor.service
```

---

## Recommended VPS Setup (Complete)

### DigitalOcean Droplet ($6/mese)

```bash
# 1. Create Droplet
# - OS: Ubuntu 22.04 LTS
# - Plan: Basic - $6/mo (512MB RAM, 10GB SSD)
# - Region: Closest to you

# 2. Initial setup
ssh root@your-droplet-ip

apt update && apt upgrade -y
apt install -y python3 python3-pip git htop

# 3. Create user
adduser betting
usermod -aG sudo betting
su - betting

# 4. Clone repo
cd ~
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds
git checkout claude/implement-new-ai-011eNsNiUbswNBeHfeTxAHpA

# 5. Setup
./deploy/setup.sh systemd

# 6. Verify
sudo systemctl status betting-monitor
sudo journalctl -u betting-monitor -f

# Done! System now runs 24/7
```

---

## Cost Breakdown

### Monthly Costs for 24/7 Operation

| Metodo | Costo/Mese | Note |
|--------|------------|------|
| **Raspberry Pi** | €3-5 | Elettricità (~3W), costo iniziale €50-80 |
| **Oracle Cloud Free** | €0 | Always Free tier |
| **Hetzner Cloud** | €4 | 2GB RAM |
| **DigitalOcean** | $6 | 512MB RAM |
| **AWS Free Tier** | €0 | Primo anno, poi ~$8/mo |
| **Contabo** | €5 | 4GB RAM |

**Raccomandazione**:
- **Best Value**: Oracle Cloud Free Tier (€0 forever)
- **Easiest**: DigitalOcean ($6/mo, ottimo supporto)
- **Lowest Power**: Raspberry Pi a casa (€3-5/mo elettricità)

---

## FAQ

**Q: Posso usare il mio PC a casa 24/7?**
A: Sì, ma sconsigliato. PC desktop consuma 100-300W (€20-60/mese), vs VPS €4-6/mese.

**Q: Cosa succede se il server crashato?**
A: Systemd riavvia automaticamente il servizio entro 10 secondi.

**Q: Posso monitorare più bot contemporaneamente?**
A: Sì, copia il service file e modifica nome/porta.

**Q: Come aggiorno il codice?**
A: `git pull && systemctl restart betting-monitor`

**Q: Posso vedere dashboard web?**
A: Al momento no, ma puoi aggiungere Grafana (opzionale in docker-compose).

---

## Next Steps

Dopo il deployment:

1. ✅ Verifica che ricevi notifiche Telegram
2. ✅ Monitora logs per 24h per assicurarti che funzioni
3. ✅ Setup auto-update script (opzionale)
4. ✅ Aggiungi health check alerts (opzionale)
5. ✅ Configura backup automatici (opzionale)

---

**🎉 Congratulazioni!** Il tuo sistema è ora completamente automatico e gira 24/7 anche con cellulare/computer spento!
