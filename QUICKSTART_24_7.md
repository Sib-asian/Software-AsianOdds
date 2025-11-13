# ⚡ QUICKSTART 24/7 - Sistema Automatico in 5 Minuti

Guida **super veloce** per avere il sistema betting automatico **attivo 24/7** anche con PC spento.

---

## 🎯 Cosa Otterrai

✅ Sistema betting automatico **sempre attivo**
✅ Notifiche Telegram **automatiche** ogni opportunità
✅ Auto-restart su crash
✅ Auto-start al boot VPS
✅ **Costo: €0/mese** (con VPS gratis)
✅ **Setup: 5 minuti**

---

## 🚀 Setup in 3 Step

### Step 1: Crea VPS Gratis (2 minuti)

**OPZIONE A: Oracle Cloud (Forever Free - CONSIGLIATO)**

1. Vai su https://www.oracle.com/cloud/free/
2. Sign up (serve carta, **NON verrà addebitato nulla**)
3. Crea Instance:
   - Shape: **VM.Standard.E2.1.Micro** (Always Free)
   - Image: **Ubuntu 22.04**
   - Boot volume: 50GB
4. Scarica chiave SSH (.pem file)
5. **Copia IP pubblico**

**OPZIONE B: Google Cloud (€300 credito)**

1. https://cloud.google.com/free
2. Sign up
3. Crea VM:
   - Region: **us-central1** (free tier)
   - Machine: **e2-micro** (free tier)
   - Image: Ubuntu 22.04
4. Click "SSH" per connetterti

**OPZIONE C: Hetzner (€4/mese)**

1. https://www.hetzner.com/cloud
2. Create Server
3. Location: Nuremberg
4. Type: **CX11** (€4.15/mese, 2GB RAM)

> 💡 **Consiglio:** Inizia con Oracle (gratis forever), poi scala a Hetzner se serve più potenza

---

### Step 2: Configura Telegram Bot (1 minuto)

**2.1 Crea Bot:**
```
1. Apri Telegram
2. Cerca @BotFather
3. Invia /newbot
4. Segui istruzioni
5. SALVA il token (es: 123456:ABC-DEF...)
```

**2.2 Ottieni Chat ID:**
```
1. Cerca @userinfobot
2. Invia /start
3. SALVA il tuo ID (es: -1001234567890)
```

---

### Step 3: Deploy (2 minuti)

**3.1 Connettiti a VPS:**

```bash
# Linux/Mac
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_VPS_IP

# Windows (PowerShell o Windows Terminal)
ssh ubuntu@YOUR_VPS_IP
```

**3.2 Deploy ONE-COMMAND:**

```bash
# Clone repo
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Setup automatico (scelta: systemd)
chmod +x deploy/setup.sh
./deploy/setup.sh systemd
```

**Durante il setup ti chiederà:**

```
Telegram Bot Token: [Incolla token da @BotFather]
Telegram Chat ID: [Incolla ID da @userinfobot]
Start monitoring now? (y/n): y
```

**FATTO! 🎉**

In 30-60 secondi riceverai messaggio Telegram:

```
🤖 LIVE MONITORING STARTED

System is now monitoring live matches and will send
notifications for:
• Value opportunities (EV > 5%)
• Significant probability shifts
• Optimal betting timing
• Important match events

Status: ACTIVE ✅
```

---

## ✅ Verifica che Funzioni

```bash
# Check status
sudo systemctl status betting-monitor

# Output:
# ● betting-monitor.service - Asian Odds Betting Monitor
#    Loaded: loaded
#    Active: active (running)  ← DEVE DIRE QUESTO
```

**Logs live:**
```bash
sudo journalctl -u betting-monitor -f
```

**Test restart:**
```bash
sudo systemctl restart betting-monitor
# Dovresti ricevere notifica Telegram "MONITORING STARTED"
```

---

## 🎮 Comandi Utili

```bash
# Start/Stop
sudo systemctl start betting-monitor
sudo systemctl stop betting-monitor

# Restart
sudo systemctl restart betting-monitor

# Status
sudo systemctl status betting-monitor

# Logs live
sudo journalctl -u betting-monitor -f

# Ultimi 100 log
sudo journalctl -u betting-monitor -n 100
```

---

## ⚙️ Configurazione (Opzionale)

Modifica impostazioni:

```bash
nano .env
```

**Parametri principali:**

```env
# Frequenza controllo (secondi)
LIVE_UPDATE_INTERVAL=60        # Default: 60s (1 minuto)

# Soglia alert EV (%)
MIN_EV_ALERT=5.0               # Default: 5% (alert se EV > 5%)

# Soglia confidenza (%)
MIN_CONFIDENCE=60.0            # Default: 60%

# Log level
LOG_LEVEL=INFO                 # DEBUG per più dettagli
```

**Dopo modifica:**
```bash
sudo systemctl restart betting-monitor
```

---

## 🐛 Troubleshooting Veloce

### ❌ Service non si avvia

```bash
# Check errori
sudo journalctl -u betting-monitor -n 50

# Test manuale
source venv/bin/activate
python start_live_monitoring.py --dry-run
```

### ❌ Notifiche non arrivano

```bash
# Test bot manualmente
curl -X POST https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage \
  -d "chat_id=<YOUR_CHAT_ID>&text=Test"

# Verifica .env
cat .env | grep TELEGRAM

# Check firewall (Oracle Cloud)
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

### ❌ Out of Memory

```bash
# Aggiungi swap (solo se RAM < 1GB)
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 🎓 Best Practices

**Primi giorni:**
- ✅ Controlla logs ogni giorno: `sudo journalctl -u betting-monitor -f`
- ✅ Verifica notifiche Telegram arrivano
- ✅ Testa restart: `sudo systemctl restart betting-monitor`

**Dopo 1 settimana:**
- ✅ Adjust MIN_EV_ALERT se troppi/pochi alert
- ✅ Check resource usage: `htop` e `df -h`
- ✅ Backup .env: `scp ubuntu@VPS:.env ./backup-env`

**Maintenance:**
- ✅ Update ogni 2-4 settimane: `git pull && sudo systemctl restart betting-monitor`
- ✅ Check logs 1x/settimana per errori
- ✅ Monitor VPS uptime (usa UptimeRobot gratis)

---

## 💰 Costi Finali

| Componente | Costo |
|------------|-------|
| **VPS Oracle** | €0/mese (forever) ✅ |
| **Telegram Bot** | €0 (gratis) ✅ |
| **TheSportsDB API** | €0 (unlimited gratis) ✅ |
| **API-Football** | €0 (100 calls/day gratis) ✅ |
| **TOTALE** | **€0/mese** 🎉 |

**Alternative VPS:**
- Google Cloud: €0 per 3 mesi (poi €5-6/mese)
- AWS: €0 per 12 mesi (poi €8/mese)
- Hetzner: €4/mese (2GB RAM, più potente)

---

## 📱 Cosa Riceverai su Telegram

**Pre-Match Opportunities:**
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
```

**Live Updates:**
```
📈 VALUE OPPORTUNITY DETECTED

🔴 LIVE - 35'
Score: 1-0
Updated Win Probability: 72.1%
```

**Daily Summary:**
```
📊 DAILY REPORT

Matches monitored: 23
Opportunities found: 7
Total potential EV: +€142.50
```

---

## 🎉 FATTO!

**Il tuo sistema è ora:**
- ✅ **Attivo 24/7** (anche con PC spento)
- ✅ **Auto-restart** su crash
- ✅ **Auto-start** al boot
- ✅ **Notifiche automatiche** ogni opportunità
- ✅ **Costo €0/mese** (con Oracle Free)

**Puoi:**
- 🏖️ Andare in vacanza
- 😴 Dormire tranquillo
- 💻 Spegnere il PC
- 📱 Ricevere alert ovunque

**Il bot GIRA SEMPRE! 🚀**

---

## 📚 Documentazione Completa

Guide dettagliate:

- **DEPLOYMENT_GUIDE.md** - Deployment completo con troubleshooting
- **VPS_PROVIDERS_GUIDE.md** - Confronto dettagliato provider VPS
- **LIVE_NOTIFICATIONS_GUIDE.md** - Setup Telegram avanzato
- **API_SETUP_GUIDE.md** - Configurazione API (opzionale)

Deploy directory:
- **deploy/README.md** - Comandi e gestione service
- **deploy/setup.sh** - Script setup automatico
- **deploy/test_setup.sh** - Test configurazione VPS

---

## 🆘 Support

**Problemi?**

1. Check logs: `sudo journalctl -u betting-monitor -f`
2. Test setup: `./deploy/test_setup.sh`
3. Vedi troubleshooting in DEPLOYMENT_GUIDE.md
4. GitHub Issues: https://github.com/Sib-asian/Software-AsianOdds/issues

---

## 🚀 Next Level

Dopo setup base, puoi:

1. **Connetti API live** - Dati real-time match (vedi LIVE_NOTIFICATIONS_GUIDE.md)
2. **Auto-fetch partite** - Sistema seleziona match automaticamente
3. **Multiple bots** - Diversi bot per diverse strategie
4. **Dashboard web** - Streamlit dashboard per monitoring
5. **Backtest** - Test strategie su dati storici

**Per ora: Goditi le notifiche automatiche! 🎰**
