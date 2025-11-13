# 🌐 VPS Provider Guide - Scegli il Tuo Server Gratis

Guida dettagliata ai migliori provider VPS **gratuiti** e low-cost per il betting monitor 24/7.

---

## 📊 Tabella Comparativa

| Provider | RAM | CPU | Storage | Bandwidth | Costo/mese | Free Period | Note |
|----------|-----|-----|---------|-----------|------------|-------------|------|
| **Oracle Cloud** | 1GB | 1 | 50GB | 10TB | **€0** | **Forever** ⭐ | Migliore gratis |
| **Google Cloud** | 0.6GB | 0.25 | 30GB | 1GB egress | **€0** | 90 giorni | €300 credito |
| **AWS** | 1GB | 1 | 30GB | 15GB | **€0** | 12 mesi | Poi ~€8/mese |
| **Azure** | 1GB | 1 | 64GB | 15GB | **€0** | 12 mesi | €200 credito |
| **Hetzner** | 2GB | 1 | 20GB SSD | 20TB | €4.15 | - | Ottimo valore |
| **DigitalOcean** | 1GB | 1 | 25GB SSD | 1TB | €6 | - | €200 credito studenti |
| **Contabo** | 4GB | 2 | 50GB SSD | 32TB | €5 | - | Più potente |
| **Linode** | 1GB | 1 | 25GB SSD | 1TB | €6 | - | €100 credito |
| **Vultr** | 1GB | 1 | 25GB SSD | 1TB | €6 | - | €100 credito |

---

## 🏆 TOP PICK: Oracle Cloud (Forever Free)

### ✅ Perché Oracle?

- **€0/mese FOREVER** (non scade mai!)
- RAM sufficiente (1GB)
- Storage generoso (50GB)
- Bandwidth altissimo (10TB/mese)
- 2 VM gratis inclusi nel Always Free tier
- No carta di credito richiesta dopo trial

### 📝 Setup Step-by-Step

#### 1. Sign Up

1. Vai su https://www.oracle.com/cloud/free/
2. Click "Start for Free"
3. Compila form:
   - Email
   - Password
   - Country: Italy
   - Cloud Account Name (es: `mybetting`)
4. Verifica email

#### 2. Add Payment (No Addebito)

**IMPORTANTE:** Oracle richiede carta per verifica identità, ma **NON addebita nulla** nel Always Free tier.

1. Aggiungi carta di credito/debito
2. Verrà fatto un pre-auth di €0.80-1.00 (rimborsato subito)
3. Completa verifica telefono (SMS)

#### 3. Crea Compute Instance

1. Login → Console
2. Menu (☰) → Compute → Instances
3. Click "Create Instance"

**Configurazione:**
```
Name: betting-monitor
Placement: (lascia default)

Image and Shape:
  → Change Image
    - Ubuntu 22.04 Minimal (consigliato)
    - Ubuntu 22.04 (alternativa)

  → Change Shape
    - Specialty and previous generation
    - VM.Standard.E2.1.Micro (Always Free-eligible) ✅
    - 1 OCPU, 1GB RAM

Networking:
  → Assign public IP: Yes ✅

Add SSH Keys:
  → Generate SSH key pair (SALVA IL .pem FILE!)
  → Oppure: Paste public key (se hai già una chiave)

Boot Volume:
  → 50GB (Always Free include fino a 200GB totali)
```

4. Click "Create"
5. Attendi 1-2 minuti
6. Copia **Public IP Address**

#### 4. Configura Firewall

Oracle ha firewall doppio (Security List + iptables), devi aprire porte:

**Security List (Cloud Console):**
1. Instances → tua-instance → Primary VNIC → Subnet
2. Security Lists → Default Security List
3. Add Ingress Rules:
   ```
   Source CIDR: 0.0.0.0/0
   IP Protocol: TCP
   Destination Port Range: 443
   Description: HTTPS for Telegram API
   ```

**iptables (su VPS):**
```bash
# Connettiti a VPS
ssh -i your-key.pem ubuntu@YOUR_VPS_IP

# Aggiungi regole
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# Oppure disabilita completamente (meno sicuro)
sudo iptables -F
sudo netfilter-persistent save
```

#### 5. Connetti e Deploy

```bash
# Linux/Mac
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_VPS_IP

# Windows (PowerShell)
ssh -i your-key.pem ubuntu@YOUR_VPS_IP
```

**Poi:**
```bash
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds
chmod +x deploy/setup.sh
./deploy/setup.sh systemd
```

### 💡 Tips Oracle

1. **Account dormant:** Se non usi per 30+ giorni, Oracle può mettere in pausa. Soluzione: accedi ogni 2-3 settimane.
2. **Multiple VMs:** Puoi avere 2 VM gratis (2x 1GB RAM o 1x ARM con 24GB RAM).
3. **Backup:** Always Free include 200GB block volume totali per backup.
4. **Monitoring:** Oracle Console ha monitoring CPU/RAM/Network gratis.

---

## 🥈 Runner-Up: Google Cloud Platform

### ✅ Vantaggi

- €300 credito per 90 giorni (poi free tier continua)
- e2-micro gratis **per sempre** (US regions)
- Ottima documentazione
- Interfaccia pulita

### 📝 Setup

1. https://cloud.google.com/free
2. Sign up con Gmail
3. Aggiungi payment (serve carta, €300 credito gratis)
4. Compute Engine → Create Instance

**Configurazione:**
```
Name: betting-monitor
Region: us-central1 (Iowa) - FREE TIER
Zone: us-central1-a

Machine type:
  → e2-micro (0.25-2 vCPU, 1GB RAM) - FREE TIER ✅

Boot disk:
  → Ubuntu 22.04 LTS
  → 30GB Standard persistent disk (FREE TIER)

Firewall:
  ✅ Allow HTTPS traffic
```

5. Create → Copia External IP
6. SSH direttamente da browser (click "SSH" button)
7. Deploy:
   ```bash
   git clone https://github.com/Sib-asian/Software-AsianOdds.git
   cd Software-AsianOdds
   ./deploy/setup.sh systemd
   ```

### 💡 Tips Google Cloud

- **Free tier after credits:** e2-micro in us-west1, us-central1, us-east1 è gratis FOREVER
- **Firewall:** Più semplice di Oracle, rules già configurati
- **Monitoring:** Stackdriver monitoring incluso

---

## 🥉 Alternative: AWS Free Tier

### ✅ Vantaggi

- t2.micro gratis per 12 mesi
- 30GB EBS storage
- 15GB bandwidth/mese
- Enorme ecosystem

### 📝 Setup

1. https://aws.amazon.com/free/
2. Create AWS Account
3. EC2 → Launch Instance

**Configurazione:**
```
AMI: Ubuntu Server 22.04 LTS (Free tier eligible)
Instance type: t2.micro (Free tier eligible) ✅
Storage: 30GB gp2 (Free tier eligible)

Security Group:
  → Add rule: HTTPS (443) from 0.0.0.0/0
  → Add rule: SSH (22) from My IP
```

4. Launch → Create key pair → Download .pem
5. Connetti:
   ```bash
   chmod 400 your-key.pem
   ssh -i your-key.pem ubuntu@ec2-XX-XXX-XXX-XX.compute-1.amazonaws.com
   ```
6. Deploy come sopra

### ⚠️ Warning AWS

- **Dopo 12 mesi:** ~€8/mese (valuta switch a Oracle)
- **Bandwidth:** Solo 15GB/mese (sufficiente per questo uso)
- **Billing alerts:** Configura per evitare sorprese

---

## 💰 Low-Cost Options (€4-6/mese)

### Hetzner (Raccomandato)

**Vantaggi:**
- €4.15/mese per 2GB RAM (doppio di Oracle!)
- 20TB bandwidth
- DC in Germania (veloce da Italia)
- Support eccellente
- No setup fees

**Setup:**
1. https://www.hetzner.com/cloud
2. Sign up
3. Create Project → Add Server
4. Location: Nuremberg (Germany)
5. Image: Ubuntu 22.04
6. Type: CX11 (2GB RAM, €4.15/mese)
7. SSH Key: Add yours
8. Create & Deploy

**Pagamento:** PayPal, Carta, SEPA

---

### DigitalOcean

**Vantaggi:**
- Interfaccia semplicissima
- 1-click apps (Docker, ecc.)
- €200 credito con GitHub Student Pack

**Pricing:**
- Basic Droplet: €6/mese (1GB RAM)
- €4/mese (512MB RAM, sufficiente)

**Setup:**
1. https://www.digitalocean.com/
2. Sign up (€200 credito se studente)
3. Create → Droplets
4. Ubuntu 22.04
5. Basic → €6/mese
6. Deploy

---

### Contabo

**Vantaggi:**
- €5/mese per 4GB RAM! (best value)
- 32TB bandwidth
- 50GB SSD

**Svantaggi:**
- Setup fee: €3 (one-time)
- Support meno responsive
- DC solo in Europa/USA

**Setup:**
1. https://contabo.com/en/vps/
2. VPS S: €5/mese
3. Image: Ubuntu 22.04
4. Deploy

---

## 🎓 Student Packs (Extra Credits)

### GitHub Student Developer Pack

**Include:**
- DigitalOcean: €200 credito
- Azure: €100 credito
- Heroku: 2 anni free
- Molti altri servizi

**Requisiti:**
- Email universitaria (.edu o @università.it)
- Student ID

**Apply:** https://education.github.com/pack

---

## 📊 Quale Scegliere?

### Per Te Consiglio:

**Scenario 1: Zero Costi**
```
✅ Oracle Cloud Always Free
   → Forever gratis
   → 1GB RAM (sufficiente)
   → Setup 15 minuti
```

**Scenario 2: Max Performance, Low Cost**
```
✅ Hetzner CX11 (€4/mese)
   → 2GB RAM (doppio!)
   → Veloce da Italia
   → Support ottimo
```

**Scenario 3: Studente**
```
✅ DigitalOcean con GitHub Pack
   → €200 credito = 33 mesi gratis!
   → Interfaccia facile
   → Poi valuta Oracle
```

**Scenario 4: Testing (poi Oracle)**
```
✅ Google Cloud (€300 credito)
   → Testa 3 mesi
   → Migra a Oracle se OK
```

---

## 🔧 Requisiti Minimi Sistema

| Componente | Minimo | Raccomandato | Note |
|------------|--------|--------------|------|
| **RAM** | 512MB | 1GB+ | Con 512MB usa swap |
| **CPU** | 1 core | 1+ core | 1 core OK per <100 match/giorno |
| **Storage** | 10GB | 20GB+ | Cache + logs crescono |
| **Bandwidth** | 1GB/mese | 10GB+/mese | ~50MB/giorno stimato |

**Il nostro sistema:**
- RAM usage: ~200-400MB
- CPU: <10% medio
- Disk: ~500MB (cresce con cache)
- Network: ~1-2GB/mese (API calls + Telegram)

**✅ Tutti i VPS sopra sono più che sufficienti!**

---

## 🛠️ Post-Setup Checklist

Dopo aver creato VPS, verifica:

- [ ] SSH funziona
- [ ] Firewall aperto su porta 443 (HTTPS)
- [ ] DNS resolve funziona (`ping google.com`)
- [ ] Spazio disco sufficiente (`df -h`)
- [ ] RAM disponibile (`free -h`)
- [ ] Auto-updates configurati

```bash
# Check veloce
curl -I https://api.telegram.org  # Deve rispondere 200
ping -c 3 google.com               # Deve rispondere
free -h                            # Almeno 100MB liberi
df -h                              # Almeno 2GB liberi
```

---

## 💡 Pro Tips

1. **Naming:** Usa nomi descrittivi (betting-monitor-prod, betting-test, ecc.)
2. **Backup:** Fai snapshot VPS prima di update major
3. **Monitoring:** Configura UptimeRobot per alert se VPS down
4. **Multiple Regions:** Se fai trading, considera VPS vicino al bookmaker (latency)
5. **Scaling:** Inizia con Oracle Free, scala a Hetzner se serve più potenza

---

## 🆘 Troubleshooting Provider-Specific

### Oracle Cloud

**Problema:** Firewall blocca Telegram
```bash
# Security List + iptables entrambi
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

**Problema:** Instance "dormant"
```
Soluzione: Login console ogni 2-3 settimane
```

### Google Cloud

**Problema:** Costi dopo trial
```
Soluzione: Switch a free tier region (us-central1)
Oppure: Migra a Oracle
```

### AWS

**Problema:** Bandwidth limit (15GB)
```
Soluzione: Monitor in CloudWatch
Se superi: considera Oracle (10TB/mese)
```

---

## 📞 Support Links

- **Oracle Cloud:** https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm
- **Google Cloud:** https://cloud.google.com/free/docs/free-cloud-features
- **AWS:** https://aws.amazon.com/free/
- **Hetzner:** https://docs.hetzner.com/
- **DigitalOcean:** https://docs.digitalocean.com/

---

## ✅ Final Recommendation

**Per iniziare OGGI:**
```
1. Oracle Cloud Always Free ← START HERE
   (15 min setup, €0 forever)

2. Se problemi con Oracle:
   → Google Cloud (€300 credito)

3. Se vuoi più potenza:
   → Hetzner CX11 (€4/mese, 2GB RAM)
```

**Happy deploying! 🚀**
