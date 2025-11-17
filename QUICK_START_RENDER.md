# 🚀 Quick Start - Render.com (Immagine 9 GB)

## ⚡ Setup in 5 Minuti

### 1. Prepara il Codice
```bash
cd Software-AsianOdds-main
git add .
git commit -m "Ready for Render deployment"
git push
```

### 2. Crea Account Render
1. Vai su: https://render.com
2. Clicca "Get Started for Free"
3. Login con GitHub (più facile)

### 3. Crea Background Worker

1. **Dashboard** → "New +" → **"Background Worker"**

2. **Connetti Repository**:
   - Seleziona "Public Git repository"
   - Inserisci: `tuo-username/Software-AsianOdds` (o il nome del tuo repo)
   - Oppure connetti direttamente GitHub account

3. **Configurazione**:
   ```
   Name: automation-24h
   Region: Frankfurt (o più vicino a te)
   Branch: main (o il tuo branch)
   Root Directory: (lascia vuoto)
   Environment: Docker
   Dockerfile Path: ./Dockerfile.automation
   Docker Context: .
   ```

4. **Plan**: Seleziona **"Free"**
   - ⚠️ Va in sleep dopo 15 min di inattività
   - Per sempre attivo: "Starter" ($7/mese)

5. **Environment Variables** (IMPORTANTE!):
   ```
   TELEGRAM_BOT_TOKEN=il_tuo_token_qui
   TELEGRAM_CHAT_ID=il_tuo_chat_id_qui
   AUTOMATION_MIN_EV=8.0
   AUTOMATION_MIN_CONFIDENCE=70.0
   AUTOMATION_UPDATE_INTERVAL=300
   PYTHONUNBUFFERED=1
   ```

6. **Clicca "Create Background Worker"** ✅

### 4. Attendi Deploy
- Il primo deploy può richiedere 5-10 minuti (build immagine 9 GB)
- Monitora i log in tempo reale
- Se vedi errori, controlla le variabili ambiente

### 5. Verifica Funzionamento
1. Vai su "Logs" nel dashboard
2. Dovresti vedere output tipo:
   ```
   ✅ Telegram configurato
   🚀 Avvio automazione 24/7...
   ```
3. Controlla Telegram per notifiche

---

## 🔧 Se Vuoi Usare Dockerfile Ottimizzato

Se vuoi ridurre la dimensione (per usare Railway dopo):

1. Cambia **Dockerfile Path** in: `./Dockerfile.optimized`
2. Riavvia deploy
3. Dimensione attesa: ~500 MB - 1.5 GB (invece di 9 GB)

---

## 📊 Monitoraggio

### Log in Tempo Reale
- Dashboard → Il tuo worker → "Logs"
- Aggiornamento automatico

### Metrics
- Dashboard → "Metrics"
- CPU, Memory, Network usage

---

## 🆘 Troubleshooting

### "Build failed"
- Verifica che `Dockerfile.automation` esista
- Controlla log per errori specifici
- Verifica che repository sia pubblico o connesso correttamente

### "Service keeps restarting"
- Controlla variabili ambiente (soprattutto TELEGRAM_BOT_TOKEN)
- Verifica log per errori Python
- Assicurati che `start_automation.py` esista

### "No Telegram notifications"
- Verifica token: https://t.me/BotFather
- Verifica chat ID: https://t.me/userinfobot
- Controlla log per errori API Telegram

### "Out of memory"
- Free tier ha 512 MB RAM
- Se insufficiente, considera upgrade a Starter ($7/mese) o ottimizza codice

---

## 💰 Costi

### Free Tier
- ✅ Gratis
- ⚠️ Sleep dopo 15 min di inattività
- 512 MB RAM
- 0.1 CPU

### Starter ($7/mese)
- ✅ Sempre attivo (no sleep)
- ✅ 512 MB RAM
- ✅ 0.5 CPU
- ✅ Support prioritario

---

## ✅ Checklist

- [ ] Codice su GitHub
- [ ] Account Render creato
- [ ] Background Worker creato
- [ ] Variabili ambiente configurate
- [ ] Deploy completato
- [ ] Log verificati
- [ ] Notifiche Telegram ricevute

---

## 🎉 Fatto!

Il tuo sistema ora gira su Render.com! 

**Prossimi passi:**
- Monitora i log per i primi giorni
- Verifica che le notifiche arrivino correttamente
- Se tutto ok, considera upgrade a Starter per sempre attivo

