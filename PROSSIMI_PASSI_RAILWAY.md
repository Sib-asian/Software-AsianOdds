# 🚀 Prossimi Passi - Railway Deploy

## ✅ Stato Attuale

Vedo che:
- ✅ Progetto Railway creato: "software-asianodds"
- ✅ Deploy in corso: "BUILDING" (sta costruendo l'immagine)
- ⏳ Aspetta che finisca il build

---

## 📋 Cosa Fare ORA

### PASSO 1: Aspetta che Finisca il Build (2-5 minuti)

Il sistema sta costruendo l'immagine Docker. Aspetta che vedi:
- ✅ "Initialization" - completato
- ⏳ "Build" - in corso (attualmente attivo)
- ⏸️ "Deploy" - non ancora iniziato
- ⏸️ "Post-deploy" - non ancora iniziato

**Aspetta finché vedi "Deploy" completato!**

---

### PASSO 2: Configura Variabili Ambiente

Una volta che il deploy è completato:

1. **Clicca sulla tab "Variables"** (in alto, accanto a "Deployments")

2. **Clicca "New Variable"** (pulsante in alto a destra)

3. **Aggiungi queste variabili UNA PER UNA:**

#### Variabile 1: Telegram Bot Token
- **Key:** `TELEGRAM_BOT_TOKEN`
- **Value:** `[incolla qui il token del bot che hai ottenuto da @BotFather]`
- Clicca **"Add"**

#### Variabile 2: Telegram Chat ID
- **Key:** `TELEGRAM_CHAT_ID`
- **Value:** `[incolla qui il tuo chat ID da @userinfobot]`
- Clicca **"Add"**

#### Variabile 3: Min EV (opzionale)
- **Key:** `AUTOMATION_MIN_EV`
- **Value:** `8.0`
- Clicca **"Add"**

#### Variabile 4: Min Confidence (opzionale)
- **Key:** `AUTOMATION_MIN_CONFIDENCE`
- **Value:** `70.0`
- Clicca **"Add"**

#### Variabile 5: Update Interval (opzionale)
- **Key:** `AUTOMATION_UPDATE_INTERVAL`
- **Value:** `300`
- Clicca **"Add"**

---

### PASSO 3: Riavvia il Servizio

Dopo aver aggiunto le variabili:

1. **Vai su "Settings"** (tab in alto)
2. **Scorri in basso**
3. **Clicca "Restart"** (pulsante rosso)
4. **Conferma** il riavvio

**Aspetta 30-60 secondi** che riavvii.

---

### PASSO 4: Verifica Log

1. **Vai su "Logs"** (tab in alto)
2. **Dovresti vedere output tipo:**

```
✅ AI Pipeline initialized
✅ Telegram Notifier initialized
✅ API Manager initialized
🚀 Starting Automation24H system...
   Min EV: 8.0%
   Min Confidence: 70.0%
   Update Interval: 300s
🔄 Running analysis cycle...
```

**Se vedi questi messaggi = FUNZIONA! ✅**

---

### PASSO 5: Testa Notifiche Telegram

Il sistema invierà notifiche automaticamente quando trova opportunità.

**Per verificare che Telegram funzioni:**
1. Apri Telegram
2. Cerca il tuo bot
3. Invia `/start`
4. Dovresti ricevere una risposta

**Nota:** Le notifiche di betting arriveranno solo quando il sistema trova vere opportunità (EV > 8%, Confidence > 70%).

---

## ❓ Problemi Comuni

### "Build failed" o "Deploy failed"

**Cosa fare:**
1. Clicca "View logs" sul deploy fallito
2. Leggi l'errore
3. Controlla che:
   - `Dockerfile.automation` esista nel repository
   - `requirements.txt` esista
   - Non ci siano errori di sintassi

**Soluzione comune:**
- Se manca `Dockerfile.automation`, assicurati che sia stato pushato su GitHub
- Riavvia il deploy

### "Service keeps restarting"

**Cosa fare:**
1. Vai su "Logs"
2. Leggi l'errore (di solito è in rosso)
3. Verifica che:
   - `TELEGRAM_BOT_TOKEN` sia configurato
   - `TELEGRAM_CHAT_ID` sia configurato
   - I valori siano corretti (senza spazi)

### "No logs" o "Service not running"

**Cosa fare:**
1. Vai su "Settings"
2. Clicca "Restart"
3. Aspetta 1 minuto
4. Controlla di nuovo i log

---

## ✅ Checklist

- [ ] Build completato (tutti i step verdi)
- [ ] Variabili ambiente configurate (almeno TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID)
- [ ] Servizio riavviato
- [ ] Log verificati (vedi messaggi di inizializzazione)
- [ ] Bot Telegram risponde a `/start`
- [ ] Sistema attivo e funzionante

---

## 🎉 Quando Tutto è Fatto

Il sistema ora:
- ✅ Gira 24/7 su Railway (gratis)
- ✅ Analizza partite ogni 5 minuti
- ✅ Ti invia notifiche Telegram quando trova opportunità
- ✅ Non consiglia basandosi su score (solo value bet reali)

**🎊 Fatto! Il sistema è operativo!**

---

## 📊 Monitoraggio

### Verifica che Giri

1. **Vai su "Metrics"** (tab in alto)
2. Dovresti vedere:
   - CPU usage (dovrebbe essere > 0%)
   - Memory usage
   - Network traffic

### Verifica Log Periodicamente

1. **Vai su "Logs"**
2. Ogni 5 minuti dovresti vedere:
   ```
   🔄 Running analysis cycle...
   Found X matches to monitor
   ✅ Cycle complete: X opportunities found
   ```

---

**Buona fortuna! 🚀**






