# ⚡ Quick Deploy - 5 Minuti

## 🚀 Deploy su Railway.app (CONSIGLIATO - Gratis)

### Passo 1: Push su GitHub (2 min)
```bash
git add .
git commit -m "Add automation 24/7 system"
git push
```

### Passo 2: Deploy su Railway (3 min)

1. **Vai su**: https://railway.app
2. **Clicca**: "Start a New Project"
3. **Seleziona**: "Deploy from GitHub repo"
4. **Autorizza** GitHub se richiesto
5. **Seleziona** repository: `Software-AsianOdds`
6. **Railway rileva automaticamente** il Dockerfile

### Passo 3: Configura Variabili (1 min)

Nella dashboard Railway:
1. **Clicca** sul servizio
2. **Vai** su "Variables"
3. **Aggiungi**:

```
TELEGRAM_BOT_TOKEN = your_bot_token
TELEGRAM_CHAT_ID = your_chat_id
AUTOMATION_MIN_EV = 8.0
AUTOMATION_MIN_CONFIDENCE = 70.0
AUTOMATION_UPDATE_INTERVAL = 300
```

### Passo 4: Deploy! ✅

Railway deploya automaticamente. Vai su "Logs" per vedere l'output.

**🎉 Fatto! Il sistema gira 24/7 gratis!**

---

## 🔄 Alternative: GitHub Actions (Completamente Gratis)

Se Railway non funziona, usa GitHub Actions:

### Passo 1: Configura Secrets

1. Vai su GitHub → Repository → **Settings** → **Secrets** → **Actions**
2. **New repository secret**:
   - Name: `TELEGRAM_BOT_TOKEN` → Value: `your_token`
   - Name: `TELEGRAM_CHAT_ID` → Value: `your_chat_id`

### Passo 2: Push Codice

Il file `.github/workflows/automation.yml` è già incluso. Basta pushare:

```bash
git add .
git commit -m "Add GitHub Actions automation"
git push
```

### Passo 3: Verifica

1. Vai su **Actions** tab
2. Vedi workflow "Automation 24/7 (Cron)"
3. Esegue ogni 5 minuti automaticamente

**✅ Fatto! Esegue ogni 5 minuti gratis!**

---

## 📱 Test Notifiche

Dopo deploy, dovresti ricevere notifiche Telegram quando trova opportunità.

**Verifica log:**
- Railway: Dashboard → Logs
- GitHub Actions: Actions tab → Vedi log

---

## ❓ Problemi?

**"Deploy failed"**
- Verifica che `requirements.txt` esista
- Controlla log per errori

**"No notifications"**
- Verifica token Telegram
- Verifica chat ID
- Controlla log per errori API

**"Service stopped"**
- Railway: Verifica crediti ($5/mese gratis)
- GitHub Actions: Verifica limiti (2000 min/mese)

---

**🎊 Sistema pronto e gratis!**






