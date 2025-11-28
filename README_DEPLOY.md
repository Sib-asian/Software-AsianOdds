# 🚀 Guida Rapida Deploy su Render.com

## ✅ Checklist Pre-Deploy

Prima di iniziare, verifica:

- [ ] Tutti i file necessari presenti (✅ Verificato automaticamente)
- [ ] Git installato o GitHub Desktop installato
- [ ] Account GitHub creato
- [ ] Codice pushato su GitHub
- [ ] Account Render.com creato

---

## 🎯 Setup Automatico (CONSIGLIATO)

### Esegui lo script automatico:

```powershell
.\SETUP_COMPLETO_AUTOMATICO.ps1
```

**Lo script farà automaticamente**:
1. ✅ Verifica tutti i file necessari
2. ✅ Verifica se Git è installato
3. ✅ Inizializza repository Git
4. ✅ Prepara commit
5. ✅ Ti dice esattamente cosa fare dopo

---

## 📋 Se Git NON è Installato

### Opzione 1: Installa Git
1. Vai su: https://git-scm.com/download/win
2. Installa Git
3. Riavvia PowerShell
4. Esegui: `.\SETUP_COMPLETO_AUTOMATICO.ps1`

### Opzione 2: GitHub Desktop (PIÙ FACILE)
1. Vai su: https://desktop.github.com/
2. Installa GitHub Desktop
3. File → Add Local Repository
4. Seleziona questa cartella
5. Publish repository

### Opzione 3: Upload Manuale
1. Vai su: https://github.com/new
2. Crea repository
3. Upload file manualmente

**Vedi**: `INSTALLA_GIT_E_SETUP.md` per dettagli completi

---

## 🚀 Deploy su Render.com

Una volta che il codice è su GitHub:

1. **Vai su**: https://render.com
2. **Crea account** (login con GitHub)
3. **New +** → **Background Worker**
4. **Connetti repository** GitHub
5. **Configura**:
   - Name: `automation-24h`
   - Environment: `Docker`
   - Dockerfile Path: `./Dockerfile.automation`
6. **Aggiungi variabili ambiente** (vedi sotto)
7. **Deploy!** ✅

**Vedi**: `GUIDA_RENDER_PASSO_PASSO.md` per guida completa passo-passo

---

## 🔐 Variabili Ambiente Render.com

Aggiungi queste variabili su Render:

```
TELEGRAM_BOT_TOKEN=il_tuo_token_qui
TELEGRAM_CHAT_ID=il_tuo_chat_id_qui
AUTOMATION_MIN_EV=8.0
AUTOMATION_MIN_CONFIDENCE=70.0
AUTOMATION_UPDATE_INTERVAL=300
PYTHONUNBUFFERED=1
```

**Come ottenerle**:
- **Token**: @BotFather su Telegram → `/newbot` → Copia token
- **Chat ID**: @userinfobot su Telegram → `/start` → Copia ID

---

## 📚 Guide Disponibili

- **`GUIDA_RENDER_PASSO_PASSO.md`** - Guida completa passo-passo per Render
- **`INSTALLA_GIT_E_SETUP.md`** - Come installare Git e fare setup
- **`REPORT_VERIFICA_STEP1.md`** - Report verifica file necessari
- **`GUIDA_DEPLOY_GRATUITO_9GB.md`** - Confronto opzioni gratuite

---

## 🆘 Problemi?

### "Git non installato"
→ Vedi: `INSTALLA_GIT_E_SETUP.md`

### "Build failed su Render"
→ Verifica che `Dockerfile.automation` esista
→ Controlla log Render per errori specifici

### "No Telegram notifications"
→ Verifica token e chat ID
→ Controlla log Render

---

## ✅ Verifica Finale

Dopo il deploy, verifica:

1. ✅ Log Render mostrano "Sistema avviato"
2. ✅ Notifiche Telegram ricevute
3. ✅ Sistema funziona correttamente

**🎉 Fatto! Il sistema gira 24/7 su Render.com!**






