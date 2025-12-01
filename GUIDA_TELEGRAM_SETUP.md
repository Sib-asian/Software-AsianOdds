# 📱 Guida Completa: Configurazione Telegram per Notifiche

Questa guida ti spiega passo-passo come ottenere il Bot Token e il Chat ID per abilitare le notifiche Telegram.

---

## 🎯 Cosa Ti Serve

1. **Telegram Bot Token** - Token del bot che invierà le notifiche
2. **Telegram Chat ID** - Il tuo ID personale per ricevere le notifiche

---

## 📋 STEP 1: Crea il Bot Telegram

### 1.1 Apri Telegram
- Apri l'app Telegram sul tuo telefono o computer
- Cerca: **@BotFather**

### 1.2 Avvia BotFather
- Clicca su **@BotFather** (dovrebbe avere un badge blu "VERIFIED")
- Clicca su **START** o invia `/start`

### 1.3 Crea un Nuovo Bot
- Invia il comando: `/newbot`
- BotFather ti chiederà: **"Alright, a new bot. How are we going to call it? Please choose a name for your bot."**
- Invia un nome per il tuo bot (es: "My Betting Bot" o "Live Betting Notifier")

### 1.4 Scegli Username
- BotFather ti chiederà: **"Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot."**
- Invia un username che finisca con `bot` (es: "my_betting_bot" o "livebetting_bot")
- ⚠️ **IMPORTANTE**: L'username deve essere unico e non già utilizzato

### 1.5 Ottieni il Token
- Se tutto va bene, BotFather ti invierà un messaggio tipo:
  ```
  Done! Congratulations on your new bot. You will find it at t.me/my_betting_bot. 
  You can now add a description, about section and profile picture for your bot, see /help for a list of commands.
  
  Use this token to access the HTTP API:
  123456789:AAHdqweRtyuiopasdfghjklzxcvbnm
  
  Keep your token secure and store it safely, it can be used by anyone to control your bot.
  ```
- **COPIA IL TOKEN** (la stringa tipo `123456789:AAHdqwe...`)
- ✅ **SALVA IL TOKEN** - Ti servirà dopo!

---

## 📋 STEP 2: Ottieni il Tuo Chat ID

### 2.1 Metodo 1: Usando @userinfobot (PIÙ SEMPLICE)

1. **Cerca su Telegram**: `@userinfobot`
2. **Avvia il bot**: Clicca su START o invia `/start`
3. **Il bot ti risponderà** con le tue informazioni, incluso il tuo **ID**
4. **COPIA IL TUO ID** (un numero tipo `1234567890`)
5. ✅ **SALVA L'ID**

### 2.2 Metodo 2: Usando @getidsbot (ALTERNATIVA)

1. **Cerca su Telegram**: `@getidsbot`
2. **Avvia il bot**: Clicca su START
3. **Invia qualsiasi messaggio** al bot
4. **Il bot ti risponderà** con il tuo Chat ID
5. **COPIA IL TUO ID**

### 2.3 Metodo 3: Manuale (se i bot non funzionano)

1. **Invia un messaggio al tuo bot** (quello che hai appena creato)
2. **Apri questo link nel browser** (sostituisci `YOUR_BOT_TOKEN` con il token che hai ottenuto):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
3. **Cerca nel risultato JSON** il campo `"chat":{"id":1234567890}`
4. **Il numero dopo `"id":` è il tuo Chat ID**

---

## 📋 STEP 3: Configura nel Sistema

### 3.1 Apri config.json

Apri il file `config.json` nella directory del progetto.

### 3.2 Inserisci i Valori

Sostituisci i valori placeholder con i tuoi valori reali:

```json
{
  "telegram_token": "123456789:AAHdqweRtyuiopasdfghjklzxcvbnm",
  "telegram_chat_id": "1234567890",
  "min_ev": 8.0,
  "min_confidence": 70.0,
  "update_interval": 780,
  "max_notifications_per_cycle": 2
}
```

**Dove:**
- `telegram_token`: Il token che hai ottenuto da BotFather (STEP 1.5)
- `telegram_chat_id`: Il tuo Chat ID (STEP 2)

### 3.3 Salva il File

Salva il file `config.json` dopo aver inserito i valori.

---

## ✅ STEP 4: Test della Configurazione

### 4.1 Riavvia Python

Riavvia il sistema Python per applicare le modifiche.

### 4.2 Verifica nei Log

Controlla i log per vedere se Telegram è stato configurato correttamente:
- ✅ Dovresti vedere: `"✅ Telegram Notifier initialized"`
- ❌ Se vedi: `"⚠️ Telegram not configured"` → Verifica i valori in config.json

### 4.3 Test Manuale (Opzionale)

Puoi testare il bot manualmente inviando un messaggio di test:

**Nel browser, apri questo URL** (sostituisci i valori):
```
https://api.telegram.org/botYOUR_BOT_TOKEN/sendMessage?chat_id=YOUR_CHAT_ID&text=Test%20notifica
```

Se ricevi "Test notifica" su Telegram → ✅ Configurazione corretta!

---

## 🔒 SICUREZZA

⚠️ **IMPORTANTE**: 
- **NON condividere** il Bot Token con nessuno
- **NON committare** config.json su GitHub se contiene valori reali
- **Mantieni** il token segreto e sicuro

---

## 🐛 TROUBLESHOOTING

### Problema: "Telegram not configured"

**Soluzione:**
1. Verifica che i valori in `config.json` siano corretti (senza spazi extra)
2. Verifica che il token inizi con numeri e contenga `:`
3. Verifica che il Chat ID sia solo numeri
4. Riavvia Python dopo aver modificato config.json

### Problema: "Unauthorized" o "Forbidden"

**Soluzione:**
1. Verifica che il Bot Token sia corretto
2. Verifica che il Chat ID sia corretto
3. Assicurati di aver inviato `/start` al bot almeno una volta

### Problema: Non ricevo notifiche

**Soluzione:**
1. Verifica che il bot sia attivo (cerca il bot su Telegram)
2. Invia `/start` al bot
3. Verifica i log per errori
4. Controlla che ci siano opportunità valide (EV > 8%, Confidence > 70%)

---

## 📝 RIEPILOGO RAPIDO

1. ✅ Crea bot con @BotFather → Ottieni **Token**
2. ✅ Ottieni Chat ID con @userinfobot → Ottieni **Chat ID**
3. ✅ Inserisci valori in `config.json`
4. ✅ Riavvia Python
5. ✅ Verifica nei log che Telegram sia configurato

---

## 💡 NOTE

- Il bot è **gratuito** e illimitato
- Le notifiche arrivano **solo a te** (il tuo Chat ID)
- Il sistema funziona anche **senza Telegram** (ma non riceverai notifiche)
- Puoi creare **più bot** per scopi diversi

---

**Hai bisogno di aiuto?** Controlla i log in `logs/automation_24h.log` per vedere eventuali errori.

