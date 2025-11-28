# 🔍 Come Recuperare i Valori Telegram Esistenti

Se hai già un bot Telegram e un chat ID, ecco come recuperarli:

---

## 📋 STEP 1: Recupera il Bot Token

### Metodo 1: Da BotFather (PIÙ SEMPLICE)

1. **Apri Telegram**
2. **Cerca**: `@BotFather`
3. **Invia il comando**: `/mybots`
4. **BotFather ti mostrerà** la lista dei tuoi bot
5. **Clicca sul bot** che vuoi usare
6. **Clicca su "API Token"** o "Edit Bot" → "API Token"
7. **BotFather ti mostrerà il token** (tipo: `123456789:AAHdqwe...`)
8. **COPIA il token** e salvalo

### Metodo 2: Se Ricordi il Nome del Bot

1. **Apri Telegram**
2. **Cerca**: `@BotFather`
3. **Invia**: `/token`
4. **BotFather ti chiederà** quale bot vuoi modificare
5. **Invia il nome del tuo bot**
6. **BotFather ti mostrerà il token**

### Metodo 3: Se Non Ricordi il Nome

1. **Apri Telegram**
2. **Cerca**: `@BotFather`
3. **Invia**: `/mybots`
4. **Vedrai la lista** di tutti i tuoi bot
5. **Scegli quello che vuoi usare**
6. **Segui i passi del Metodo 1**

---

## 📋 STEP 2: Recupera il Chat ID

### Metodo 1: Usando @userinfobot (PIÙ SEMPLICE)

1. **Apri Telegram**
2. **Cerca**: `@userinfobot`
3. **Invia**: `/start`
4. **Il bot ti risponderà** con le tue informazioni
5. **Cerca il campo "Id"** (un numero tipo `1234567890`)
6. **COPIA questo numero** - è il tuo Chat ID

### Metodo 2: Usando il Tuo Bot

1. **Invia un messaggio qualsiasi** al tuo bot (quello che hai appena recuperato il token)
2. **Apri questo link nel browser** (sostituisci `YOUR_BOT_TOKEN` con il token che hai recuperato):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
3. **Cerca nel risultato JSON** il campo `"chat":{"id":1234567890}`
4. **Il numero dopo `"id":` è il tuo Chat ID**

**Esempio di risultato:**
```json
{
  "ok": true,
  "result": [
    {
      "update_id": 123456789,
      "message": {
        "chat": {
          "id": 1234567890,  ← QUESTO È IL TUO CHAT ID
          "first_name": "Tuo Nome",
          ...
        },
        ...
      }
    }
  ]
}
```

### Metodo 3: Se Hai Inviato Messaggi al Bot Prima

1. **Apri Telegram**
2. **Vai alla chat** con il tuo bot
3. **Invia qualsiasi messaggio** (es: "test")
4. **Poi usa il Metodo 2** per recuperare il Chat ID

---

## 📋 STEP 3: Inserisci i Valori in config.json

Una volta recuperati entrambi i valori:

1. **Apri** `config.json`
2. **Sostituisci** `YOUR_TELEGRAM_BOT_TOKEN_HERE` con il token che hai recuperato
3. **Sostituisci** `YOUR_TELEGRAM_CHAT_ID_HERE` con il chat ID che hai recuperato
4. **Salva** il file

**Esempio:**
```json
{
  "telegram_token": "123456789:AAHdqweRtyuiopasdfghjklzxcvbnm",
  "telegram_chat_id": "1234567890",
  ...
}
```

---

## ✅ STEP 4: Verifica

### Test Rapido

Apri questo link nel browser (sostituisci i valori):
```
https://api.telegram.org/botYOUR_BOT_TOKEN/sendMessage?chat_id=YOUR_CHAT_ID&text=Test%20notifica
```

Se ricevi "Test notifica" su Telegram → ✅ **Configurazione corretta!**

---

## 🐛 TROUBLESHOOTING

### "Non trovo il mio bot in /mybots"

**Possibili cause:**
- Il bot è stato eliminato
- Stai usando un account Telegram diverso
- Il bot è stato creato da qualcun altro

**Soluzione:** Crea un nuovo bot seguendo la guida `GUIDA_TELEGRAM_SETUP.md`

### "Il token non funziona"

**Possibili cause:**
- Token copiato male (controlla spazi extra)
- Token scaduto o revocato
- Bot eliminato

**Soluzione:** 
1. Vai su @BotFather
2. Usa `/mybots` → Seleziona il bot → "API Token" → "Revoke token"
3. Poi "Generate new token"
4. Copia il nuovo token

### "Non trovo il Chat ID"

**Soluzione:**
1. Assicurati di aver inviato almeno un messaggio al bot
2. Usa @userinfobot (metodo più semplice)
3. Oppure usa il metodo del browser con getUpdates

---

## 💡 NOTE

- **Il token è segreto**: Non condividerlo con nessuno
- **Il Chat ID è personale**: Ogni account Telegram ha un ID unico
- **Puoi avere più bot**: Ogni bot ha il suo token
- **Puoi usare lo stesso bot**: Per più progetti (stesso token, stesso Chat ID)

---

**Hai recuperato i valori?** Inseriscili in `config.json` e riavvia Python!

