# 📢 Come Ottenere il Chat ID di un Canale Telegram

Per i **canali Telegram**, il Chat ID è diverso da quello di una chat privata. Ecco come trovarlo:

---

## 🎯 Metodo 1: Script Automatico (PIÙ SEMPLICE)

### Step 1: Aggiungi il Bot al Canale

1. **Apri Telegram**
2. **Vai al tuo canale**
3. **Clicca sul nome del canale** (in alto)
4. **Clicca su "Amministratori"** o "Administrators"
5. **Clicca su "Aggiungi amministratore"** o "Add Administrator"
6. **Cerca il tuo bot** (username del bot)
7. **Aggiungilo come amministratore**
8. **Assicurati che abbia il permesso** "Post Messages" o "Pubblica messaggi"

### Step 2: Invia un Messaggio

1. **Invia un messaggio qualsiasi** nel canale (puoi anche essere tu a inviarlo)

### Step 3: Esegui lo Script

```bash
python trova_chat_id_canale.py
```

Lo script:
- Ti chiederà di inviare un messaggio nel canale
- Recupererà automaticamente il Chat ID
- Ti mostrerà il Chat ID corretto
- Può aggiornare automaticamente config.json

---

## 🎯 Metodo 2: Manuale (Alternativo)

### Step 1: Aggiungi il Bot al Canale

Come nel Metodo 1, aggiungi il bot come amministratore del canale.

### Step 2: Invia un Messaggio

Invia un messaggio qualsiasi nel canale.

### Step 3: Recupera il Chat ID

Apri questo link nel browser (sostituisci `YOUR_BOT_TOKEN` con il tuo token):

```
https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
```

### Step 4: Cerca il Chat ID

Nel risultato JSON, cerca un campo tipo:

```json
{
  "update_id": 123456789,
  "channel_post": {
    "chat": {
      "id": -1001234567890,  ← QUESTO È IL CHAT ID DEL CANALE
      "title": "Nome del Canale",
      "type": "channel"
    },
    ...
  }
}
```

**Il Chat ID del canale inizia sempre con `-100`** seguito da altri numeri.

---

## 🎯 Metodo 3: Usando @getidsbot

1. **Aggiungi @getidsbot al canale** come amministratore
2. **Invia un messaggio** nel canale
3. **@getidsbot ti risponderà** con il Chat ID del canale

---

## ⚠️ IMPORTANTE

### Chat ID Canale vs Chat Privata

- **Chat privata**: Chat ID positivo (es: `311951419`)
- **Canale**: Chat ID negativo che inizia con `-100` (es: `-1001234567890`)

### Permessi Bot

Il bot **DEVE** essere amministratore del canale con almeno il permesso "Post Messages" per poter inviare messaggi.

---

## ✅ Verifica

Dopo aver ottenuto il Chat ID corretto:

1. **Aggiorna config.json** con il Chat ID del canale
2. **Testa la notifica**:
   ```bash
   python test_telegram_notification.py
   ```

Se ricevi la notifica nel canale → ✅ **Configurazione corretta!**

---

## 🐛 TROUBLESHOOTING

### "Chat not found"

**Causa**: Bot non aggiunto al canale o Chat ID errato

**Soluzione**:
1. Verifica che il bot sia amministratore del canale
2. Verifica che il bot abbia il permesso "Post Messages"
3. Invia un messaggio nel canale
4. Usa lo script `trova_chat_id_canale.py` per trovare il Chat ID corretto

### "Bot is not a member of the channel"

**Causa**: Bot non aggiunto al canale

**Soluzione**:
1. Aggiungi il bot al canale come amministratore
2. Assicurati che abbia il permesso "Post Messages"

### "Not enough rights to send text messages"

**Causa**: Bot non ha il permesso di inviare messaggi

**Soluzione**:
1. Vai alle impostazioni del canale
2. Vai su "Amministratori"
3. Seleziona il bot
4. Assicurati che abbia il permesso "Post Messages" attivo

---

## 💡 NOTE

- Il Chat ID del canale è **sempre negativo** e inizia con `-100`
- Il bot deve essere **amministratore** del canale
- Puoi avere **più canali** con Chat ID diversi
- Ogni canale ha un **Chat ID unico**

---

**Hai trovato il Chat ID del canale?** Aggiorna config.json e testa la notifica!

