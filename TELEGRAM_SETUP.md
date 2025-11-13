# 📱 Telegram Notifications Setup (Opzionale)

Guida rapida per ricevere notifiche Telegram quando analizzi partite manualmente.

---

## 🎯 Cosa Fa

Quando analizzi una partita con Frontendcloud.py e il sistema trova un'opportunità di betting:
- ✅ Ricevi notifica su Telegram
- ✅ Con tutti i dettagli (odds, stake, EV%, confidence)
- ✅ Funziona anche da mobile

**NON è un sistema automatico 24/7**, ricevi notifiche solo quando:
- Apri Frontendcloud.py
- Inserisci manualmente una partita
- Clicchi "Analizza Match"
- Sistema trova opportunità → Telegram notification

---

## ⚡ Setup Veloce (3 minuti)

### Step 1: Crea Telegram Bot

```
1. Apri Telegram
2. Cerca: @BotFather
3. Invia: /newbot
4. Segui istruzioni
5. SALVA il token (es: 123456789:AAH...)
```

### Step 2: Ottieni Chat ID

```
1. Cerca: @userinfobot
2. Invia: /start
3. SALVA il tuo ID (es: 1234567890)
```

### Step 3: Configura in ai_system/config.py

Apri il file `ai_system/config.py` e modifica:

```python
# Telegram notifications (OPTIONAL)
telegram_enabled: bool = True  # ← Cambia da False a True

telegram_bot_token: str = "123456789:AAH..."  # ← Incolla tuo token
telegram_chat_id: str = "1234567890"  # ← Incolla tuo chat ID
```

**Salva il file.**

---

## ✅ Test

1. Avvia Frontendcloud.py:
   ```bash
   streamlit run Frontendcloud.py
   ```

2. Inserisci una partita qualsiasi

3. Clicca "Analizza Match"

4. Se trova opportunità → Ricevi messaggio Telegram! 📱

---

## 📱 Esempio Notifica

```
⚽ BETTING OPPORTUNITY ⚽

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

🤖 AI Ensemble
Dixon-Coles: 65.0%
XGBoost: 71.0%
LSTM: 68.0%

⏰ 14:32:15
```

---

## ⚙️ Configurazione Avanzata

In `ai_system/config.py` puoi modificare:

```python
# Soglie notifiche
telegram_min_ev: float = 5.0           # Notifica solo se EV > 5%
telegram_min_confidence: float = 60.0  # Solo se confidence > 60%

# Rate limiting
telegram_rate_limit_seconds: int = 3  # Secondi tra messaggi
```

---

## 🔕 Disabilitare Telegram

Se non vuoi più notifiche, in `ai_system/config.py`:

```python
telegram_enabled: bool = False  # ← Cambia a False
```

Il sistema continuerà a funzionare normalmente, semplicemente non invierà notifiche.

---

## 🐛 Troubleshooting

### Non ricevo notifiche

**1. Verifica configurazione:**
```bash
cat ai_system/config.py | grep telegram_enabled
# Deve essere: telegram_enabled: bool = True
```

**2. Test bot manualmente:**
```bash
curl -X POST https://api.telegram.org/bot<TUO_TOKEN>/sendMessage \
  -d "chat_id=<TUO_CHAT_ID>&text=Test"
```

Se ricevi "Test" su Telegram → bot funziona.

**3. Verifica bot sia avviato:**
- Cerca il tuo bot su Telegram
- Invia `/start` al bot

**4. Check logs:**
Quando analizzi partita, guarda logs in Frontendcloud.py per errori Telegram.

---

## 💡 Note

- **Gratis:** Telegram bot è completamente gratuito
- **Privacy:** Solo tu ricevi notifiche (tuo chat ID)
- **Opzionale:** Sistema funziona anche senza Telegram
- **Manuale:** Notifiche solo quando TU analizzi partite

---

Telegram configurato! Quando trovi opportunità, ricevi notifica istantanea! 📱
