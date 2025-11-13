# 📱 Notifiche Telegram - Setup Veloce

## 🎯 Cosa Fa

Ricevi notifica Telegram quando analizzi una partita manualmente e il sistema trova un'opportunità.

**Funziona solo per analisi manuali** (quando TU usi l'app), non è un sistema automatico 24/7.

---

## ⚡ Setup (2 minuti)

### 1. Crea Bot Telegram

```
Telegram → Cerca @BotFather → /newbot
Salva il TOKEN
```

### 2. Ottieni Chat ID

```
Telegram → Cerca @userinfobot → /start
Salva il tuo ID
```

### 3. Configura

Apri `ai_system/config.py`:

```python
telegram_enabled: bool = True
telegram_bot_token: str = "TUO_TOKEN_QUI"
telegram_chat_id: str = "TUO_ID_QUI"
```

Salva.

---

## ✅ Fatto!

Quando analizzi partite e trovi opportunità → 📱 Notifica Telegram

Se non configuri → App funziona ugualmente, solo senza notifiche.

---

**Guida completa:** `TELEGRAM_SETUP.md`
