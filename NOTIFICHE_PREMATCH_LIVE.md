# 📱 Notifiche Telegram - Pre-Match vs Live

## 🎯 Risposta Rapida

**Ricevi notifiche per ENTRAMBE:**
- ✅ **PRE-MATCH**: Partite che iniziano nelle prossime 24h
- ✅ **LIVE**: Partite già iniziate (nelle ultime 2h)

---

## 📊 Come Funziona

### PRE-MATCH (Partite Future)

**Quando:** Partite che iniziano nelle prossime 24h

**Esempio:**
- Ora: 14:00
- Partita: Inter vs Juventus alle 20:45
- ✅ Analizzata e notificata se trova opportunità

**Vantaggi:**
- Più tempo per piazzare scommessa
- Quote più stabili
- Meno pressione

---

### LIVE (Partite in Corso)

**Quando:** Partite già iniziate (nelle ultime 2h)

**Esempio:**
- Ora: 20:50
- Partita: Inter vs Juventus iniziata alle 20:45
- ✅ Analizzata e notificata se trova opportunità

**Vantaggi:**
- Quote possono cambiare rapidamente
- Opportunità value bet durante la partita
- Analisi basata su situazione reale

**Limitazioni:**
- Solo partite iniziate nelle ultime 2h (per evitare partite finite)
- Quote possono cambiare velocemente
- Meno tempo per piazzare scommessa

---

## ⚙️ Configurazione

### Filtrare Solo Pre-Match

Se vuoi SOLO pre-match, modifica `automation_24h.py`:

```python
# Cambia questa riga:
min_past = now - timedelta(hours=2)  # Partite live

# In:
min_past = now  # Nessuna partita live
```

### Filtrare Solo Live

Se vuoi SOLO live, modifica `automation_24h.py`:

```python
# Cambia questa riga:
max_future = now + timedelta(hours=24)  # Pre-match

# In:
max_future = now  # Nessuna pre-match
```

---

## 📋 Strategia Attuale

**Il sistema analizza:**
1. ✅ **Pre-Match**: Partite nelle prossime 24h
2. ✅ **Live**: Partite iniziate nelle ultime 2h

**Perché entrambe?**
- Pre-match: più tempo, quote stabili
- Live: opportunità durante partita, quote dinamiche

---

## 🎯 Cosa Ricevi

**Notifica Telegram include:**
- Tipo partita (PRE-MATCH o LIVE)
- Partita e lega
- Market consigliato
- Stake suggerito
- Odds
- Expected Value
- Confidence

**Esempio notifica:**
```
⚽ AUTO-24H BETTING OPPORTUNITY ⚽

📅 Match
Inter vs Juventus
🏆 Serie A
🕐 PRE-MATCH (20:45)

💰 Recommendation
Market: 1X2_HOME
Stake: €133.68
Odds: 1.90
...
```

---

## ⚠️ Importante

**Il sistema NON consiglia basandosi su score:**
- ✅ Analizza probabilità vs quote
- ✅ Cerca vero valore
- ❌ NON fa "1-0 quindi gioca 1"

**Per partite LIVE:**
- Analizza opportunità value bet reali
- Non consiglia solo perché una squadra sta vincendo
- Verifica sempre probabilità vs quote

---

## 📊 Statistiche

Nel dashboard puoi vedere:
- Quante opportunità pre-match
- Quante opportunità live
- Performance per tipo
- ROI per pre-match vs live

---

## ✅ Riepilogo

**Ricevi notifiche per:**
- ✅ Partite PRE-MATCH (prossime 24h)
- ✅ Partite LIVE (iniziate ultime 2h)

**Entrambe le tipologie sono analizzate e notificate se trovano vere opportunità VALUE BET!** 🎯

