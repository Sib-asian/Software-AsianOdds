# 📊 Da Dove Vengono le Partite?

## 🎯 Risposta Rapida

Il sistema ottiene le partite da **TheOddsAPI**, un servizio che fornisce:
- ✅ Partite di calcio reali
- ✅ Quote aggiornate da vari bookmaker
- ✅ Partite nelle prossime 24h
- ✅ Partite live (in corso)

---

## 🔍 Come Funziona

### 1. **TheOddsAPI** (Fonte Principale)

**Endpoint:** `https://api.the-odds-api.com/v4/sports/soccer/odds`

**Cosa fornisce:**
- Partite di calcio (soccer) da tutto il mondo
- Quote da bookmaker europei (regione EU)
- Market 1X2 (Head-to-Head)
- Quote in formato decimale (es: 2.10, 3.40)

**Filtri applicati:**
- ✅ Solo partite nelle prossime 24h
- ✅ Solo partite con quote valide (1, X, 2)
- ✅ Quote migliori da tutti i bookmaker

**Configurazione:**
```env
THEODDS_API_KEY=your_api_key_here
```

**Quota API:**
- Free tier: 500 richieste/mese
- Il sistema usa 1 chiamata ogni 5 minuti
- Totale: ~864 chiamate/mese (entro il limite)

---

### 2. **Fallback: Mock Data**

Se TheOddsAPI non è configurata o non disponibile:
- ✅ Il sistema usa dati mock per testing
- ✅ Permette di testare il sistema senza API
- ✅ Log mostrano chiaramente quando usa mock

---

## 📋 Cosa Viene Analizzato

### Partite Monitorate:

1. **Pre-Match** (Prossime 24h):
   - Partite che iniziano nelle prossime 24 ore
   - Analizzate per trovare VALUE BET
   - Quote aggiornate ogni 5 minuti

2. **Live** (In Corso):
   - Partite attualmente in gioco
   - Analizzate per opportunità live betting
   - Quote aggiornate in tempo reale

### Leghe/Campionati:

Il sistema analizza **tutte le partite di calcio** disponibili su TheOddsAPI:
- Serie A (Italia)
- Premier League (Inghilterra)
- La Liga (Spagna)
- Bundesliga (Germania)
- Ligue 1 (Francia)
- E tutte le altre leghe disponibili

---

## ⚙️ Configurazione

### 1. Ottieni API Key TheOddsAPI:

1. Vai su: https://the-odds-api.com/
2. Registrati (gratis)
3. Ottieni la tua API key
4. Aggiungi al file `.env`:
   ```env
   THEODDS_API_KEY=your_api_key_here
   ```

### 2. Verifica Configurazione:

Il sistema controllerà automaticamente:
- ✅ Se `THEODDS_API_KEY` è configurata
- ✅ Se l'API è disponibile
- ✅ Se ci sono partite disponibili

---

## 🔄 Ciclo di Aggiornamento

### Ogni 5 Minuti:

1. **Fetch Partite:**
   - Chiama TheOddsAPI
   - Ottiene partite nelle prossime 24h
   - Filtra partite con quote valide

2. **Analisi:**
   - Per ogni partita, calcola:
     - Probabilità di vittoria (AI)
     - Expected Value (EV)
     - Confidence level
     - Vero valore (probabilità vs quote)

3. **Notifiche:**
   - Se trova opportunità VALUE BET:
     - EV > 8%
     - Confidence > 70%
     - Vero valore rilevato
   - Invia notifica Telegram

---

## 📊 Esempio Partita

**Input da TheOddsAPI:**
```json
{
  "id": "abc123",
  "sport_key": "soccer",
  "home_team": "Inter",
  "away_team": "Juventus",
  "commence_time": "2025-11-18T20:45:00Z",
  "bookmakers": [
    {
      "key": "bet365",
      "markets": [{
        "key": "h2h",
        "outcomes": [
          {"name": "Inter", "price": 2.10},
          {"name": "Draw", "price": 3.40},
          {"name": "Juventus", "price": 3.20}
        ]
      }]
    }
  ]
}
```

**Output Sistema:**
```python
{
  'id': 'abc123',
  'home': 'Inter',
  'away': 'Juventus',
  'league': 'Soccer',
  'date': datetime(2025, 11, 18, 20, 45),
  'odds_1': 2.10,  # Miglior quota per Inter
  'odds_x': 3.40,  # Miglior quota per Pareggio
  'odds_2': 3.20   # Miglior quota per Juventus
}
```

---

## 🚨 Troubleshooting

### "Nessuna partita trovata"

**Possibili cause:**
1. `THEODDS_API_KEY` non configurata
2. API key non valida
3. Nessuna partita nelle prossime 24h
4. Problemi di connessione

**Soluzione:**
- Verifica `.env` file
- Controlla log per errori API
- Il sistema userà mock data come fallback

### "API quota exhausted"

**Causa:**
- Raggiunto limite chiamate API

**Soluzione:**
- Attendi reset mensile (TheOddsAPI)
- Oppure aggiorna a piano pagamento

### "Using mock data"

**Causa:**
- TheOddsAPI non disponibile o non configurata

**Soluzione:**
- Configura `THEODDS_API_KEY` in `.env`
- Riavvia servizio

---

## 📝 Log Esempi

**Partite Reali Trovate:**
```
✅ Trovate 15 partite reali da TheOddsAPI
🔄 Running analysis cycle...
   Found 15 matches to monitor
```

**Mock Data (Testing):**
```
ℹ️  No real matches found, using mock data for testing
⚠️  API Manager not available, using mock data
```

**Errore API:**
```
⚠️  TheOddsAPI request failed: 401 Unauthorized
ℹ️  THEODDS_API_KEY non configurata, usando mock data
```

---

## ✅ Riepilogo

- **Fonte:** TheOddsAPI (partite reali di calcio)
- **Frequenza:** Ogni 5 minuti
- **Filtri:** Prossime 24h, quote valide
- **Fallback:** Mock data se API non disponibile
- **Configurazione:** `THEODDS_API_KEY` in `.env`

**Il sistema analizza automaticamente tutte le partite disponibili e ti notifica solo le vere opportunità VALUE BET!** 🎯

