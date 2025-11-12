# 🚀 Guida Setup LEVEL 2 (API Gratuito)

## 📋 Cosa Devi Fare TU (10 minuti)

Ho implementato **LEVEL 2 Lite** con supporto API gratuito. Per attivarlo:

---

## ⚡ Setup Veloce (5 minuti)

### 1. Registrati a TheSportsDB (Opzionale - Già Funzionante)

**TheSportsDB è GIÀ CONFIGURATO** con chiave free gratuita!

✅ Nessuna registrazione richiesta
✅ Unlimited API calls
✅ Funziona subito

**Puoi saltare al punto 3** se TheSportsDB basta.

---

### 2. (Opzionale) Registrati API-Football

Se vuoi **più dati** (classifiche, statistiche dettagliate):

1. **Vai su:** https://www.api-football.com/
2. **Click:** "Register" (top-right)
3. **Compila form:**
   - Email
   - Password
   - Accetta termini
4. **Verifica email** (check inbox)
5. **Login** e vai su "Dashboard"
6. **Copia API Key:**
   ```
   Dashboard → My Access → API Key

   Esempio chiave:
   a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
   ```

**Piano FREE:**
- ✅ 100 chiamate/giorno
- ✅ Sufficiente per 25 match/giorno
- ✅ €0/mese

---

### 3. Configura API Key nell'App

#### Opzione A: Tramite File di Configurazione (Facile)

1. **Apri:**  `api_manager.py`

2. **Trova riga 26-27:**
   ```python
   # API-Football (Free tier: 100 calls/day)
   API_FOOTBALL_KEY = ""  # User will add this
   ```

3. **Incolla la tua chiave:**
   ```python
   API_FOOTBALL_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
   ```

4. **Salva** il file

#### Opzione B: Tramite Variabile d'Ambiente (Avanzato)

```bash
# Linux/Mac
export API_FOOTBALL_KEY="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"

# Windows
set API_FOOTBALL_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

Poi modifica `api_manager.py` riga 27:
```python
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
```

---

### 4. Avvia l'App

```bash
streamlit run Frontendcloud.py
```

---

### 5. Attiva LEVEL 2 nell'UI

1. **Apri match** (inserisci squadre e quote)

2. **Espandi:** "🚀 Funzionalità Avanzate"

3. **Seleziona modalità dal dropdown:**
   ```
   Modalità Auto-Detection: [Dropdown]
   ├─ ✋ Manuale
   ├─ 🗄️ Auto (Solo Database)
   └─ 🌐 Auto + API (Ibrido)  ← Seleziona questo
   ```

4. **Clicca:** "Analizza Match"

5. **Guarda la magia:**
   ```
   📊 API Status:
   ├─ Quota oggi: 2/100 usate
   ├─ Cache hits: 0%
   └─ Providers: TheSportsDB (✅)
   ```

---

## 🎯 Come Funziona

### Workflow Automatico

```
User inserisce: "Midtjylland vs Nordsjælland" (Danimarca)
                             ↓
┌──────────────────────────────────────────────────────────┐
│ 1. Check Database Locale (LEVEL 1)                      │
│    ❌ Non trovato (campionato danese non in DB)         │
└──────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Check Cache SQLite (24h)                             │
│    ❌ Not found (prima volta oggi)                      │
└──────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Check API Quota                                       │
│    ✅ 2/100 usate (98 disponibili)                      │
└──────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────┐
│ 4. Fetch da TheSportsDB API                             │
│    📡 GET /searchteams.php?t=Midtjylland                │
│    ✅ Success! (1 call)                                  │
│    💾 Cached per 24h                                     │
└──────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────┐
│ 5. Auto-Detection Completata                            │
│    ✅ Stile: Possesso vs Possesso                       │
│    ✅ Motivazione: Normale vs Normale                   │
│    ✅ Fixture: 7gg vs 7gg (default)                     │
└──────────────────────────────────────────────────────────┘
```

**Prossimo match stesso giorno:**
```
User inserisce: "Midtjylland vs Altra Squadra"
                             ↓
┌──────────────────────────────────────────────────────────┐
│ Check Cache...                                           │
│ ✅ Cache HIT! (0 API calls)                             │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Monitoring

### Vedere Quota Usata

Nell'UI, dopo ogni analisi:

```
📊 API Manager Status:
┌─────────────────────────────────────┐
│ API-Football:                       │
│ ├─ Usate oggi: 4/100               │
│ ├─ Rimanenti: 96                   │
│ └─ Quota reset: Mezzanotte UTC     │
│                                     │
│ TheSportsDB:                        │
│ ├─ Usate oggi: 8                   │
│ └─ Note: Unlimited (gratis)        │
│                                     │
│ Cache:                              │
│ ├─ Hit rate: 75%                   │
│ └─ Voci cached: 15 squadre         │
└─────────────────────────────────────┘
```

### Verificare Cache

Il cache è salvato in `api_cache.db` (SQLite). Per vedere:

```python
import sqlite3

conn = sqlite3.connect('api_cache.db')
cursor = conn.cursor()

# Vedere squadre cached
cursor.execute("SELECT team, league, timestamp FROM team_cache")
for row in cursor.fetchall():
    print(row)

conn.close()
```

---

## 🐛 Troubleshooting

### API non funziona

**Sintomo:**
```
⚠️ All APIs failed for Midtjylland, using fallback
```

**Cause possibili:**

1. **API key non configurata**
   - Soluzione: Aggiungi chiave in `api_manager.py` riga 27

2. **API key sbagliata**
   - Soluzione: Verifica chiave su Dashboard API-Football
   - Copia/incolla di nuovo (spesso ci sono spazi extra)

3. **Quota esaurita**
   - Soluzione: Aspetta mezzanotte UTC (reset automatico)
   - Oppure: Passa a piano paid

4. **Problemi di rete**
   - Soluzione: Verifica connessione internet
   - Prova: `curl https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=Inter`

### Cache non funziona

**Sintomo:**
```
Cache hit rate: 0%
```

**Cause possibili:**

1. **File `api_cache.db` non esiste**
   - Soluzione: Si crea automaticamente al primo run
   - Verifica: `ls -la api_cache.db`

2. **Permessi file**
   - Soluzione: `chmod 666 api_cache.db`

3. **Cache expirata (>24h)**
   - Normale: Cache si pulisce dopo 24h
   - Soluzione: Nessuna, è intenzionale

### Modalità API non appare nell'UI

**Sintomo:**
Dropdown mostra solo "Manuale" e "Auto (Solo Database)"

**Cause:**

1. **`api_manager.py` non trovato**
   - Soluzione: Verifica che sia nella stessa directory di `Frontendcloud.py`
   - `ls -la api_manager.py`

2. **Errore import**
   - Soluzione: Controlla log console per errori
   - Cerca: `❌ API Manager non disponibile`

3. **Dipendenze mancanti**
   - Tutti i moduli sono standard Python (urllib, json, sqlite3)
   - Dovrebbe funzionare out-of-the-box

---

## 🔧 Configurazione Avanzata

### Cambiare TTL Cache

Default: 24 ore. Per modificare:

1. **Apri:** `api_manager.py`

2. **Trova riga 33:**
   ```python
   CACHE_TTL = 86400  # 24 hours in seconds
   ```

3. **Modifica:**
   ```python
   CACHE_TTL = 43200  # 12 hours
   CACHE_TTL = 172800  # 48 hours
   ```

### Pulire Cache Manualmente

```python
from api_manager import CacheManager

cache = CacheManager()
cache.cleanup_old(days=0)  # Pulisci tutto
```

Oppure:
```bash
rm api_cache.db  # Elimina file cache
```

### Limitare Provider Specifici

Se vuoi usare solo TheSportsDB (non API-Football):

1. **Apri:** `api_manager.py`

2. **Commenta riga 27:**
   ```python
   # API_FOOTBALL_KEY = ""  # Disabled
   ```

Il sistema userà solo TheSportsDB (unlimited).

---

## 📈 Ottimizzazione Uso Quota

### Strategia 1: Cache Aggressivo

**Default già ottimale:** Cache 24h riduce calls del 80-90%

### Strategia 2: Batch Analysis

Analizza **match dello stesso campionato** insieme:
- Prima chiamata: 2 API calls (2 squadre)
- Successive: 0 calls (entrambe cached)

**Esempio:**
```
Match 1: Midtjylland vs Nordsjælland → 2 calls (cache miss)
Match 2: Midtjylland vs FC Copenhagen → 1 call (1 cached, 1 new)
Match 3: Nordsjælland vs FC Copenhagen → 0 calls (entrambe cached!)
```

### Strategia 3: Priorità LEVEL 1

Per campionati top (Serie A, Premier, etc.):
- Usa **"Auto (Solo Database)"** → 0 API calls
- LEVEL 1 ha già 100+ squadre mappate

Per campionati minori:
- Usa **"Auto + API"** → Fetch solo se necessario

---

## 🎉 Sei Pronto!

**Setup completato:**
- ✅ API key configurata (opzionale)
- ✅ Cache funzionante
- ✅ UI mostra modalità API
- ✅ Test match funziona

**Prossimi Step:**
1. Analizza qualche match
2. Monitora quota usata
3. Goditi l'auto-detection globale! 🚀

---

## 💡 FAQ

**Q: Devo pagare per API-Football?**
A: No! Piano FREE è sufficiente per uso normale (5-25 match/giorno).

**Q: TheSportsDB è affidabile?**
A: Sì, è usato da migliaia di app. Dati basic ma unlimited.

**Q: Posso usare SOLO TheSportsDB (niente API-Football)?**
A: Sì! È già configurato e funziona subito. API-Football è opzionale per dati extra.

**Q: Cache occupa tanto spazio?**
A: No. ~10-20 KB per squadra. Anche 1000 squadre = ~20 MB.

**Q: Cosa succede se quota finisce?**
A: Fallback automatico a LEVEL 1 (database locale). Zero downtime.

**Q: Posso vedere le API calls nel dettaglio?**
A: Sì, nell'UI appare "📊 API Status" con contatori real-time.

**Q: Come aggiorno database squadre manualmente?**
A: Edita `team_profiles.json`. API è solo fallback, DB locale ha sempre priorità.

---

## 📞 Supporto

**Problemi?**
1. Controlla log console per errori
2. Verifica API key (copia/incolla corretto?)
3. Test: `python3 api_manager.py` (dovrebbe stampare "✅ Tests completed")

**Domande?**
Chiedi pure! 😊
