# REPORT: BUG DI GESTIONE STATO (STATE MANAGEMENT)
# Analisi Repository: Software-AsianOdds

## SEVERITA CRITICITA
- CRITICAL: Problemi che causano data loss o inconsistenza grave
- HIGH: Memory leaks, race condition, cache stale
- MEDIUM: Stato inconsistente, accessi non sicuri
- LOW: Code smell, non segue best practice

---

## 1. CACHE GLOBALI SENZA TTL (MEMORY LEAKS CRITICI)

### BUG #1: _STATS_BOMB_MATCH_CACHE - UNBOUNDED GROWTH
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 11451-11504
**Severita**: CRITICAL

```python
_STATS_BOMB_MATCH_CACHE: Dict[str, List[Dict[str, Any]]] = {}  # Line 11452

def statsbomb_get_matches(competition_id: int, season_id: int):
    cache_key = f"{competition_id}_{season_id}"
    if cache_key in _STATS_BOMB_MATCH_CACHE:
        return _STATS_BOMB_MATCH_CACHE[cache_key]
    # ... fetch ...
    _STATS_BOMB_MATCH_CACHE[cache_key] = matches  # Line 11501
    return _STATS_BOMB_MATCH_CACHE[cache_key]
```

**Problema**: 
- Dictionary globale senza limite di size
- Nessun TTL/cache invalidation
- Cresce infinitamente durante l'esecuzione dell'app
- Memory leak progressivo

**Impatto**: 
- La memoria cresce senza limite
- App rallenta nel tempo
- Se app gira 24h, può crashare per OOM

**Race condition possibile**: SI
- Accesso concorrente dalla API e da UI
- Dict.update() in Python non è atomico
- Due thread potrebbero sovrascrivere contemporaneamente

**Fix**:
```python
from functools import lru_cache
from time import time

class CachedStatsBomb:
    def __init__(self, max_size=100, ttl=3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest]
            del self.timestamps[oldest]
        self.cache[key] = value
        self.timestamps[key] = time()
```

---

### BUG #2: _STATS_BOMB_EVENTS_CACHE - UNBOUNDED GROWTH
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 11452, 11507-11519
**Severita**: CRITICAL

```python
_STATS_BOMB_EVENTS_CACHE: Dict[int, List[Dict[str, Any]]] = {}  # Line 11453

def statsbomb_get_events(match_id: int):
    if match_id in _STATS_BOMB_EVENTS_CACHE:
        return _STATS_BOMB_EVENTS_CACHE[match_id]
    # ...
    _STATS_BOMB_EVENTS_CACHE[match_id] = events  # Line 11516
```

**Problema**: Identico a #1
- Nessun size limit
- Nessun TTL
- Ogni match_id aggiunto è conservato per sempre

**Impatto**: CRITICAL
- Match events sono grandi strutture dati JSON (100KB+)
- Molti match = gigabyte di memoria
- Memory leak progressivo

**Race condition possibile**: SI

**Fix**: Identico a #1

---

### BUG #3: _STATS_BOMB_TEAM_METRICS_CACHE - UNBOUNDED GROWTH
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 11454, 11522-11685
**Severita**: CRITICAL

```python
_STATS_BOMB_TEAM_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}  # Line 11454

def statsbomb_get_team_metrics(team_name: str):
    cache_key = team_name.strip().lower()
    if cache_key in _STATS_BOMB_TEAM_METRICS_CACHE:
        return deepcopy(_STATS_BOMB_TEAM_METRICS_CACHE[cache_key])
    # ...
    _STATS_BOMB_TEAM_METRICS_CACHE[cache_key] = result  # Line 11536, 11591, 11666
```

**Problema**: Identico a #1 e #2
- Nessun limite
- Nessun TTL

**Impatto**: CRITICAL

**Race condition possibile**: SI

---

### BUG #4: _FOOTBALL_DATA_MATCH_CACHE - UNBOUNDED GROWTH
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 11151-11178
**Severita**: CRITICAL

```python
_FOOTBALL_DATA_MATCH_CACHE: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}  # Line 347

def football_data_get_recent_matches(team_id: int, limit: int = 5):
    cache_key = (int(team_id), int(limit))
    if cache_key in _FOOTBALL_DATA_MATCH_CACHE:
        return _FOOTBALL_DATA_MATCH_CACHE[cache_key]
    # ...
    _FOOTBALL_DATA_MATCH_CACHE[cache_key] = matches  # Line 11174
```

**Problema**: Identico a #1, #2, #3
- Nessun TTL
- Nessun size limit

**Impatto**: CRITICAL

**Race condition possibile**: SI

---

### BUG #5: API_CACHE - UNBOUNDED GROWTH
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 344-345
**Severita**: HIGH

```python
API_CACHE = {}  # Line 345

def retry_with_cache(...):
    # Lines 489-493: Read with TTL check
    with _API_CACHE_LOCK:
        if cache_key in API_CACHE:
            cached_data, cached_time = API_CACHE[cache_key]
            if time.time() - cached_time < CACHE_EXPIRY:
                return cached_data
    
    # Lines 510-511: Write
    with _API_CACHE_LOCK:
        API_CACHE[cache_key] = (result, time.time())
```

**Problema**:
- Ha TTL (CACHE_EXPIRY secondi)
- MA: non rimuove le entry scadute, solo non le usa
- Dictionary cresce infinitamente (entry scadute rimangono)
- Accessi (489-511, 9702-9717) sono thread-safe con lock, MA...

**Impatto**: MEDIUM
- Memory leak parziale (cresce meno velocemente grazie a TTL)
- Stale cache dopo CACHE_EXPIRY (non rimuove i dati vecchi)
- In production con 24h+ uptime, cache invecchia ma non viene ripulito

**Race condition possibile**: NO (ha lock)

**Fix**:
```python
# Aggiungi cleanup periodico
def cleanup_expired_api_cache():
    with _API_CACHE_LOCK:
        now = time.time()
        expired = [k for k, (_, t) in API_CACHE.items() 
                   if now - t >= CACHE_EXPIRY]
        for k in expired:
            del API_CACHE[k]
        logger.info(f"🧹 Cleaned {len(expired)} expired API cache entries")

# Chiama periodicamente (ogni 1h)
# O: Usa maxsize dict con TTL
```

---

## 2. SESSION STATE ISSUES

### BUG #6: st.session_state["preview_cache"] - INCONSISTENT INITIALIZATION
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 14915-14916, 15419-15422
**Severita**: HIGH

```python
# PRIMO CHECK (line 14915)
if "preview_cache" not in st.session_state:
    st.session_state["preview_cache"] = {}

# SECONDO CHECK (lines 15419-15422)
if "preview_cache" not in st.session_state:
    st.session_state["preview_cache"] = {}
if "preview_cache_timestamps" not in st.session_state:
    st.session_state["preview_cache_timestamps"] = {}
```

**Problema**:
- Inizializzazione in 2 posti diversi
- preview_cache_timestamps inizializzato solo nel secondo check
- Se rerun accade tra i due check, timestamps potrebbe essere mancante
- preview_cache MAY EXIST ma preview_cache_timestamps NO

**Impatto**: HIGH
- KeyError se accedi a preview_cache_timestamps quando non esiste
- Stato inconsistente tra cache e timestamps
- Crash durante operazioni cache

**Race condition possibile**: SI (in Streamlit rerun)
- Durante rerun, session state viene letto/scritto da UI e callback
- Due thread potrebbero entrare in check2 contemporaneamente

**Fix**:
```python
# CENTRALIZED INITIALIZATION
def ensure_cache_initialized():
    """Ensure both cache and timestamps exist"""
    if "preview_cache" not in st.session_state:
        st.session_state["preview_cache"] = {}
    if "preview_cache_timestamps" not in st.session_state:
        st.session_state["preview_cache_timestamps"] = {}

# Call ONCE all'inizio della pagina
ensure_cache_initialized()
```

---

### BUG #7: preview_cache SIZE LIMIT NOT ENFORCED CONSISTENTLY
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 15454-15459
**Severita**: MEDIUM

```python
if len(st.session_state["preview_cache"]) >= 50:
    oldest_key = min(st.session_state["preview_cache_timestamps"],
                     key=st.session_state["preview_cache_timestamps"].get)
    del st.session_state["preview_cache"][oldest_key]
    del st.session_state["preview_cache_timestamps"][oldest_key]
```

**Problema**:
- Size limit SOLO quando aggiunge nuova entry
- Se user refresh page → session_state reset
- Cache rebuilds senza limite fino a 50
- Timing issue: check a 50, poi aggiunge → può arrivare a 51 entry

**Impatto**: MEDIUM
- Occupazione memoria extra
- Possibile crescita oltre limite dichiarato

**Race condition possibile**: SI
- Se due thread aggiungono simultaneamente:
  - Thread A: len() = 49
  - Thread B: len() = 49
  - Entrambi passano il check
  - Risultato: 51 entry

**Fix**:
```python
# Usa lock se in contesto multi-thread
# O: decrement size limit durante aggiunta
MAX_CACHE_SIZE = 50

# ATOMI: Check + Add
if len(st.session_state["preview_cache"]) >= MAX_CACHE_SIZE:
    # Remove 10% (5 entries) to avoid thrashing
    items_to_remove = MAX_CACHE_SIZE // 10
    for _ in range(items_to_remove):
        oldest_key = min(st.session_state["preview_cache_timestamps"],
                        key=st.session_state["preview_cache_timestamps"].get)
        del st.session_state["preview_cache"][oldest_key]
        del st.session_state["preview_cache_timestamps"][oldest_key]

st.session_state["preview_cache"][cache_key] = value
st.session_state["preview_cache_timestamps"][cache_key] = time.time()
```

---

### BUG #8: st.session_state ACCESSI MISTI (GET vs DIRECT ACCESS)
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: Multiple (15813-15819, 15800-15860)
**Severita**: MEDIUM

```python
# UNSAFE: Direct access (line 15819)
auto_features = st.session_state["preview_cache"][cache_key]

# SAFE: Using .get() (line 15814)
cache_time = st.session_state.get("preview_cache_timestamps", {}).get(cache_key, 0)
```

**Problema**:
- Misto di accessi sicuri e unsafe
- Se key manca → KeyError crash
- Inconsistente: a volte usa .get(), a volte no

**Impatto**: HIGH
- Possibili crash runtime
- Difficile debuggare

**Race condition possibile**: SI
- Tra il check `if key in dict` e l'accesso `dict[key]`
- Altro thread potrebbe eliminare la key

**Fix**:
```python
# SEMPRE usare .get() con default
auto_features = st.session_state.get("preview_cache", {}).get(cache_key)
if auto_features is None:
    # Handle missing cache gracefully
    auto_features = compute_features()
```

---

## 3. GLOBAL VARIABLE MUTATIONS

### BUG #9: TEAM_PROFILES - GLOBAL STATE MUTABLE
**File**: `/home/user/Software-AsianOdds/auto_features.py`
**Linee**: 103-121
**Severita**: MEDIUM

```python
TEAM_PROFILES = load_team_profiles()  # Line 103 - GLOBAL

def reload_team_profiles(json_path: str = "team_profiles.json") -> Dict:
    global TEAM_PROFILES  # Line 118
    TEAM_PROFILES = load_team_profiles(json_path)  # Line 119
    logger.info(f"🔄 Team profiles ricaricati")
    return TEAM_PROFILES
```

**Problema**:
- Global mutable dictionary
- Modificabile con `reload_team_profiles()`
- Se file JSON cambia durante esecuzione, stato diventa inconsistente
- Thread A legge TEAM_PROFILES
- Thread B chiama reload_team_profiles() → modifica TEAM_PROFILES
- Thread A usa dati vecchi/nuovi inconsistenti

**Impatto**: MEDIUM
- Stato inconsistente tra thread
- Calcoli basati su dati parzialmente aggiornati

**Race condition possibile**: SI
- reload_team_profiles() non è thread-safe
- Accesso concorrente da calcoli e reload

**Fix**:
```python
import threading

class TeamProfilesManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._profiles = load_team_profiles()
        self._timestamp = time.time()
    
    def get(self):
        with self._lock:
            return deepcopy(self._profiles)  # Return copy
    
    def reload(self, json_path="team_profiles.json"):
        with self._lock:
            self._profiles = load_team_profiles(json_path)
            self._timestamp = time.time()

TEAM_PROFILES_MGR = TeamProfilesManager()
```

---

## 4. STATO DIPENDENTE DA ORDINE DI ESECUZIONE

### BUG #10: dashboard.py - SESSION STATE INITIALIZATION TIMING
**File**: `/home/user/Software-AsianOdds/dashboard.py`
**Linee**: 98-131
**Severita**: HIGH

```python
# Lines 98-100
if "api_insight" not in st.session_state:
    st.session_state["api_insight"] = None
    st.session_state["api_error"] = None

# Lines 122-131
st.session_state["api_insight"] = { ... }  # Updated via button
st.session_state["api_error"] = None

# Lines 418-421
if st.session_state.get("api_error"):
    st.error(...)
api_insight = st.session_state.get("api_insight")
```

**Problema**:
- Stato dipende dall'ordine di bottoni cliccati
- Se user clicca bottone prima che session_state sia inizializzato
- Oppure: rerun order is non-deterministic
- api_error viene letto prima di essere scritto in altri path

**Impatto**: MEDIUM
- UI mostra valori vecchi/None
- Errori non visualizzati se accadono nel primo rerun

**Race condition possibile**: SI (minimal in Streamlit single-threaded ma...)
- Session state reads/writes during callback execution

**Fix**:
```python
# Centralized state initialization with sentinel pattern
def init_session_state():
    defaults = {
        "api_insight": None,
        "api_error": None,
        "api_loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()  # Call at module level
```

---

### BUG #11: Frontendcloud.py - MATCH DATA INITIALIZATION SCATTERED
**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Linee**: 14885-14916
**Severita**: MEDIUM

```python
# Lines 14885-14900
if "current_match" not in st.session_state:
    st.session_state["current_match"] = {
        "home_team": "Casa",
        "away_team": "Trasferta",
        # ... many more fields
    }

# Lines 14903-14912 (separate if statements)
if "position_home_auto" not in st.session_state:
    st.session_state["position_home_auto"] = None
# ... 6 more separate if statements ...
```

**Problema**:
- Stato dipendente da ordine di esecuzione di if statements
- Se rerun avviene tra gli if statement, stato parziale
- 10+ separate if statements per inizializzazione
- Difficile garantire stato consistente

**Impatto**: MEDIUM
- Uno dei campi potrebbe non essere inizializzato
- UI widget accede a campo non inizializzato → crash

**Race condition possibile**: SI
- Rerun durante inizializzazione

**Fix**:
```python
# Atomic initialization
def init_match_state():
    if "current_match" not in st.session_state:
        st.session_state["current_match"] = {...}
    
    match_state_template = {
        "position_home_auto": None,
        "position_away_auto": None,
        "is_derby_auto": False,
        "is_cup_auto": False,
        "is_end_season_auto": False,
        "auto_mode_selection": "✋ Manuale",
        "preview_cache": {},
    }
    
    for key, default_value in match_state_template.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_match_state()  # Single call
```

---

## SUMMARY DELLA SEVERITA

| BUG | File | Linea | Severita | Tipo | TTL? | Thread-Safe? |
|-----|------|-------|----------|------|------|--------------|
| #1 | Frontendcloud.py | 11451-11504 | CRITICAL | Cache Growth | NO | NO |
| #2 | Frontendcloud.py | 11452, 11507 | CRITICAL | Cache Growth | NO | NO |
| #3 | Frontendcloud.py | 11454, 11522 | CRITICAL | Cache Growth | NO | NO |
| #4 | Frontendcloud.py | 347, 11151 | CRITICAL | Cache Growth | NO | NO |
| #5 | Frontendcloud.py | 344-345 | HIGH | Cache Growth | YES (partial) | YES |
| #6 | Frontendcloud.py | 14915, 15419 | HIGH | Init Inconsistency | - | NO |
| #7 | Frontendcloud.py | 15454 | MEDIUM | Cache Size Race | - | NO |
| #8 | Frontendcloud.py | 15813-15819 | MEDIUM | Unsafe Access | - | NO |
| #9 | auto_features.py | 103-121 | MEDIUM | Global Mutation | - | NO |
| #10 | dashboard.py | 98-131 | HIGH | Init Timing | - | NO |
| #11 | Frontendcloud.py | 14885-14916 | MEDIUM | Scattered Init | - | NO |

---

## RECOMMENDATIONS (PRIORITA)

### 1. IMMEDIATE (Critical)
- Implementare TTL + size limit per _STATS_BOMB_* caches
- Implementare TTL + size limit per _FOOTBALL_DATA_MATCH_CACHE
- Centralize session_state initialization

### 2. HIGH PRIORITY
- Implementare cleanup periodico per API_CACHE
- Rendere TEAM_PROFILES thread-safe
- Usar sempre .get() per session_state accessi

### 3. MEDIUM PRIORITY
- Implementare atomic session_state updates
- Aggiungere lock per preview_cache operations
- Refactor scattered initialization calls

