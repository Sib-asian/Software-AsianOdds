# FIX: BUG DI GESTIONE STATO
# Strategie concrete di correzione

---

## FIX #1-4: CACHE GLOBALI SENZA TTL

### Problema (linee 11451-11685 Frontendcloud.py)
```python
_STATS_BOMB_MATCH_CACHE = {}  # UNBOUNDED - MEMORY LEAK
_STATS_BOMB_EVENTS_CACHE = {}  # UNBOUNDED - MEMORY LEAK
_STATS_BOMB_TEAM_METRICS_CACHE = {}  # UNBOUNDED - MEMORY LEAK
_FOOTBALL_DATA_MATCH_CACHE = {}  # UNBOUNDED - MEMORY LEAK
```

### Soluzione: Implementare classe CachedStorage con TTL

**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Azione**: Sostituire linee 11451-11687 con:

```python
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

class CachedStorage:
    """Generic cache with TTL and size limit."""
    
    def __init__(self, name: str, max_size: int = 100, ttl_seconds: int = 3600):
        self.name = name
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: Dict[Any, Any] = {}
        self.timestamps: Dict[Any, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self._lock:
            if key not in self.cache:
                return None
            
            age = time.time() - self.timestamps[key]
            if age > self.ttl:
                logger.debug(f"[{self.name}] Cache expired for key {key} (age: {age:.1f}s)")
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: Any, value: Any) -> None:
        """Set value, removing oldest entry if at capacity."""
        with self._lock:
            # Enforce size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest 10% to avoid thrashing
                items_to_remove = max(1, self.max_size // 10)
                for _ in range(items_to_remove):
                    oldest_key = min(self.timestamps, key=self.timestamps.get)
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                logger.info(f"[{self.name}] Cache evicted {items_to_remove} oldest entries")
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            logger.debug(f"[{self.name}] Cached key {key} (size: {len(self.cache)}/{self.max_size})")
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info(f"[{self.name}] Cache cleared")
    
    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [k for k, t in self.timestamps.items() 
                      if now - t > self.ttl]
            for k in expired:
                del self.cache[k]
                del self.timestamps[k]
            if expired:
                logger.info(f"[{self.name}] Cleaned {len(expired)} expired entries")
            return len(expired)
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)

# Initialize caches with TTL (1 hour default)
_STATS_BOMB_MATCH_CACHE = CachedStorage("StatsBomb Matches", max_size=500, ttl_seconds=3600)
_STATS_BOMB_EVENTS_CACHE = CachedStorage("StatsBomb Events", max_size=200, ttl_seconds=3600)
_STATS_BOMB_TEAM_METRICS_CACHE = CachedStorage("StatsBomb Team Metrics", max_size=100, ttl_seconds=3600)
_FOOTBALL_DATA_MATCH_CACHE = CachedStorage("Football Data Matches", max_size=300, ttl_seconds=3600)
_STATS_BOMB_COMP_CACHE = CachedStorage("StatsBomb Competitions", max_size=1, ttl_seconds=86400)
```

**Update delle funzioni** (linee 11491-11504):
```python
def statsbomb_get_matches(competition_id: int, season_id: int) -> List[Dict[str, Any]]:
    """Recupera lista partite per competition/season (cache con TTL)."""
    cache_key = f"{competition_id}_{season_id}"
    
    # Try cache first
    cached = _STATS_BOMB_MATCH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        matches = _statsbomb_fetch_json(f"matches/{competition_id}/{season_id}.json")
        if not isinstance(matches, list):
            matches = []
        _STATS_BOMB_MATCH_CACHE.set(cache_key, matches)
    except requests.exceptions.RequestException:
        _STATS_BOMB_MATCH_CACHE.set(cache_key, [])
    
    return _STATS_BOMB_MATCH_CACHE.get(cache_key) or []
```

Applica pattern simile per:
- `statsbomb_get_events()` (linea 11507)
- `statsbomb_get_team_metrics()` (linea 11522)
- `football_data_get_recent_matches()` (linea 11151)

---

## FIX #5: API_CACHE STALE ENTRIES

### Problema (linea 345)
```python
API_CACHE = {}  # Ha TTL ma non rimuove entry scadute
```

### Soluzione: Aggiungere cleanup periodico

**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Azione**: Dopo linea 514 (dopo return result), aggiungere:

```python
# Periodic cleanup function (call every hour)
def cleanup_api_cache() -> int:
    """Remove expired entries from API cache."""
    with _API_CACHE_LOCK:
        now = time.time()
        expired = [k for k, (_, t) in API_CACHE.items() 
                   if now - t >= CACHE_EXPIRY]
        for k in expired:
            del API_CACHE[k]
        if expired:
            logger.info(f"🧹 Cleaned {len(expired)} expired API cache entries (total: {len(API_CACHE)})")
        return len(expired)

# Schedule cleanup (in main streamlit execution)
# Option 1: Call at module startup
if not st.session_state.get("_api_cache_cleanup_scheduled"):
    import atexit
    def schedule_cleanup():
        import threading
        def cleanup_loop():
            while True:
                time.sleep(3600)  # Every hour
                cleanup_api_cache()
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    schedule_cleanup()
    st.session_state["_api_cache_cleanup_scheduled"] = True
```

---

## FIX #6: SESSION STATE INITIALIZATION INCONSISTENCY

### Problema (linee 14915-14916, 15419-15422)
```python
# PRIMO CHECK
if "preview_cache" not in st.session_state:
    st.session_state["preview_cache"] = {}

# SECONDO CHECK (timestamps manca!)
if "preview_cache" not in st.session_state:
    st.session_state["preview_cache"] = {}
if "preview_cache_timestamps" not in st.session_state:  # Only here
    st.session_state["preview_cache_timestamps"] = {}
```

### Soluzione: Centralizzare initialization

**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Azione**: 
1. Inserire funzione all'inizio (dopo st.set_page_config):

```python
def init_session_state():
    """Initialize all session state variables atomically."""
    defaults = {
        # Match data
        "current_match": None,
        "position_home_auto": None,
        "position_away_auto": None,
        "is_derby_auto": False,
        "is_cup_auto": False,
        "is_end_season_auto": False,
        "auto_mode_selection": "✋ Manuale",
        
        # Cache (ALWAYS TOGETHER)
        "preview_cache": {},
        "preview_cache_timestamps": {},
        
        # Database
        "database_initialized": False,
        "soccer_leagues": [],
        "events_for_league": [],
        "selected_event_prices": {},
        "selected_league_key": None,
        "selected_event_id": None,
        
        # API
        "api_insight": None,
        "api_error": None,
        "api_loading": False,
        
        # Telegram
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "telegram_prob_threshold": 0.0,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Call at module level
init_session_state()
```

2. Rimuovere i vecchi check sparsi (linee 14915-14916, 15419-15422, 14454-14474, etc.)

---

## FIX #7: PREVIEW CACHE SIZE LIMIT RACE

### Problema (linee 15454-15459)
```python
if len(st.session_state["preview_cache"]) >= 50:
    # ... remove oldest
```

### Soluzione: Atomic cache management

**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Azione**: Dopo `init_session_state()`, aggiungere:

```python
class PreviewCacheManager:
    """Manage preview cache with atomic operations."""
    
    MAX_SIZE = 50
    CLEANUP_THRESHOLD = 0.9  # Cleanup at 90% capacity
    
    @staticmethod
    def try_add(cache_key: str, value: Any) -> bool:
        """Try to add entry. Returns True if added."""
        cache = st.session_state.get("preview_cache", {})
        timestamps = st.session_state.get("preview_cache_timestamps", {})
        
        # Check if needs cleanup
        if len(cache) >= PreviewCacheManager.MAX_SIZE:
            PreviewCacheManager.cleanup()
        
        # Now we have space
        st.session_state["preview_cache"][cache_key] = value
        st.session_state["preview_cache_timestamps"][cache_key] = time.time()
        return True
    
    @staticmethod
    def cleanup():
        """Remove oldest 20% of entries."""
        cache = st.session_state.get("preview_cache", {})
        timestamps = st.session_state.get("preview_cache_timestamps", {})
        
        if not timestamps:
            return
        
        # Remove oldest 20% (10 entries if MAX_SIZE=50)
        items_to_remove = max(1, PreviewCacheManager.MAX_SIZE // 5)
        sorted_by_age = sorted(timestamps.items(), key=lambda x: x[1])
        
        for key, _ in sorted_by_age[:items_to_remove]:
            if key in st.session_state["preview_cache"]:
                del st.session_state["preview_cache"][key]
            if key in st.session_state["preview_cache_timestamps"]:
                del st.session_state["preview_cache_timestamps"][key]
        
        logger.info(f"🧹 Removed {items_to_remove} oldest cache entries (new size: {len(cache)})")
    
    @staticmethod
    def get(cache_key: str) -> Optional[Any]:
        """Get value safely."""
        cache = st.session_state.get("preview_cache", {})
        timestamps = st.session_state.get("preview_cache_timestamps", {})
        
        if cache_key not in cache:
            return None
        
        # Check TTL (5 minutes = 300 seconds)
        age = time.time() - timestamps.get(cache_key, 0)
        if age > 300:
            # Expired
            if cache_key in st.session_state["preview_cache"]:
                del st.session_state["preview_cache"][cache_key]
            if cache_key in st.session_state["preview_cache_timestamps"]:
                del st.session_state["preview_cache_timestamps"][cache_key]
            return None
        
        return cache[cache_key]
```

**Usage** (replace linee 15454-15464):
```python
# Instead of:
# if len(st.session_state["preview_cache"]) >= 50: ...

# Use:
is_cache_valid = (
    PreviewCacheManager.get(cache_key) is not None
)

if is_cache_valid:
    auto_features_preview = PreviewCacheManager.get(cache_key)
else:
    # Compute features
    auto_features_preview = auto_detect_all_features(...)
    PreviewCacheManager.try_add(cache_key, auto_features_preview)
```

---

## FIX #8: UNSAFE SESSION STATE ACCESS

### Problema
```python
# UNSAFE (linea 15819)
auto_features = st.session_state["preview_cache"][cache_key]

# SAFE
cache_time = st.session_state.get("preview_cache_timestamps", {}).get(cache_key, 0)
```

### Soluzione: Consistent .get() pattern

**File**: `/home/user/Software-AsianOdds/Frontendcloud.py`
**Azione**: Replace ALL direct accesses with .get():

```python
# PATTERN: Always use .get() with default
value = st.session_state.get("key", default_value)

# For nested:
value = st.session_state.get("dict_key", {}).get("inner_key", default)

# Examples (linea 15819):
# BEFORE:
auto_features = st.session_state["preview_cache"][cache_key]

# AFTER:
auto_features = st.session_state.get("preview_cache", {}).get(cache_key)
if auto_features is None:
    # Handle gracefully
    logger.warning(f"Cache miss for {cache_key}")
    auto_features = compute_features()
```

---

## FIX #9: GLOBAL TEAM_PROFILES MUTATION

### Problema (auto_features.py linee 103-121)
```python
TEAM_PROFILES = load_team_profiles()  # Global

def reload_team_profiles(json_path):
    global TEAM_PROFILES
    TEAM_PROFILES = load_team_profiles(json_path)  # Unsafe mutation
```

### Soluzione: Thread-safe manager

**File**: `/home/user/Software-AsianOdds/auto_features.py`
**Azione**: Sostituire linee 103-121 con:

```python
import threading
from copy import deepcopy

class TeamProfilesManager:
    """Thread-safe manager for team profiles."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._profiles = load_team_profiles()
        self._timestamp = time.time()
        logger.info(f"✅ TeamProfilesManager initialized with {len(self._profiles.get('teams', {}))} teams")
    
    def get(self, copy: bool = True) -> Dict:
        """Get current profiles (optionally as copy)."""
        with self._lock:
            return deepcopy(self._profiles) if copy else self._profiles
    
    def get_team(self, team_name: str) -> Optional[Dict]:
        """Get specific team profile."""
        with self._lock:
            teams = self._profiles.get("teams", {})
            return deepcopy(teams.get(team_name))
    
    def reload(self, json_path: str = "team_profiles.json") -> bool:
        """Reload profiles from file. Thread-safe."""
        try:
            with self._lock:
                new_profiles = load_team_profiles(json_path)
                self._profiles = new_profiles
                self._timestamp = time.time()
                logger.info(f"🔄 Team profiles reloaded from {json_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to reload team profiles: {e}")
            return False
    
    def is_stale(self, max_age_hours: int = 1) -> bool:
        """Check if profiles are stale."""
        with self._lock:
            age_hours = (time.time() - self._timestamp) / 3600
            return age_hours > max_age_hours

# Global instance (thread-safe)
TEAM_PROFILES_MGR = TeamProfilesManager()

# For backward compatibility
def get_team_profiles() -> Dict:
    """Public API to get team profiles."""
    return TEAM_PROFILES_MGR.get()

def reload_team_profiles(json_path: str = "team_profiles.json") -> Dict:
    """Public API to reload profiles."""
    TEAM_PROFILES_MGR.reload(json_path)
    return TEAM_PROFILES_MGR.get()
```

**Update callers**: Dove era usato `TEAM_PROFILES`, aggiungere:
```python
# BEFORE:
team_config = TEAM_PROFILES["teams"][team_name]

# AFTER:
team_config = TEAM_PROFILES_MGR.get_team(team_name)
if team_config is None:
    logger.warning(f"Team {team_name} not found in profiles")
    team_config = {}  # Use defaults
```

---

## FIX #10-11: SESSION STATE INITIALIZATION TIMING

### Problema
- Inizializzazione sparsa in 10+ posti
- Dipendente dall'ordine di esecuzione
- Rerun durante init causa stato parziale

### Soluzione: Singola funzione di init

Vedi **FIX #6** sopra - `init_session_state()` centralizzato.

Rimuovere tutti gli altri check:
- Linee 14454-14474: `database_initialized`, `soccer_leagues`, etc.
- Linee 14885-14916: `current_match`, match state fields
- dashboard.py linee 98-100: `api_insight`, `api_error`

---

## CHECKLIST IMPLEMENTAZIONE

### Priority 1 (CRITICAL)
- [ ] Implementare `CachedStorage` per _STATS_BOMB_* e _FOOTBALL_DATA_*
- [ ] Centralizzare `init_session_state()`
- [ ] Rimuovere vecchi check sparsi

### Priority 2 (HIGH)
- [ ] Implementare `PreviewCacheManager`
- [ ] Implementare `TeamProfilesManager`
- [ ] Aggiungere cleanup per API_CACHE

### Priority 3 (MEDIUM)
- [ ] Controllare e aggiornare ALL session_state accessi a usare .get()
- [ ] Aggiungere logging per cache operations
- [ ] Aggiungere test per race condition

### Testing
```python
# Test cache TTL
def test_cache_ttl():
    cache = CachedStorage("test", ttl_seconds=2)
    cache.set("key", "value")
    assert cache.get("key") == "value"
    time.sleep(2.1)
    assert cache.get("key") is None

# Test size limit
def test_cache_size_limit():
    cache = CachedStorage("test", max_size=3)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")
    cache.set("k4", "v4")  # Should evict k1 (oldest)
    assert cache.get("k1") is None
    assert cache.get("k4") == "v4"
```

