# 🎯 PRIORITÀ MIGLIORAMENTI - QUICK REFERENCE

**Ultimo aggiornamento:** 12 Nov 2024
**Rating Software:** ⭐⭐⭐⭐ 8.5/10 (ECCELLENTE)

---

## 🚨 PRIORITÀ IMMEDIATA (Fare ORA)

### 1️⃣ Testing Framework ⚡ ROI: ⭐⭐⭐⭐⭐
**Tempo:** 2-3 giorni
**Impatto:** CRITICO - Permette refactoring sicuro

```bash
# Setup rapido
pip install pytest pytest-cov pytest-mock
mkdir -p tests/{unit,integration}

# Test critici da scrivere SUBITO:
tests/unit/test_kelly_criterion.py       # 10 casi edge
tests/unit/test_dixon_coles.py          # tau function
tests/unit/test_shin_normalization.py   # sum=1.0 guaranteed
tests/unit/test_poisson.py              # overflow cases
tests/integration/test_prediction.py    # End-to-end

# Target: 50% coverage in 3 giorni
pytest tests/ --cov=Frontendcloud --cov-report=html
```

**Perché è urgente:**
- ❌ Nessun test = paura di toccare codice
- ✅ Con tests = refactoring sicuro
- ✅ Catch regression immediata

---

### 2️⃣ Caching Funzioni Pure ⚡ ROI: ⭐⭐⭐⭐
**Tempo:** 1 giorno
**Impatto:** +20-30% performance

```python
from functools import lru_cache

# TOP 10 FUNZIONI DA CACHARE (ordine priorità):
@lru_cache(maxsize=1024)
def poisson_pmf(k: int, lam: float) -> float: ...

@lru_cache(maxsize=256)
def tau_dixon_coles(h: int, a: int, lh: float, la: float, rho: float) -> float: ...

@lru_cache(maxsize=512)
def normalize_three_way_shin(o1: float, ox: float, o2: float): ...

@lru_cache(maxsize=512)
def normalize_two_way_shin(o1: float, o2: float): ...

@lru_cache(maxsize=128)
def decimali_a_prob(odds: float) -> float: ...

@lru_cache(maxsize=128)
def safe_round(x: Optional[float], nd: int = 3): ...

@lru_cache(maxsize=64)
def skellam_pmf(k: int, mu1: float, mu2: float) -> float: ...

@lru_cache(maxsize=256)
def poisson_probabilities(lam: float, max_k: int): ...  # ATTENZIONE: return tuple not list

@lru_cache(maxsize=128)
def entropia_poisson(lam: float, max_k: int = 15): ...

@lru_cache(maxsize=64)
def calc_handicap_from_skellam(lambda_h: float, lambda_a: float, handicap: float): ...
```

**Nota importante:**
```python
# ⚠️ @lru_cache richiede argomenti hashable
# Se funzione ritorna list/dict, cambia a tuple/frozendict
# Esempio:
@lru_cache(maxsize=256)
def poisson_probabilities(lam: float, max_k: int) -> tuple:  # non List!
    probs = calculate_probs(lam, max_k)
    return tuple(probs)  # Convert to tuple for hashability
```

---

## 🟡 PRIORITÀ ALTA (Prossime 2 settimane)

### 3️⃣ Type Hints Complete ⚡ ROI: ⭐⭐⭐
**Tempo:** 2-3 giorni
**Impatto:** Migliora maintainability

```bash
# 1. Installa mypy
pip install mypy

# 2. Check gaps attuali
mypy Frontendcloud.py --ignore-missing-imports > type_gaps.txt

# 3. Fix progressivo
# Start: funzioni pubbliche (API esterne)
# Then: funzioni private
# Last: helper functions

# Target: 90% coverage in 1 settimana
```

**Template da seguire:**
```python
# PRIMA (❌ NO TYPE HINTS):
def calculate_something(x, y, config=None):
    return x + y if config else x - y

# DOPO (✅ TYPE HINTS COMPLETI):
from typing import Optional, Union

def calculate_something(
    x: float,
    y: float,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """Calcola qualcosa con x e y."""
    return x + y if config else x - y
```

---

### 4️⃣ Database Indices ⚡ ROI: ⭐⭐⭐⭐
**Tempo:** 2-3 ore
**Impatto:** Query 10-50x più veloci

```sql
-- Aggiungi questi indices al database SQLite

-- 1. Matches table
CREATE INDEX IF NOT EXISTS idx_matches_date
    ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams
    ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_matches_league
    ON matches(league);

-- 2. Bets table
CREATE INDEX IF NOT EXISTS idx_bets_status
    ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_date_status
    ON bets(bet_date, status);
CREATE INDEX IF NOT EXISTS idx_bets_market
    ON bets(market);

-- 3. Predictions table
CREATE INDEX IF NOT EXISTS idx_predictions_match
    ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_market
    ON predictions(market_type);
CREATE INDEX IF NOT EXISTS idx_predictions_date
    ON predictions(prediction_time);

-- 4. Performance stats
CREATE INDEX IF NOT EXISTS idx_perf_date_market
    ON performance_stats(date, market);
```

**Come applicare:**
```python
# In initialize_database() function
def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # ... existing table creation ...

    # ADD: Create indices
    indices_sql = """
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
    CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
    -- ... other indices ...
    """
    cursor.executescript(indices_sql)
    conn.commit()
    conn.close()
```

---

## 🟢 PRIORITÀ MEDIA (Prossimo mese)

### 5️⃣ Thread Safety ⚡ ROI: ⭐⭐⭐
**Tempo:** 1 settimana
**Impatto:** Abilita deployment multi-thread

**Pattern da applicare:**
```python
import threading

# GLOBAL CACHE
_CACHE = {}
_CACHE_LOCK = threading.Lock()

def get_from_cache(key):
    with _CACHE_LOCK:  # Thread-safe access
        return _CACHE.get(key)

def set_to_cache(key, value):
    with _CACHE_LOCK:  # Thread-safe modification
        _CACHE[key] = value
```

**Cache da proteggere:**
1. `_STATS_BOMB_COMP_CACHE` (globale)
2. `_STATS_BOMB_MATCH_CACHE` (dict)
3. `_FACTORIAL_CACHE_ARRAY` (numpy array)
4. `TTLCache` instances (se shared)

---

### 6️⃣ Refactoring Monolite ⚡ ROI: ⭐⭐⭐⭐
**Tempo:** 3-4 settimane
**Impatto:** Maintainability +50%

**Target structure:**
```
src/
├── core/
│   ├── __init__.py
│   ├── math_models.py      # 200 lines: Dixon-Coles, Poisson
│   ├── kelly.py            # 150 lines: Kelly variants
│   ├── calibration.py      # 200 lines: Platt, Isotonic
│   └── shin.py             # 150 lines: Shin normalization
├── data/
│   ├── __init__.py
│   ├── api_client.py       # 300 lines: API abstraction
│   ├── cache.py            # 100 lines: TTL cache
│   ├── validators.py       # 200 lines: Input validation
│   └── database.py         # 300 lines: SQLite ops
├── features/
│   ├── __init__.py
│   ├── xg.py              # 250 lines: xG integration
│   ├── weather.py         # 150 lines: Weather data
│   ├── form.py            # 200 lines: Team form
│   └── injuries.py        # 150 lines: Injury data
├── utils/
│   ├── __init__.py
│   ├── config.py          # 100 lines: ModelConfig
│   ├── logging.py         # 100 lines: Structured logging
│   └── metrics.py         # 150 lines: Performance metrics
└── prediction.py          # 500 lines: Main orchestration
```

**Benefit:**
- ✅ Import selettivo (faster startup)
- ✅ Testing più facile (isolato)
- ✅ Team collaboration (no conflicts)
- ✅ Reusability (moduli riusabili)

---

## 📋 CHECKLIST RAPIDA

### ✅ Da Fare SUBITO (Oggi/Domani)
- [ ] Setup pytest + scrivere 5 test critici
- [ ] Aggiungere @lru_cache a top 10 funzioni
- [ ] Creare branch dedicato: `feature/testing-caching`

### ✅ Da Fare Questa Settimana
- [ ] Completare 30+ test (target 50% coverage)
- [ ] Aggiungere indices database
- [ ] Run mypy e fixare top 20 type gaps

### ✅ Da Fare Questo Mese
- [ ] Type hints 90% coverage
- [ ] Thread safety per cache globali
- [ ] Start refactoring: split core/math_models.py

---

## 🎯 QUICK WINS (Max 4 ore ciascuno)

### Win #1: Factorial Cache Thread-Safe
```python
# PRIMA:
_FACTORIAL_CACHE_ARRAY = np.array(...)  # Global

# DOPO:
import threading
_factorial_lock = threading.Lock()

def get_factorial(k: int) -> float:
    with _factorial_lock:
        if k < _FACTORIAL_CACHE_ARRAY.shape[0]:
            return _FACTORIAL_CACHE_ARRAY[k]
        # ...
```

### Win #2: Config Validation at Startup
```python
# All'avvio, valida che config sia sensato
def validate_config():
    assert 0 < model_config.EPSILON < 0.01
    assert 0 < model_config.TOL_DIVISION_ZERO < 1e-6
    assert 0.1 < model_config.RHO_MIN < model_config.RHO_MAX < 1.0
    # ... other assertions
    logger.info("✅ Config validation passed")

# Call at module import
validate_config()
```

### Win #3: API Retry Exponential Backoff
```python
# Già implementato parzialmente, ma migliora:
def api_call_with_retry(func, max_retries=4):
    for attempt in range(max_retries):
        try:
            return func()
        except requests.Timeout:
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential: 1s, 2s, 4s, 8s
                time.sleep(delay)
            else:
                raise
```

### Win #4: Lazy Loading API Clients
```python
# PRIMA: Tutti i client caricati all'import
api_football = APIFootballClient()
understat = UnderstatClient()
# ...

# DOPO: Lazy loading
_api_football = None
def get_api_football():
    global _api_football
    if _api_football is None:
        _api_football = APIFootballClient()
    return _api_football
```

---

## 💡 TIPS & TRICKS

### Testing Best Practices
```python
# 1. Use fixtures per setup/teardown
import pytest

@pytest.fixture
def sample_odds():
    return {"odds_1": 1.50, "odds_x": 4.00, "odds_2": 6.00}

def test_normalization(sample_odds):
    result = normalize_three_way_shin(**sample_odds)
    assert abs(sum([1/o for o in result]) - 1.0) < 1e-6

# 2. Parametrize per test multipli
@pytest.mark.parametrize("k,lam,expected", [
    (0, 1.0, 0.368),
    (1, 1.0, 0.368),
    (5, 2.5, 0.067),
])
def test_poisson_pmf(k, lam, expected):
    result = poisson_pmf(k, lam)
    assert abs(result - expected) < 0.01

# 3. Mock API calls
from unittest.mock import patch, Mock

@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value = Mock(status_code=200, json=lambda: {"data": []})
    result = fetch_api_data()
    assert result == {"data": []}
```

### Caching Gotchas
```python
# ⚠️ ATTENZIONE: @lru_cache con float può dare problemi
@lru_cache(maxsize=128)
def bad_cache(x: float):  # ❌ Float equality issues
    return x ** 2

# SOLUZIONE: Round o usa Decimal
from decimal import Decimal

@lru_cache(maxsize=128)
def good_cache(x: float):
    x_rounded = round(x, 6)  # Round to 6 decimals
    return x_rounded ** 2
```

---

**🚀 Start con Testing + Caching questa settimana per ROI immediato!**
