# REPORT PROBLEMI DI PERFORMANCE - Software-AsianOdds

## SOMMARIO
**Criticità:** ALTA | **File principale:** Frontendcloud.py (16.650 linee) | **Impatto stimato:** 500-2000ms per operazioni critiche

---

## 1. O(n²) NESTED LOOPS - CRITICI ⚠️

### PROBLEMA PRINCIPALE: Loop annidati su matrice (mg × mg)
- **mg** (max goals) è typicamente 10, quindi 121 iterazioni
- Ma può scalare (worst case: 15×15 = 225)
- **Numero di funzioni affette: 9+**

#### 1.1 LINEA 7916-7918: dist_gol_da_matrice()
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        p = mat[h][a]
        # ⚠️ Kahan summation + 4 validazioni per iterazione
        if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
            continue
```
- **Complessità:** O(mg²) ≈ O(121) @ mg=10
- **Operazioni per iterazione:** 5 (type check, comparison, NaN check, isfinite, Kahan math)
- **Impatto:** ~600 operazioni per matrice
- **Frequenza:** Chiamata in ogni calcolo probabilità match
- **Fix suggerito:** Vectorizzare con NumPy `np.where()` oppure pre-validare matrice una volta

#### 1.2 LINEA 8011-8023: dist_totalgol_da_matrice()
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        p = mat[h][a]
        if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
            continue
        tot = h + a
        if tot < len(dist):
            # Kahan summation
```
- **Complessità:** O(mg²)
- **Protezioni ridondanti:** 5 check per ogni iterazione
- **Fix suggerito:** Validare matrice UNA VOLTA al caricamento, non in ogni loop

#### 1.3 LINEA 8127-8152: prob_esito_over_from_matrix()
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        if h + a <= soglia:
            continue
        p = mat[h][a]
        # ⚠️ 4 validazioni per ogni iterazione
        if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
            continue
        if esito == '1' and h > a:  # Esito check inside loop
```
- **Complessità:** O(mg²)
- **Anti-pattern:** Esito check ripetuto 121 volte (dovrebbe essere fuori loop)
- **Impatto:** ~600 iterazioni × 6 operazioni = 3600+ operazioni

#### 1.4-1.9: ALTRE FUNZIONI CON O(mg²)
Linee **8209-8210, 8288-8289, 8357-8358, 8423-8424, 8486-8487, 8545-8546, 8583-8587**
- Tutte hanno pattern identico: `for h in range(mg + 1): for a in range(mg + 1):`
- Totale: **7+ funzioni con nested loops**

**IMPATTO TOTALE:** Se mg=10, ogni funzione fa ~121 iterazioni
- Se chiamate una volta per match: OK (~1-2ms per funzione)
- Se callate in loop per 100 match: **100-200ms overhead**
- Per UI Streamlit con calcoli in real-time: VISIBILE (stuttering)

**FIX URGENTE:**
```python
# PRIMA (current - O(mg²))
for h in range(mg + 1):
    for a in range(mg + 1):
        if validation_check(mat[h][a]):
            accumulate(h, a)

# DOPO (vectorized - O(mg) o O(1) con caching)
# Opzione 1: NumPy - se possibile convertire a array
valid_mat = np.where((mat > 0) & (np.isfinite(mat)), mat, 0)
result = vectorized_operation(valid_mat)

# Opzione 2: Pre-validate matrice una sola volta, poi accedi direttamente
# Opzione 3: Caching se stessa matrice viene usata multiple volte
```

---

## 2. DEEPCOPY INEFFICIENTE 🔴

### PROBLEMA: deepcopy() ritornato da cache function
**Linee:** 11531, 11537, 11592, 11667, 11686

```python
# LINEA 11531
def _get_team_metrics_cached(team_name: str):
    cache_key = f"team_metrics::{team_name}"
    if cache_key in _STATS_BOMB_TEAM_METRICS_CACHE:
        return deepcopy(_STATS_BOMB_TEAM_METRICS_CACHE[cache_key])  # ❌ deepcopy ogni volta!
    ...
    return deepcopy(result)  # ❌ deepcopy anche del risultato nuovo
```

- **Complessità:** O(n) dove n = size del dict ritornato
- **Frequenza:** ALTA (ogni volta che chiami `_get_team_metrics_cached()`)
- **Impatto:** ~10-50ms per deepcopy di dict complesso
- **Numero di occorrenze:** 5 linee

**Motivo della inefficienza:** Se cache contiene dict con 100+ chiavi/valori, deepcopy è costoso
- Memoria: Copia intera struttura dati
- CPU: Walk completo di tutte le proprietà

**FIX SUGGERITO:**
```python
# PROBLEMA ORIGINALE
return deepcopy(_STATS_BOMB_TEAM_METRICS_CACHE[cache_key])

# SOLUZIONE: Non copiare, implementare read-only interface
# Opzione 1: Ritornare MappingProxyType (read-only dict view)
from types import MappingProxyType
return MappingProxyType(_STATS_BOMB_TEAM_METRICS_CACHE[cache_key])

# Opzione 2: Se devi modificare, copia solo gli elementi necessari
# Opzione 3: Usa dataclass con @dataclass(frozen=True) per immutabilità
```

---

## 3. PANDAS .apply() ANTI-PATTERNS 🟠

### PROBLEMA 3.1: apply() con lambda su intera colonna (LINEA 7170-7172)
```python
df_complete["weight"] = df_complete["days_ago"].apply(
    lambda x: time_decay_weight(x, half_life_days)
)
```

- **Complessità:** O(n) dove n = numero righe
- **Operazione per riga:** function call `time_decay_weight()` + overhead lambda
- **Impatto:** ~1-5ms per 1000 righe (funzione è semplice)
- **Meglio con:** Vectorizzazione NumPy

**FIX SUGGERITO:**
```python
# PRIMA
df_complete["weight"] = df_complete["days_ago"].apply(lambda x: time_decay_weight(x, half_life_days))

# DOPO - Vectorizzare la formula se possibile
# time_decay_weight probabilmente fa: exp(-days * ln(2) / half_life)
decay_factor = np.log(2) / half_life_days
df_complete["weight"] = np.exp(-df_complete["days_ago"] * decay_factor)
# Speedup: 10-100x più veloce
```

### PROBLEMA 3.2: apply() con axis=1 (LINEA 14581-14582)
```python
df_del["label"] = df_del.apply(
    lambda r: f"{r.get('timestamp','?')} – {r.get('match','(no name)')}",
    axis=1,  # ❌ LENTISSIMO! Passa intera riga come Series
)
```

- **Complessità:** O(n) con overhead ALTO (axis=1 passa Series)
- **Impatto:** MOLTO PIÙ LENTO di apply() con axis=0
- **Meglio con:** String operations direttamente su colonne

**FIX SUGGERITO:**
```python
# PRIMA
df_del["label"] = df_del.apply(lambda r: f"{r.get('timestamp','?')} – {r.get('match','(no name)')}", axis=1)

# DOPO - Vectorizzato
df_del["label"] = df_del["timestamp"].fillna("?") + " – " + df_del["match"].fillna("(no name)")
# Speedup: 50-100x per grandi dataset
```

---

## 4. LISTA COMPREHENSION CON FUNCTION CALL (LINEA 6212)

```python
probs = np.array([poisson_pmf(k, lam_float) for k in range(max_k + 1)], dtype=np.float64)
```

- **Complessità:** O(max_k) ≈ O(15)
- **Operazione per item:** `poisson_pmf()` function call (contiene math.exp, factorial lookup)
- **Impatto:** ~0.5-2ms per distribuzione

**Questo NON è male** (max_k=15 è piccolo), ma se callato 1000x diventa 500-2000ms

**FIX SUGGERITO:**
```python
# Se stessa distribuzione usata multiple volte, cache:
@functools.lru_cache(maxsize=128)
def cached_poisson_distribution(lam_float: float, max_k: int):
    return np.array([poisson_pmf(k, lam_float) for k in range(max_k + 1)], dtype=np.float64)
```

---

## 5. MANCANZA DI STREAMLIT CACHING 🔴

### PROBLEMA: Computazioni Heavy senza @st.cache_data

**File:** Frontendcloud.py (app principale Streamlit)

Nessun utilizzo di:
```python
@st.cache_data  # ❌ Nessun caching di dati computati
@st.cache_resource  # ❌ Nessun caching di risorse (API clients, modelli)
```

**Impatto:**
- Ogni interazione Streamlit (slider, tab change, button) **re-esegue l'intero calcolo**
- Tutte le 121 operazioni O(mg²) vengono ricalcolate
- API calls vengono rifatte (se presenti)
- Dataframe processing viene ripetuto

**Esempio di impatto:**
1. Utente seleziona match
2. Streamlit re-runs lo script intero
3. Tutti i calcoli di probabilità vengono rifatti
4. Se 10 operazioni × 100ms = 1000ms lag visibile

**FIX SUGGERITO:**
```python
import streamlit as st

@st.cache_data
def calculate_match_probabilities(match_data: dict) -> dict:
    """Cachea risultati calcoli probabilità"""
    # Tutte le operazioni O(mg²) qui dentro
    return {...}

# Poi nel main:
results = calculate_match_probabilities(match_data)  # Cached se same input
```

---

## 6. PROTEZIONI ECCESSIVE IN LOOP

### PROBLEMA: Type/validation checks ripetuti in ogni iterazione

Tutte le funzioni O(mg²) hanno:
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        p = mat[h][a]
        # 4-5 check per OGNI iterazione
        if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
            continue
```

Con mg=10: **121 iterazioni × 5 check = 605 validazioni**

**FIX SUGGERITO:**
```python
# Validate matrice UNA sola volta all'inizio
def _validate_matrix_once(mat):
    """Convalida matrice una sola volta, ritorna matrice pulita"""
    mat = np.array(mat, dtype=float)
    # Sostituisci invalidi con 0
    mat[~(np.isfinite(mat) & (mat >= 0))] = 0.0
    return mat

# Poi nei loop, accedi direttamente (senza check):
for h in range(mg + 1):
    for a in range(mg + 1):
        p = mat[h][a]  # ✅ Già validato, niente check
```

---

## 7. ASSENZA EARLY TERMINATION

### PROBLEMA: Loop che potrebbero terminare presto

**Linea 8129-8130** (prob_esito_over_from_matrix):
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        if h + a <= soglia:
            continue  # ❌ Fa comunque le validazioni per niente
        # ... calcolo
```

Se soglia=4, per mg=10:
- Prima 15 iterazioni (h=0-2, a=0-5 approssimativamente) → tutte saltate
- Ancora valida (isinstance, NaN check, isfinite)

**FIX SUGGERITO:**
```python
for h in range(mg + 1):
    for a in range(mg + 1):
        if h + a <= soglia:
            continue
        # Validazione QUI (solo se necessario)
        if not isinstance(p, (int, float)):
            continue
```

---

## 8. FILE STRUCTURE - MONOLITIC (non performance ma code smell)

- **Frontendcloud.py:** 16.650 linee (MONOLITO)
- **advanced_features.py:** 778 linee (OK)
- **auto_features.py:** 744 linee (OK)
- **api_manager.py:** 663 linee (OK)

**Impatto:** 
- Difficile capire dove sono i bottleneck
- Import overhead (tutto caricato all'inizio)
- Hot-reload lento

---

## SUMMARY - RANKING IMPACT

### TOP 5 QUICK WINS (Easy to fix, High impact)

| # | Problema | Linee | Severity | Time Estimate | Speedup |
|---|----------|-------|----------|----------------|---------|
| 1 | Rimuovi deepcopy dal cache | 11531, 11537, 11592, 11667, 11686 | CRITICO | 10 min | 10-100x |
| 2 | Apply axis=0 instead axis=1 | 14581 | ALTO | 5 min | 50-100x |
| 3 | Validate matrice once, non in loop | 7916, 8011, 8127 etc | ALTO | 30 min | 5-10x |
| 4 | Aggiungi @st.cache_data | Main flow | MEDIO | 20 min | 2-10x (user perceived) |
| 5 | Vectorizza apply con formula | 7170 | MEDIO | 15 min | 10-100x |

### TOP 3 LONG-TERM IMPROVEMENTS

| # | Miglioramento | Effort | Impact | Priority |
|---|---------------|--------|--------|----------|
| 1 | Convertire O(mg²) loops a NumPy | 4-6 ore | 10-50x su hot paths | ALTO |
| 2 | Split Frontendcloud.py in moduli | 8-12 ore | Code quality, maintainability | MEDIO |
| 3 | LRU cache per poisson_pmf | 30 min | 10-50x se chiamato multiple volte | BASSO |

---

## STIMA TEMPO TOTALE IMPATTO

**Current (no fixes):**
- Match calculation: ~200-500ms (per match)
- UI rerun: ~1000-2000ms (full recalc Streamlit)
- Multi-match analysis (100 match): ~20-50s

**After Quick Wins (#1-5):**
- Match calculation: ~50-100ms (4-10x speedup)
- UI rerun: ~200-500ms (2-5x speedup)
- Multi-match analysis: ~5-10s

**After Long-term (#1):**
- Match calculation: ~10-30ms (10-50x speedup)

