# REPORT EDGE CASES NON GESTITI - Software-AsianOdds

## CRITICITÀ RISCONTRATE

### 1. ARRAY/LIST OPERATIONS - ACCESSO SENZA CHECK LUNGHEZZA

#### 1.1 - Frontendcloud.py:2452
**Problema**: `.split()[0]` su stringa senza validazione se esiste primo elemento
```python
return team_name.split()[0].title()
```
**Edge Case**: Se `team_name.split()` restituisce lista vuota (team_name = "" o solo spazi)
**Impatto**: `IndexError` a runtime
**Fix**: `return (team_name.split()[0] if team_name.split() else "Unknown").title()`

#### 1.2 - Frontendcloud.py:3624
**Problema**: `.split()[-1]` assume che market.split() abbia almeno 1 elemento
```python
threshold = float(market.split()[-1])
```
**Edge Case**: Se `market` è stringa vuota o solo spazi
**Impatto**: `IndexError` o `ValueError` quando non parsabile
**Fix**: Validare lunghezza split prima di accedere all'indice

#### 1.3 - Frontendcloud.py:11833
**Problema**: `.split(",")[0]` assume primo elemento esiste
```python
city = stadium_location.split(",")[0].strip()
```
**Edge Case**: Se `stadium_location` è None o stringa vuota
**Impatto**: `AttributeError` se stadium_location è None
**Fix**: `city = (stadium_location or "").split(",")[0].strip() if stadium_location else ""`

#### 1.4 - api_manager.py:184
**Problema**: `cursor.fetchone()[0]` accessa indice 0 senza check None
```python
count = cursor.fetchone()[0]
```
**Edge Case**: Se query restituisce NULL (malformata o vuota)
**Impatto**: `TypeError: 'NoneType' object is not subscriptable`
**Fix**: `result = cursor.fetchone(); count = result[0] if result else 0`

#### 1.5 - api_manager.py:339
**Problema**: `result[0] if result else 0` ma potrebbe esplodere
```python
return result[0] if result else 0
```
**Edge Case**: Se result è una lista vuota `[]`
**Impatto**: `IndexError: list index out of range`
**Fix**: `return result[0] if result and len(result) > 0 else 0`

#### 1.6 - api_manager.py:451 & Frontendcloud.py:4297, 4778
**Problema**: `data["teams"][0]` senza protezione, `response[0]` senza check
```python
return data["teams"][0]  # Return first match
return response[0]
```
**Edge Case**: Lista `teams` o `response` vuota
**Impatto**: `IndexError: list index out of range`
**Fix**: `return data["teams"][0] if data.get("teams") and len(data["teams"]) > 0 else None`

#### 1.7 - Frontendcloud.py:2136
**Problema**: `value_bets[0]` accesso diretto senza lunghezza
```python
logger.info(f"🔥 Top bet: {value_bets[0]['home_team']}...")
```
**Edge Case**: Se `value_bets` è lista vuota
**Impatto**: `IndexError`
**Fix**: `if value_bets: logger.info(f"🔥 Top bet: {value_bets[0]['home_team']}...")`

#### 1.8 - Frontendcloud.py:2236
**Problema**: `.get('weather', [{}])[0]` accesso non sicuro a indice
```python
weather = closest_forecast.get('weather', [{}])[0]
```
**Edge Case**: Se 'weather' è lista vuota `[]`
**Impatto**: `IndexError`
**Fix**: `weather = (closest_forecast.get('weather') or [{}])[0] if closest_forecast.get('weather') else {}`

#### 1.9 - Frontendcloud.py:4914
**Problema**: `weather_list[0]` senza verifica lunghezza
```python
condition_label = (weather_list[0] or {}).get("description", "").lower()
```
**Edge Case**: Se `weather_list` è vuota
**Impatto**: `IndexError`
**Fix**: `condition_label = ((weather_list[0] if weather_list else {}) or {}).get("description", "").lower()`

#### 1.10 - advanced_features.py:195, 199, 203
**Problema**: `result.x[0]`, `x0[0]` accesso senza lunghezza
```python
return result.x[0], result.x[1]
return apply_physical_constraints_to_lambda(x0[0], x0[1], total_target)
```
**Edge Case**: Se `result.x` ha meno di 2 elementi
**Impatto**: `IndexError`
**Fix**: Validare `len(result.x) >= 2` prima di accedere

---

### 2. VALORI NULLI/EMPTY - GESTIONE INCOMPLETA

#### 2.1 - auto_features.py:343
**Problema**: `.split("-")` su string potenzialmente invalida
```python
start, end = map(int, pos_range.split("-"))
```
**Edge Case**: Se `pos_range` non contiene "-" o non è numerica
**Impatto**: `ValueError: not enough values to unpack` o `ValueError: invalid literal`
**Fix**: Aggiungere try/except e validazione

#### 2.2 - Frontendcloud.py:578-579
**Problema**: `s or ""` ma s potrebbe essere non-string
```python
s = str(s) if s is not None else ""
return (s or "").lower().replace(" ", "").replace("-", "").replace("/", "")
```
**Edge Case**: Accettabile ma meglio aggiungere controllo tipo
**Impatto**: Possibile TypeError se s è oggetto non convertibile
**Fix**: Validare tipo esplicitamente

#### 2.3 - dashboard.py:110
**Problema**: `.strip()` su potentially None string
```python
if not home_team_input.strip() or not away_team_input.strip():
```
**Edge Case**: Se `home_team_input` è None
**Impatto**: `AttributeError: 'NoneType' object has no attribute 'strip'`
**Fix**: `if not (home_team_input or "").strip() or not (away_team_input or "").strip():`

#### 2.4 - Frontendcloud.py:12285, 12286
**Problema**: `.split()` e accesso senza validazione
```python
footer = "\n\n" + sections[-1].split("📈 <b>Riepilogo:</b>")[0]
sections[-1] = sections[-1].split("🤖 <i>Analisi automatica")[0]
```
**Edge Case**: Se sections è vuota, split non torna nulla, indice -1 non esiste
**Impatto**: `IndexError`
**Fix**: Validare `if sections and len(sections) > 0` e accesso con bounds checking

---

### 3. OPERAZIONI MATEMATICHE - LIMITI E SPECIAL VALUES

#### 3.1 - Frontendcloud.py:6195-6204
**Problema**: `math.exp(-lam)` potrebbe generare OverflowError con lambda molto grande
```python
try:
    probs[0] = math.exp(-lam_float)
except OverflowError:
    logger.warning(f"Overflow exp(-lam)...")
    probs[0] = 0.0
```
**Buono**: Ha try/except, ma...
**Edge Case**: Lambda > 700 causa overflow, fallback a 0.0 potrebbe essere scorretto
**Impatto**: Probabilità non normalizzate
**Fix**: `probs[0] = math.exp(-min(lam_float, 100.0))` con limite massimo

#### 3.2 - Frontendcloud.py:1505
**Problema**: `mu1 / mu2` divisione senza protezione
```python
ratio_term = (mu1 / mu2) ** (abs(k) / 2.0)
```
**Edge Case**: Se `mu2 == 0` oppure se ratio_term diventa infinito
**Impatto**: `ZeroDivisionError` o overflow
**Fix**: `mu2_safe = max(mu2, model_config.TOL_DIVISION_ZERO); ratio_term = ...`

#### 3.3 - advanced_features.py:58, 89, 96
**Problema**: `max(total_current, 0.01)` per evitare divisione per zero, MA...
```python
scale = 0.5 / max(total_current, 0.01)
scale = total_target / max(total_current, 0.01)
scale_fix = 0.5 / max(total_after_scale, 0.01)
```
**Edge Case**: Se total_current esattamente 0, uso 0.01 di fallback, potrebbe causare scale enormi
**Impatto**: Lambda o probabilità molto distorte
**Fix**: Aggiungere clamp su scala: `scale = max(0.1, min(10.0, scale))`

#### 3.4 - Frontendcloud.py:1164
**Problema**: `np.sqrt(sqrt_arg)` senza validazione di sqrt_arg
```python
sqrt_term = np.sqrt(sqrt_arg)
```
**Edge Case**: Se `sqrt_arg < 0` (errori numerici precedenti)
**Impatto**: `RuntimeWarning: invalid value encountered in sqrt` → NaN
**Fix**: `sqrt_arg_safe = max(sqrt_arg, 0.0); sqrt_term = np.sqrt(sqrt_arg_safe)`

---

### 4. STRING OPERATIONS - PARSING FRAGILE

#### 4.1 - Frontendcloud.py:13560
**Problema**: `.split()` e parsing senza validazione
```python
total_in_entry = float(parts[1].split()[0])
```
**Edge Case**: Se parts[1].split() restituisce lista vuota
**Impatto**: `IndexError`
**Fix**: Aggiungere bounds check: `if parts[1].split(): total_in_entry = float(...)`

#### 4.2 - Frontendcloud.py:4813-4815
**Problema**: `.strip()` su potentially None city
```python
query_city = city.strip() if city else ""
if country:
    query_city = f"{query_city},{country}".strip(",")
```
**Edge Case**: Se city è None correttamente gestito, ma se country è None?
**Impatto**: Potrebbe generare malformed query
**Fix**: `country = (country or "").strip()`

#### 4.3 - auto_features.py:165, 170, 223, 232
**Problema**: `.lower()` su potentially None
```python
league_lower = league_name.lower()
if name.lower() == league_lower:
```
**Edge Case**: Se league_name è None
**Impatto**: `AttributeError: 'NoneType' object has no attribute 'lower'`
**Fix**: `league_lower = (league_name or "").lower()`

---

### 5. BARE EXCEPT CLAUSES - SILENT FAILURES

#### 5.1-5.7 - Frontendcloud.py:8856, 8989, 8996, 9003, 10670, 11958, 12426
**Problema**: Naked `except:` clause che cattura tutte le eccezioni
```python
except:
    # Fallback: funzione identità
    return lambda p: p, 1.0

except:
    pass

except:
    dt = datetime.now()

except:
    return []
```
**Impatto**: Nasconde KeyboardInterrupt, SystemExit, memoria insufficiente
**Fix**: Specifiche eccezioni: `except (ValueError, RuntimeError, KeyError) as e:`

---

### 6. EMPTY DATAFRAMES - MISSING CHECKS

#### 6.1 - dashboard.py:226
**Problema**: `.cumsum()` su potenziale DataFrame vuoto
```python
if not df_profit.empty:
    df_profit['cumulative_profit'] = df_profit['daily_profit'].cumsum()
```
**Buono**: Ha check, ma...
**Edge Case**: Se df_profit ha 1 sola riga, cumsum() funziona ma semantica è discutibile
**Impatto**: Basso, solo semantico
**Fix**: Aggiungere controllo `if len(df_profit) > 1`

#### 6.2 - advanced_features.py:369-371
**Problema**: `.mean()` su subset potentially vuoto
```python
n_samples = mask.sum()
if n_samples >= 10:
    predicted_mean = df.loc[mask, prob_col].mean()
```
**Buono**: Ha check su n_samples, ma...
**Edge Case**: Se mask è all False (no matching rows)
**Impatto**: Basso, protetto da >= 10
**Fix**: OK come è

#### 6.3 - Frontendcloud.py:3930-3931
**Problema**: `.mean()` su potentially empty DataFrame
```python
home_goals_avg = historical_matches['home_score'].mean()
away_goals_avg = historical_matches['away_score'].mean()
```
**Edge Case**: Se historical_matches è vuoto
**Impatto**: `.mean()` ritorna NaN, propagazione errore downstream
**Fix**: `if len(historical_matches) > 0: home_goals_avg = ... else: home_goals_avg = 1.5`

---

### 7. DATETIME/TIMEZONE - PARSING E FORMATI

#### 7.1 - auto_features.py:383-400
**Problema**: `.fromisoformat()` con format handling
```python
try:
    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
except:
    try:
        dt = datetime.strptime(datetime_str, fmt)
    except ValueError as e:
```
**Edge Case**: `datetime_str = None` - errore silenzioso nella replace
**Impatto**: `AttributeError: 'NoneType' object has no attribute 'replace'`
**Fix**: `if not datetime_str: raise ValueError("datetime_str is None")`

#### 7.2 - Frontendcloud.py:2224
**Problema**: `datetime.fromtimestamp()` con timestamp potenzialmente invalido
```python
forecast_dt = datetime.fromtimestamp(forecast['dt'])
```
**Edge Case**: Se `forecast['dt']` è 0, -1, o troppo grande (timestamp overflow)
**Impatto**: `ValueError: year is out of range` o `OSError`
**Fix**: `try: forecast_dt = datetime.fromtimestamp(int(forecast['dt'])) except: forecast_dt = datetime.now()`

---

### 8. FILE I/O E ERRORI

#### 8.1 - auto_features.py:78-79
**Problema**: `json.load()` senza validazione file
```python
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
```
**Edge Case**: File corrotto, JSON malformato, permessi insufficienti, disco pieno
**Impatto**: `json.JSONDecodeError`, `IOError`, `UnicodeDecodeError`
**Fix**: Aggiungere try/except specifiche

#### 8.2 - advanced_features.py:330
**Problema**: `pd.read_csv()` senza protezione
```python
df = pd.read_csv(csv_path)
```
**Edge Case**: File non esiste, permessi insufficienti, CSV corrotto
**Impatto**: `FileNotFoundError`, `PermissionError`, `pd.errors.ParserError`
**Fix**: Aggiungere try/except

#### 8.3 - api_manager.py:376-377, 410-411, 437-438
**Problema**: Network requests senza timeout globale su alcuni
```python
with urllib.request.urlopen(req, timeout=10) as response:
    data = json.loads(response.read().decode())
```
**Edge Case**: Timeout non aggiunto ovunque, network down, charset non UTF-8
**Impatto**: `urllib.error.URLError`, `UnicodeDecodeError`
**Fix**: Standardizzare timeout e aggiungere fallback su charset

---

### 9. DIVISIONE PER ZERO - EDGE CASES SFUMATI

#### 9.1 - advanced_features.py:761-762
**Problema**: Protezione di `max(lambda_h_start, 1e-9)` è buona ma...
```python
lambda_h_change = ((lambda_h - lambda_h_start) / max(lambda_h_start, 1e-9)) * 100 if lambda_h_start != 0 else 0.0
```
**Edge Case**: Se lambda_h_start è esattamente 0, ritorna 0.0, ma se è 1e-12 (tiny), ritorna cambio enorme
**Impatto**: Statistiche distorte
**Fix**: `epsilon = 0.1; ... / max(abs(lambda_h_start), epsilon)`

#### 9.2 - Frontendcloud.py:1275-1276, 1286-1287
**Problema**: `probs.sum()` potrebbe ritornare 0.0 dopo normalizzazione
```python
sum_probs = probs.sum()
if sum_probs > model_config.TOL_DIVISION_ZERO:
    fair_probs = probs / sum_probs
```
**Edge Case**: Se sum_probs esattamente 0.0 (tutti NaN), ritorna uniforme
**Impatto**: Fallback non sempre corretto
**Fix**: Aggiungere validazione che result sia finito: `if np.all(np.isfinite(fair_probs))`

---

### 10. ASSUNZIONI HARDCODED

#### 10.1 - Frontendcloud.py:4484
**Problema**: Assumere 18 squadre in campionato
```python
# Calcola distanza da salvezza (assumendo 18 squadre, 3 retrocedono)
```
**Edge Case**: Alcuni campionati hanno 20 squadre (Premier League), 16 (alcuni campionati)
**Impatto**: Calcoli di motivazione scorretti per campionati diversi
**Fix**: Parametrizzare numero squadre per campionato

#### 10.2 - Frontendcloud.py:3949
**Problema**: Assumere almeno 3 risultati storici senza validare
```python
if len(historical_matches) > 0:
    home_goals_avg = historical_matches['home_score'].mean()
```
**Edge Case**: Se storico ha 1 sola partita, media non rappresentativa
**Impatto**: Probabilità basate su campione troppo piccolo
**Fix**: `if len(historical_matches) >= 5:` e altrimenti usare prior

#### 10.3 - api_manager.py:301
**Problema**: Assunzione su rate limit
```python
# Check per-minute quota (simplified: assume 1 call/6 seconds)
```
**Edge Case**: Endpoint vari hanno rate limit diversi
**Impatto**: Possibili rate limit violation
**Fix**: Implementare rate limiter specifico per endpoint

---

## RIEPILOGO CRITICITÀ

| Tipo | Conteggio | Severità |
|------|-----------|----------|
| Array access senza check | 10 | ALTA |
| Bare except clauses | 7 | MEDIA |
| String split senza validazione | 5 | MEDIA |
| Divisione per zero (sfumata) | 2 | MEDIA |
| File I/O senza protezione | 3 | MEDIA |
| DateTime parsing fragile | 2 | BASSA |
| DataFrame operations su vuoto | 2 | MEDIA |
| Assunzioni hardcoded | 3 | MEDIA |

**TOTALE PROBLEMI TROVATI: 34**

---

## PATTERN PERICOLOSI IDENTIFICATI

1. ❌ **`list[0]` senza `len(list) > 0`** - 10 occorrenze
2. ❌ **`except:` naked** - 7 occorrenze  
3. ❌ **`.split()[index]` senza bounds** - 5 occorrenze
4. ❌ **`/ max(x, epsilon)` con epsilon piccolo** - 2 occorrenze
5. ❌ **File operations senza try/except** - 3 occorrenze
6. ❌ **`.mean()` su DataFrame potenzialmente vuoto** - 2 occorrenze
7. ❌ **Hardcoded assumptions (18 squadre, 3 retrocessi)** - 1 occorrenza

---

## RACCOMANDAZIONI

1. **Creare utility function per safe list access**:
   ```python
   def safe_get_index(lst, index, default=None):
       return lst[index] if lst and len(lst) > index else default
   ```

2. **Unificare gestione None su string operations**:
   ```python
   def safe_lower(s):
       return (s or "").lower()
   ```

3. **Sostituire bare except con specifiche**:
   ```python
   except (ValueError, RuntimeError, KeyError) as e:
       logger.exception(f"Errore specifico: {e}")
   ```

4. **Aggiungere pre-validation su input**:
   - `if not isinstance(x, (int, float)): raise TypeError(...)`

5. **Usare Decimal per operazioni monetarie** (se applicabile)

6. **Implementare schema validation** con pydantic/marshmallow

