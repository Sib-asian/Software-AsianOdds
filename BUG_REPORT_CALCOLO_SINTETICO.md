# 🐛 Bug Report - Sistema Calcolo Sintetico Quote

## ❌ BUG CRITICI TROVATI

### 1. **RICORSIONE INFINITA POTENZIALE** (CRITICO)

**Location**: `live_betting_advisor.py:557`

**Codice problematico**:
```python
elif 'btts_no' in market_lower:
    # BTTS No = opposto di BTTS Yes
    btts_yes_odds = self._calculate_synthetic_odds(match_data, 'btts_yes', live_data)
```

**Problema**:
- `btts_no` chiama ricorsivamente `_calculate_synthetic_odds('btts_yes')`
- Se la chiamata ricorsiva fallisce o torna in loop, possibile stack overflow
- Non c'è protezione contro ricorsione infinita

**Impatto**: ⚠️ ALTO - Possibile crash applicazione

**Fix**:
Calcolare inline senza ricorsione:
```python
elif 'btts_no' in market_lower:
    if odds_over_1_5 and odds_1 and odds_2:
        # Calcola BTTS Yes inline (no ricorsione)
        prob_over_1_5 = 1 / odds_over_1_5 if odds_over_1_5 > 1.01 else 0
        balance = min(odds_1 / odds_2, odds_2 / odds_1) if odds_1 > 0 and odds_2 > 0 else 0
        prob_btts_yes = prob_over_1_5 * (0.5 + balance * 0.4)

        # Aggiustamenti live
        if live_data and score_home > 0 and score_away > 0:
            prob_btts_yes = 0.99  # Non 1.0 (odds invalide)
        elif live_data and minute > 60 and (score_home == 0 or score_away == 0):
            prob_btts_yes *= 0.7

        # BTTS No = opposto
        prob_btts_no = 1 - prob_btts_yes
        if prob_btts_no > 0.15:
            odds_btts_no = 1 / prob_btts_no
            if 1.1 <= odds_btts_no <= 5.0:
                return round(odds_btts_no, 2)
```

---

### 2. **WIN TO NIL - NULL REFERENCE** (CRITICO)

**Location**: `live_betting_advisor.py:611-622`

**Codice problematico**:
```python
elif 'win_to_nil_home' in market_lower:
    if odds_1:
        # Calcola clean sheet home
        odds_clean = self._calculate_synthetic_odds(match_data, 'clean_sheet_home', live_data)
        if odds_clean:  # Controlla se esiste
            prob_win = 1 / odds_1
            prob_clean = 1 / odds_clean  # OK se odds_clean non è None
            prob_win_to_nil = prob_win * prob_clean
```

**Problema**:
- Se `_calculate_synthetic_odds('clean_sheet_home')` ritorna `None`, ok (gestito con `if odds_clean`)
- Ma se ritorna `0` o valore invalido, divisione problematica: `1 / odds_clean`

**Impatto**: ⚠️ MEDIO - Possibile divisione per zero

**Fix**:
```python
if odds_clean and odds_clean > 1.01:  # Verifica > 1.01
    prob_win = 1 / odds_1
    prob_clean = 1 / odds_clean
```

---

### 3. **BTTS YES - ODDS INVALIDE (1.0)** (MEDIO)

**Location**: `live_betting_advisor.py:542-543`

**Codice problematico**:
```python
# Aggiustamento live: se già entrambe segnato, BTTS = 100%
if live_data and score_home > 0 and score_away > 0:
    prob_btts = 1.0  # BUG: questo porta a odds = 1.0 (invalide!)
```

**Problema**:
- Se `prob_btts = 1.0`, poi `odds_btts = 1 / 1.0 = 1.0`
- Quote 1.0 sono invalide (non si può vincere nulla)
- Minimo odds valide = 1.01

**Impatto**: ⚠️ MEDIO - Quote invalide calcolate

**Fix**:
```python
if live_data and score_home > 0 and score_away > 0:
    prob_btts = 0.99  # Max 99% → odds = 1.01 (minimo valido)
```

---

### 4. **DIVISION BY ZERO - TOTAL_XG** (MEDIO)

**Location**: `live_betting_advisor.py:504-507`

**Codice problematico**:
```python
total_xg = xg_home_weighted + xg_away_weighted
if total_xg > 0.1:  # Protezione parziale
    prob_home = xg_home_weighted / total_xg
    prob_away = xg_away_weighted / total_xg
```

**Problema**:
- Protezione con `total_xg > 0.1` funziona
- Ma se `total_xg == 0.1` esattamente, divisione per valore molto piccolo → odds enormi

**Impatto**: ⚠️ BASSO - Edge case raro

**Fix**: Aumentare soglia minima
```python
if total_xg > 0.5:  # Richiedi attività minima più alta
```

---

### 5. **EXACT SCORE - REGEX FAILURE** (MEDIO)

**Location**: `live_betting_advisor.py:643-647`

**Codice problematico**:
```python
score_match = re.search(r'(\d+)[_-](\d+)', market_lower)
if score_match and odds_1 and odds_x and odds_2:
    expected_home = int(score_match.group(1))  # OK se score_match esiste
    expected_away = int(score_match.group(2))
```

**Problema**:
- Se regex fallisce (formato mercato inatteso), `score_match = None`
- Il check `if score_match` protegge, MA...
- Se format è tipo "exact_score_home_2" (solo un numero), regex matcha ma group(2) non esiste → IndexError

**Impatto**: ⚠️ BASSO - Solo se mercati con naming strano

**Fix**: Validazione più robusta
```python
score_match = re.search(r'(\d+)[_-](\d+)', market_lower)
if score_match and len(score_match.groups()) >= 2 and odds_1 and odds_x and odds_2:
    expected_home = int(score_match.group(1))
    expected_away = int(score_match.group(2))
```

---

## ⚠️ PROBLEMI DI PERFORMANCE

### 6. **CALCOLI RIDONDANTI**

**Location**: Multiple locations

**Problema**:
```python
# Calcolato più volte in diversi punti
possession_away = 100 - possession_home  # Ripetuto 5+ volte
```

**Impatto**: 🔵 BASSO - Spreco CPU minimo

**Fix**: Calcolare una volta e riusare
```python
# All'inizio della funzione
if live_data:
    possession_away = 100 - possession_home
```

---

### 7. **MANCANZA CACHE**

**Problema**:
Se `_get_real_odds` viene chiamato multiple volte per stesso mercato, ricalcola tutto

**Impatto**: 🔵 BASSO - Ma su 100+ mercati si accumula

**Fix**: Aggiungere cache semplice
```python
@lru_cache(maxsize=128)
def _calculate_synthetic_odds_cached(self, match_data_hash, market, live_data_hash):
    # Calcola e ritorna
```

---

## 🔧 EDGE CASES NON GESTITI

### 8. **CLEAN SHEET - SCORE NEGATIVO**

**Problema**: Se per qualche bug `score_away < 0`, check fallisce

**Fix**:
```python
if score_away > 0 or score_away < 0:  # Valida anche negativi (bug upstream)
    return 50.0
```

Meglio:
```python
if score_away != 0:  # Più robusto
    return 50.0
```

---

### 9. **ODDS NEGATIVE O NaN**

**Problema**: Nessun controllo se odds_1, odds_2, odds_x sono NaN o negative

**Fix**: Validazione globale all'inizio
```python
def _calculate_synthetic_odds(self, match_data, market, live_data):
    try:
        odds_1 = match_data.get('odds_1')
        odds_x = match_data.get('odds_x')
        odds_2 = match_data.get('odds_2')

        # Validazione globale
        for odd_name, odd_value in [('odds_1', odds_1), ('odds_x', odds_x), ('odds_2', odds_2)]:
            if odd_value is not None:
                if math.isnan(odd_value) or math.isinf(odd_value) or odd_value <= 0:
                    logger.warning(f"⚠️ {odd_name} invalido: {odd_value}, ignoro")
                    if odd_name == 'odds_1':
                        odds_1 = None
                    elif odd_name == 'odds_x':
                        odds_x = None
                    elif odd_name == 'odds_2':
                        odds_2 = None
```

---

### 10. **POSSESSO > 100% O < 0%**

**Problema**: Se API dà possesso errato (es. 120%), calcoli sballati

**Fix**: Normalizzazione
```python
if live_data:
    possession_home = max(0, min(100, live_data.get('possession_home', 50)))
```

---

## 📊 MIGLIORAMENTI LOGICI

### 11. **BTTS - FORMULA MIGLIORABILE**

**Attuale**:
```python
prob_btts = prob_over_1_5 * (0.5 + balance * 0.4)
```

**Problema**: Non considera che in partite con molti gol (Over 3.5+), BTTS molto più probabile

**Miglioramento**:
```python
# Se Over 2.5 probabile, BTTS ancora più probabile
if odds_over_2_5 and odds_over_2_5 < 2.0:  # Over 2.5 molto probabile
    prob_btts = prob_over_1_5 * (0.6 + balance * 0.35)  # Aumenta probabilità
else:
    prob_btts = prob_over_1_5 * (0.5 + balance * 0.4)
```

---

### 12. **NEXT GOAL - NON CONSIDERA MOMENTUM**

**Attuale**: Usa solo xG corrente

**Miglioramento**: Considera anche trend recente
```python
# Aggiungi peso per ultimi 15' di gioco
if minute >= 15:
    # Se home ha più tiri negli ultimi 15', aumenta xG
    recent_shots_home = live_data.get('shots_last_15min_home', 0)
    recent_shots_away = live_data.get('shots_last_15min_away', 0)

    momentum_factor_home = 1 + (recent_shots_home - recent_shots_away) * 0.05
    xg_home_weighted *= max(0.8, min(1.2, momentum_factor_home))
```

---

### 13. **CLEAN SHEET - CONSIDERA SOLO TIRI, NON XG**

**Attuale**:
```python
shots_factor = max(0.3, 1 - (shots_on_target_away * 0.1))
```

**Miglioramento**: Usa anche dangerous attacks
```python
# Più granulare
threat_level = (shots_on_target_away * 0.15) + (dangerous_attacks_away * 0.02)
shots_factor = max(0.2, 1 - threat_level)
```

---

## 🎯 VALIDAZIONI MANCANTI

### 14. **MANCA VALIDAZIONE INPUT LIVE_DATA**

**Problema**: Se `live_data` ha chiavi mancanti o valori None, fallisce silenziosamente

**Fix**: Validazione all'inizio
```python
if live_data:
    # Normalizza tutti i valori con defaults sicuri
    score_home = int(live_data.get('score_home') or 0)
    score_away = int(live_data.get('score_away') or 0)
    minute = int(live_data.get('minute') or 0)
    possession_home = float(live_data.get('possession_home') or 50)
    # ... etc

    # Validazione range
    minute = max(0, min(120, minute))  # 0-120 minuti
    possession_home = max(0, min(100, possession_home))  # 0-100%
```

---

## 🔥 RIEPILOGO PRIORITÀ

| Bug | Severità | Probabilità | Priorità Fix |
|-----|----------|-------------|--------------|
| **#1 Ricorsione BTTS** | 🔴 ALTA | 🟡 MEDIA | **URGENTE** |
| **#2 Win to Nil NULL** | 🟠 MEDIA | 🟡 MEDIA | **ALTA** |
| **#3 BTTS odds 1.0** | 🟠 MEDIA | 🟢 BASSA | **MEDIA** |
| **#4 Division total_xg** | 🟡 BASSA | 🟢 BASSA | BASSA |
| **#5 Regex exact score** | 🟡 BASSA | 🟢 BASSA | BASSA |
| **#6-7 Performance** | 🔵 MINIMA | 🟢 BASSA | OPZIONALE |
| **#8-10 Edge cases** | 🟡 BASSA | 🟢 BASSA | MEDIA |
| **#11-13 Miglioramenti** | 🔵 ENHANCEMENT | - | OPZIONALE |
| **#14 Validazione input** | 🟠 MEDIA | 🟡 MEDIA | **ALTA** |

---

## 🚀 PIANO DI FIX

### Fase 1 - Bug Critici (URGENTE)
1. ✅ Fix ricorsione BTTS (calcolo inline)
2. ✅ Fix Win to Nil null check
3. ✅ Fix BTTS odds 1.0 → 0.99

### Fase 2 - Validazioni (ALTA)
4. ✅ Validazione input live_data
5. ✅ Validazione odds NaN/negative
6. ✅ Normalizzazione possesso/minuto

### Fase 3 - Edge Cases (MEDIA)
7. ✅ Clean sheet score check robusto
8. ✅ Total_xg soglia più alta
9. ✅ Exact score regex validation

### Fase 4 - Miglioramenti (OPZIONALE)
10. ⏳ BTTS formula migliorata
11. ⏳ Next goal momentum
12. ⏳ Clean sheet threat level
13. ⏳ Performance cache

---

Procedo con implementazione fix?
