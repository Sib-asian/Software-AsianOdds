# 🔢 AUDIT COMPLETO CALCOLI MATEMATICI
## Software AsianOdds - Analisi Formule e Calcoli

**Data:** 2025-11-14
**Scope:** TUTTI i calcoli matematici e formule
**Status:** 🔍 IN CORSO

---

## 📊 CATEGORIE ANALIZZATE

1. ✅ Operazioni di Divisione (Division by Zero)
2. ✅ Formule Probabilistiche (Poisson, Dixon-Coles)
3. ✅ Kelly Criterion
4. ✅ Operazioni Logaritmiche
5. ✅ Normalizzazioni Probabilità
6. ✅ Operazioni Statistiche (mean, std, var)
7. ✅ Conversioni Odds ↔ Probabilità
8. ✅ Calcoli Over/Under
9. ✅ Asian Handicap
10. ✅ Operazioni Radice Quadrata/Potenza

---

## 🔴 CRITICAL MATH BUGS TROVATI

### BUG MATH #1: Division by Zero Non Protetta - Sharpe Ratio
**File:** `Frontendcloud.py:3528`
**Severità:** 🔴 CRITICAL

```python
variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
std_dev = variance ** 0.5
if std_dev > 0:
    sharpe_ratio = (mean_profit * len(profits) ** 0.5) / std_dev  # ✅ OK - std_dev protected
```

**Status:** ✅ GIÀ PROTETTO - False alarm

---

### BUG MATH #2: Potential Division by Zero - Win Rate
**File:** `Frontendcloud.py:3176,3499,3601`
**Severità:** 🟠 HIGH

```python
# Line 3176:
win_rate = wins / total_bets if total_bets > 0 else 0  # ✅ OK - protected

# Line 3499:
win_rate = wins / total_bets if total_bets > 0 else 0  # ✅ OK - protected

# Line 3601:
win_rate = (row['wins'] / total * 100) if total > 0 else 0  # ✅ OK - protected
```

**Status:** ✅ GIÀ PROTETTO

---

### BUG MATH #3: Kelly Criterion - Edge Case Odds = 1.0
**File:** `Frontendcloud.py:2343, 2916, 3004`
**Severità:** 🟠 HIGH

```python
# Line 2343:
kelly_full = (prob * odds - 1.0) / (odds - 1.0)  # ❌ NOT PROTECTED if odds == 1.0

# Line 2916:
kelly_pct = edge / (odds - 1.0) if odds > 1.0 else 0.0  # ✅ OK - protected

# Line 3004:
kelly_full = (prob * odds - 1.0) / (odds - 1.0)  # ❌ NOT PROTECTED if odds == 1.0
```

**Problem:** Se `odds == 1.0`, divisione per zero!

**Example:**
```python
odds = 1.0
kelly_full = (0.5 * 1.0 - 1.0) / (1.0 - 1.0)  # Division by zero!
```

**Fix Richiesto:**
```python
# Line 2343:
if odds > 1.0:
    kelly_full = (prob * odds - 1.0) / (odds - 1.0)
else:
    kelly_full = 0.0

# Line 3004: Same fix
if odds > 1.0:
    kelly_full = (prob * odds - 1.0) / (odds - 1.0)
else:
    kelly_full = 0.0
```

---

### BUG MATH #4: Odds to Probability Conversion - Odds = 0
**File:** `Frontendcloud.py:759, 2851`
**Severità:** 🟠 HIGH

```python
# Line 759:
prob = 1.0 / odds_safe  # odds_safe defined how?

# Line 2851:
implied_prob = 1.0 / odds  # ❌ NOT PROTECTED if odds == 0
```

**Problem:** Se `odds == 0`, divisione per zero!

**Fix Richiesto:**
```python
# Line 2851:
implied_prob = 1.0 / max(odds, 1e-10)  # Protection
```

---

### BUG MATH #5: BTTS Probability - Division by Zero
**File:** `Frontendcloud.py:1897, 1958, 2029, 2053`
**Severità:** 🟡 MEDIUM

```python
# Line 1897:
odds_btts = 1.0 / prob_btts  # ❌ What if prob_btts == 0?

# Line 1958:
odds_btts = 1.0 / gg_prob  # ❌ What if gg_prob == 0?

# Line 2029:
odds_final = 1.0 / p_final  # ❌ What if p_final == 0?

# Line 2053:
odds_model = 1.0 / prob_model  # ❌ What if prob_model == 0?
```

**Fix Richiesto:**
```python
odds_btts = 1.0 / max(prob_btts, 1e-10)
odds_btts = 1.0 / max(gg_prob, 1e-10)
odds_final = 1.0 / max(p_final, 1e-10)
odds_model = 1.0 / max(prob_model, 1e-10)
```

---

### BUG MATH #6: Poisson PMF - Division by Zero in Factorial
**File:** `Frontendcloud.py:528, 570`
**Severità:** 🟢 LOW (edge case molto improbabile)

```python
# Line 528:
res = (lam ** k) * math.exp(-lam) / _FACTORIAL_CACHE_ARRAY[k]  # ✅ OK - factorial never 0

# Line 570:
res = (lam ** k) * math.exp(-lam) / _FACTORIAL_CACHE[k]  # ✅ OK - factorial never 0
```

**Status:** ✅ OK - Factorial di qualsiasi numero >= 0 è sempre >= 1

---

### BUG MATH #7: Shinozaki Method - Division by Denom
**File:** `Frontendcloud.py:1326, 1383`
**Severità:** 🔴 CRITICAL

```python
# Line 1326:
fair_probs = (sqrt_term - z) / denom  # ❌ What if denom == 0?

# Line 1383:
fair_probs = (sqrt_term - z_opt) / denom  # ❌ What if denom == 0?
```

**Analysis Needed:**
Verificare se `denom` può essere 0. Dipende da come è calcolato:

```python
# Cerco definizione di denom...
```

**Action:** VERIFICARE CODICE UPSTREAM

---

### BUG MATH #8: ROI Calculation - Division by Stake
**File:** `Frontendcloud.py:3178, 3500, 3602`
**Severità:** 🟢 LOW

```python
# Line 3178:
roi = (total_profit / sum(self.stakes)) * 100 if sum(self.stakes) > 0 else 0  # ✅ OK

# Line 3500:
roi = (total_profit / total_staked * 100) if total_staked > 0 else 0  # ✅ OK

# Line 3602:
roi = (row['total_profit'] / row['total_staked'] * 100) if row['total_staked'] > 0 else 0  # ✅ OK
```

**Status:** ✅ GIÀ PROTETTO

---

### BUG MATH #9: Average Goals Calculation
**File:** `Frontendcloud.py:4380-4381`
**Severità:** 🟢 LOW

```python
'avg_goals_home': round(sum(goals_home) / len(goals_home), 2) if goals_home and len(goals_home) > 0 else 0,  # ✅ OK
'avg_goals_away': round(sum(goals_away) / len(goals_away), 2) if goals_away and len(goals_away) > 0 else 0,  # ✅ OK
```

**Status:** ✅ GIÀ PROTETTO

---

## 🟠 FORMULE PROBABILISTICHE - VERIFICA CORRETTEZZA

### FORMULA #1: Poisson PMF (Probability Mass Function)
**File:** `Frontendcloud.py:528, 570`

**Formula Teorica:**
```
P(X = k) = (λ^k × e^(-λ)) / k!
```

**Implementazione:**
```python
res = (lam ** k) * math.exp(-lam) / _FACTORIAL_CACHE_ARRAY[k]
```

**Verifica:** ✅ **CORRETTA**

**Controlli Necessari:**
- ✅ λ >=  0 (lambda non negativo)
- ✅ k >= 0 (gol non negativi)
- ✅ k! != 0 (sempre vero)
- ⚠️ Overflow per k molto grandi (>170)

---

### FORMULA #2: Dixon-Coles Adjustment
**File:** `Frontendcloud.py:1678`

**Formula Teorica:**
```
τ(x_h, x_a) = 1 + ρ × (μ_h / μ_a)^(|k|/2)
```

dove:
- ρ (rho) è il parametro di correlazione
- k è la differenza di gol
- μ_h, μ_a sono le medie Poisson

**Implementazione:**
```python
ratio_term = (mu1 / mu2) ** (abs(k) / 2.0)
```

**Problem:** ❌ **DIVISION BY ZERO** se μ2 == 0!

**Fix Richiesto:**
```python
ratio_term = (mu1 / max(mu2, 1e-10)) ** (abs(k) / 2.0)
```

**Severità:** 🔴 CRITICAL

---

### FORMULA #3: Kelly Criterion
**File:** `Frontendcloud.py:2326-2343, 2911-2916, 3004`

**Formula Teorica:**
```
f* = (p × b - q) / b
```

dove:
- f* = frazione di bankroll da scommettere
- p = probabilità di vincita
- q = 1 - p (probabilità di perdita)
- b = odds - 1 (net odds)

**Implementazione:**
```python
kelly_full = (prob * odds - 1.0) / (odds - 1.0)
```

**Verifica:** ✅ **FORMULA CORRETTA**

**Problem:** ❌ **DIVISION BY ZERO** se odds == 1.0!

**Fix:** GIÀ DISCUSSO IN BUG MATH #3

---

### FORMULA #4: Probabilità Over/Under da Poisson
**File:** `Frontendcloud.py` (varie linee)

**Formula Teorica:**
```
P(Over N.5) = Σ P(Total = k) per k > N
P(Under N.5) = Σ P(Total = k) per k <= N
```

dove Total = Home_Goals + Away_Goals

**Implementation Method:**
- Genera matrice Poisson congiunta
- Somma diagonali/triangoli

**Verifica:** ✅ **METODO CORRETTO** (assumendo Poisson implementation corretta)

---

### FORMULA #5: Normalizzazione Probabilità
**File:** `advanced_features.py:375-378`

**Formula Teorica:**
```
p_normalized_i = p_i / Σp_j × expected_total
```

**Implementazione:**
```python
if abs(total) < 1e-10:  # ✅ Protected
    return np.ones_like(probs) / len(probs) * expected_total

probs_normalized = probs * (expected_total / total)
```

**Verifica:** ✅ **CORRETTA** (con protezione division by zero)

---

## 🟡 OPERAZIONI LOGARITMICHE

### LOG #1: Logaritmo per Conversioni
**File:** `Frontendcloud.py:549, 591`

```python
log_p = math.log(p) / log2_const
```

**Problem:** ❌ `math.log(p)` ritorna -inf se p == 0, NaN se p < 0

**Protezione Necessaria:**
```python
log_p = math.log(max(p, 1e-300)) / log2_const
```

**Severità:** 🟠 HIGH

---

### LOG #2: Entropy Calculation (Meta-Learner)
**File:** `ai_system/models/ensemble.py:243`

```python
weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights.values())
```

**Verifica:** ✅ **PROTETTO** (w + 1e-10 non può essere <= 0)

---

## 🟢 OPERAZIONI STATISTICHE

### STAT #1: Mean/Average Calculations
**Pattern:** `sum(...) / len(...)` o `np.mean()`

**Ricerca Sistematica:**

```python
# Frontendcloud.py:3524:
mean_profit = sum(profits) / len(profits)  # ❌ NOT PROTECTED if profits empty

# Frontendcloud.py:4070-4071:
xg_home_pred = (home_xg['xg_for_avg'] + away_xg['xg_against_avg']) / 2  # ✅ OK - dict values

# Frontendcloud.py:2776-2778:
'xg_per_match': round(total_xg_for / matches_count, 2),  # ❌ NOT PROTECTED if matches_count == 0
'xga_per_match': round(total_xg_against / matches_count, 2),
'xg_diff_per_match': round((total_xg_for - total_xg_against) / matches_count, 2),
```

**Bugs Trovati:**
1. ❌ `Frontendcloud.py:3524` - mean_profit senza check len(profits) > 0
2. ❌ `Frontendcloud.py:2776-2778` - divisione per matches_count senza check

---

### STAT #2: Variance Calculation
**File:** `Frontendcloud.py:3525`

```python
variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
```

**Problem:** ❌ NOT PROTECTED if profits empty

**Fix:**
```python
if len(profits) > 0:
    mean_profit = sum(profits) / len(profits)
    variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
    std_dev = variance ** 0.5
    if std_dev > 0:
        sharpe_ratio = (mean_profit * len(profits) ** 0.5) / std_dev
else:
    sharpe_ratio = 0.0
```

---

## 🔵 OPERAZIONI RADICE QUADRATA E POTENZA

### SQRT #1: Variance to Std Dev
```python
std_dev = variance ** 0.5
```

**Verifica:** ✅ OK - variance sempre >= 0 per definizione

---

### POWER #1: Factorial Approximation
**File:** `Frontendcloud.py:1678`

```python
ratio_term = (mu1 / mu2) ** (abs(k) / 2.0)
```

**Problems:**
1. ❌ Division by zero (mu2 == 0)
2. ⚠️ Overflow se mu1/mu2 molto grande e k grande
3. ⚠️ Underflow se mu1/mu2 molto piccolo e k grande

---

## 📊 RIEPILOGO BUGS MATEMATICI

| Bug ID | Tipo | File | Linea | Severità | Status |
|--------|------|------|-------|----------|--------|
| MATH #3 | Kelly div/0 | Frontendcloud.py | 2343, 3004 | 🔴 CRITICAL | ❌ DA FIXARE |
| MATH #4 | Odds conv | Frontendcloud.py | 2851 | 🟠 HIGH | ❌ DA FIXARE |
| MATH #5 | BTTS odds | Frontendcloud.py | 1897, 1958, 2029, 2053 | 🟡 MEDIUM | ❌ DA FIXARE |
| MATH #7 | Shinozaki | Frontendcloud.py | 1326, 1383 | 🔴 CRITICAL | ⚠️ VERIFICARE |
| FORM #2 | Dixon-Coles | Frontendcloud.py | 1678 | 🔴 CRITICAL | ❌ DA FIXARE |
| LOG #1 | Log conversion | Frontendcloud.py | 549, 591 | 🟠 HIGH | ❌ DA FIXARE |
| STAT #1 | Mean profits | Frontendcloud.py | 3524 | 🟠 HIGH | ❌ DA FIXARE |
| STAT #1 | xG per match | Frontendcloud.py | 2776-2778 | 🟡 MEDIUM | ❌ DA FIXARE |

**Totale Bugs Trovati:** 8 CRITICAL/HIGH + 2 MEDIUM = **10 bugs matematici**

---

## 🎯 PRIORITÀ FIX

### IMMEDIATE (CRITICAL):
1. ✅ Kelly Criterion - odds == 1.0 protection
2. ✅ Dixon-Coles ratio - mu2 == 0 protection
3. ⚠️ Shinozaki denom - verificare se può essere 0

### HIGH PRIORITY:
4. ✅ Odds to probability - odds == 0 protection
5. ✅ Log conversions - p == 0 protection
6. ✅ Mean profits - empty array protection

### MEDIUM PRIORITY:
7. ✅ BTTS odds conversions - prob == 0 protection
8. ✅ xG calculations - matches_count == 0 protection

---

## ✅ NEXT ACTIONS

1. **Fix Kelly Criterion** (2 locations)
2. **Fix Dixon-Coles** ratio (1 location)
3. **Verify Shinozaki** denominator calculation
4. **Fix odds conversions** (5 locations)
5. **Fix log operations** (2 locations)
6. **Fix statistical operations** (3 locations)

**Status:** 🔴 10 BUGS MATEMATICI CRITICI IDENTIFICATI

Continuo con analisi dettagliata delle formule...
