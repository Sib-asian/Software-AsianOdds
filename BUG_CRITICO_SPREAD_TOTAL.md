# 🚨 BUG CRITICO: Spread e Total NON Vengono Rispettati

## Data: 2025-11-13
## Severità: **CRITICA** 🔴

---

## Problema Identificato

Quando l'utente fornisce `spread_apertura` e `total_apertura` (o corrente), la formula attuale **NON mantiene esattamente questi valori**.

### Test Case

**Input utente**:
- `spread_apertura = 0.75`
- `total_apertura = 2.5`

**Output calcolato** (Frontendcloud.py:10490-10526):
- `lambda_h = 2.0737`
- `lambda_a = 0.7535`
- **spread calcolato = lambda_h - lambda_a = 1.32** ❌ (dovrebbe essere 0.75)
- **total calcolato = lambda_h + lambda_a = 2.83** ❌ (dovrebbe essere 2.5)

### Errore

- Errore spread: **+76%** (1.32 vs 0.75)
- Errore total: **+13%** (2.83 vs 2.5)

Questo è **INACCETTABILE** perché viola i vincoli espliciti dell'utente.

---

## Causa del Bug

### Formula Attuale (ERRATA)

File: `Frontendcloud.py:10490-10526`

```python
lambda_total_ap = total_apertura_safe / 2.0

spread_clamped = max(-2.0, min(2.0, spread_apertura_safe))
spread_factor_ap = math.exp(spread_clamped * 0.5)

sqrt_ha = math.sqrt(home_advantage)

lambda_h_ap = lambda_total_ap * spread_factor_ap * sqrt_ha
lambda_a_ap = lambda_total_ap / spread_factor_ap / sqrt_ha
```

**Problema**: Questa formula introduce `home_advantage` e `exp()` che **alterano** i valori di spread e total forniti dall'utente.

### Verifica Matematica

Con `spread = 0.75`, `total = 2.5`, `home_advantage = 1.3`:

```
lambda_total_ap = 2.5 / 2 = 1.25
spread_factor_ap = exp(0.75 * 0.5) = exp(0.375) = 1.455
sqrt_ha = sqrt(1.3) = 1.140

lambda_h_ap = 1.25 * 1.455 * 1.140 = 2.074
lambda_a_ap = 1.25 / 1.455 / 1.140 = 0.753

spread_calc = 2.074 - 0.753 = 1.321 ≠ 0.75 ❌
total_calc = 2.074 + 0.753 = 2.827 ≠ 2.5 ❌
```

La formula **NON preserva** i vincoli matematici base.

---

## Soluzione

### Formula Corretta

Per mantenere **esattamente** spread e total forniti dall'utente:

```python
# Formula CORRETTA: risolve il sistema
# lambda_h - lambda_a = spread
# lambda_h + lambda_a = total

lambda_h = (total + spread) / 2.0
lambda_a = (total - spread) / 2.0
```

### Verifica

```
lambda_h = (2.5 + 0.75) / 2 = 1.625
lambda_a = (2.5 - 0.75) / 2 = 0.875

spread_calc = 1.625 - 0.875 = 0.75 ✅
total_calc = 1.625 + 0.875 = 2.50 ✅
```

**Perfetto!** I vincoli sono rispettati.

---

## Impatto

### Conseguenze del Bug Attuale

1. **Calcoli Errati**: Tutti i mercati che dipendono da spread/total apertura sono sbagliati
2. **Movement Factor Invalido**: Il blend tra apertura e corrente usa valori alterati
3. **Perdita di Informazione**: L'informazione fornita dall'utente viene distorta
4. **Incoerenza**: Lo spread calcolato non corrisponde a quello osservato dal mercato

### Mercati Affetti

- ✅ 1X2 base (usa solo quote, non affetto)
- ❌ **Tutti i calcoli che usano spread/total apertura/corrente**
- ❌ **Movement factor** (calcola spread da lambda alterati)
- ❌ **Blend apertura/corrente** (usa lambda sbagliati)

---

## Codice da Correggere

### Localizzazione

**File**: `Frontendcloud.py`
**Funzione**: `apply_market_movement_blend()`
**Righe**: 10490-10526

### Patch Proposta

```python
# PRIMA (ERRATO):
lambda_total_ap = total_apertura_safe / 2.0
spread_clamped = max(-2.0, min(2.0, spread_apertura_safe))
spread_factor_ap = math.exp(spread_clamped * 0.5)
sqrt_ha = math.sqrt(home_advantage)
lambda_h_ap = lambda_total_ap * spread_factor_ap * sqrt_ha
lambda_a_ap = lambda_total_ap / spread_factor_ap / sqrt_ha

# DOPO (CORRETTO):
lambda_h_ap = (total_apertura_safe + spread_apertura_safe) / 2.0
lambda_a_ap = (total_apertura_safe - spread_apertura_safe) / 2.0
```

### Validazione Post-Fix

```python
# Verifica che spread e total siano mantenuti
spread_check = lambda_h_ap - lambda_a_ap
total_check = lambda_h_ap + lambda_a_ap

assert abs(spread_check - spread_apertura_safe) < 1e-6, \
    f"Spread non mantenuto: {spread_check} vs {spread_apertura_safe}"
assert abs(total_check - total_apertura_safe) < 1e-6, \
    f"Total non mantenuto: {total_check} vs {total_apertura_safe}"
```

---

## Domande da Chiarire

### 1. Home Advantage

La formula attuale usa `home_advantage` nel calcolo di lambda da spread.

**Domanda**: Se l'utente fornisce `spread = 0.75`, questo spread:
- ❓ **Include già** l'effetto home advantage?
- ❓ **Deve essere aggiustato** con home advantage?

**Se include già**: Usa la formula corretta senza `home_advantage`
**Se deve essere aggiustato**: Dobbiamo modificare lo spread prima, ma poi mantenere il risultato

### 2. Interpretazione Spread

**Domanda**: Qual è la definizione di spread?

- ❓ `spread = lambda_h - lambda_a` (Home più favorita se > 0)
- ❓ `spread = lambda_a - lambda_h` (Away più favorita se > 0)
- ❓ Spread è l'handicap asian (es. -0.75 favorisce Home)

**Nel codice attuale**: `spread = lambda_h - lambda_a`

**Problema dell'utente**: "Spread aumenta 0.75→1.25 ma P(2) non aumenta"
- Se spread = lambda_h - lambda_a, allora quando aumenta, P(2) **deve diminuire** ✅
- Se l'utente si aspetta P(2) aumenti, forse interpreta spread al contrario ❓

---

## Test di Validazione

### Test 1: Mantenimento Spread/Total

```python
def test_spread_total_preservation():
    spread_input = 0.75
    total_input = 2.5

    # Formula corretta
    lambda_h = (total_input + spread_input) / 2.0
    lambda_a = (total_input - spread_input) / 2.0

    spread_calc = lambda_h - lambda_a
    total_calc = lambda_h + lambda_a

    assert abs(spread_calc - spread_input) < 1e-10
    assert abs(total_calc - total_input) < 1e-10
```

### Test 2: Impatto su Mercati Away

```python
def test_spread_increase_decreases_prob_away():
    """
    Quando spread aumenta (Home diventa più favorita),
    P(Away) deve diminuire.
    """
    # Spread basso → Away più forte
    lambda_h1 = (2.5 + 0.5) / 2 = 1.5
    lambda_a1 = (2.5 - 0.5) / 2 = 1.0
    prob_away1 = calc_prob_away(lambda_h1, lambda_a1)

    # Spread alto → Away più debole
    lambda_h2 = (2.5 + 1.5) / 2 = 2.0
    lambda_a2 = (2.5 - 1.5) / 2 = 0.5
    prob_away2 = calc_prob_away(lambda_h2, lambda_a2)

    assert prob_away2 < prob_away1  # P(Away) diminuisce ✅
```

---

## Raccomandazioni

### 1. FIX IMMEDIATO (Critico)

✅ **Sostituire la formula** in `apply_market_movement_blend()`:

```python
# Linee 10490-10526
lambda_h_ap = (total_apertura_safe + spread_apertura_safe) / 2.0
lambda_a_ap = (total_apertura_safe - spread_apertura_safe) / 2.0

# Verifica
assert abs((lambda_h_ap - lambda_a_ap) - spread_apertura_safe) < 1e-6
assert abs((lambda_h_ap + lambda_a_ap) - total_apertura_safe) < 1e-6
```

### 2. CHIARIRE con Utente

❓ **Interpretazione spread**:
- Confermare che `spread = lambda_h - lambda_a`
- Confermare che spread > 0 significa Home favorita
- Spiegare che spread aumenta → P(Away) diminuisce (matematica corretta)

❓ **Home Advantage**:
- Lo spread fornito include già home advantage?
- Se sì, non serve moltiplicare per `sqrt(home_advantage)`

### 3. RIFARE Test Completi

Dopo il fix, ripetere tutti i test con la formula corretta per verificare che:
- ✅ Spread e total vengono mantenuti esattamente
- ✅ Movement factor calcola movimenti corretti
- ✅ Blend tra apertura e corrente usa valori corretti
- ✅ Probabilità mercati Away coerenti con spread

---

## Conclusione

### Bug Confermato: 🔴 CRITICO

La formula attuale **NON rispetta** i valori di spread e total forniti dall'utente.

### Fix: ✅ SEMPLICE

Usare la formula lineare diretta che garantisce matematicamente il rispetto dei vincoli.

### Dopo il Fix

La domanda dell'utente ("perché P(2+multigol) non aumenta quando spread aumenta?") avrà una risposta chiara:
- **Matematicamente**: Quando spread aumenta, Home diventa più favorita, quindi P(2) **deve diminuire**
- **Se l'utente si aspetta il contrario**: C'è un'interpretazione errata della definizione di spread

---

**Stato**: 🔴 **BUG CONFERMATO - FIX NECESSARIO**
**Priorità**: **MASSIMA**
**Complessità Fix**: **BASSA** (5 minuti)
**Impatto**: **ALTO** (tutti i calcoli con spread/total apertura/corrente)
