# 🔧 Analisi e Miglioramenti Calcoli Market Movement Analyzer

## 📋 Riepilogo Problemi Identificati

Dopo un'analisi approfondita del file `market_movement_analyzer.py`, ho identificato **7 aree critiche** dove i calcoli sono troppo grezzi e possono essere significativamente migliorati.

---

## ❌ Problema 1: Calcolo Intensità Troppo Semplice

### Situazione Attuale:
```python
# Calcola intensità
abs_movement = abs(movement)
if abs_movement < 0.3:
    intensity = MovementIntensity.LIGHT
elif abs_movement < 0.6:
    intensity = MovementIntensity.MEDIUM
else:
    intensity = MovementIntensity.STRONG
```

### Problemi:
1. **Soglie fisse e arbitrarie**: 0.3 e 0.6 sono valori hardcoded senza considerare il contesto
2. **Non considera i valori discreti**: Negli spread/total, i valori sono multipli di 0.25, quindi 0.3 non ha senso
3. **Non normalizza per il valore di base**: Un movimento di 0.5 su uno spread da -0.5 è molto più significativo che su uno spread da -2.5

### ✅ Miglioramento Proposto:
```python
def _calculate_intensity(self, movement: float, opening_value: float, 
                        closing_value: float, is_spread: bool) -> MovementIntensity:
    """Calcola intensità normalizzata e contestualizzata"""
    
    abs_movement = abs(movement)
    
    # Normalizza per valore base (movimento relativo)
    if abs(opening_value) > 0.01:
        relative_movement = abs_movement / abs(opening_value)
    else:
        relative_movement = abs_movement
    
    # Arrotonda a step discreti (0.25)
    discrete_steps = round(abs_movement / 0.25)
    
    # Intensità basata su step discreti + movimento relativo
    if discrete_steps == 0:
        return MovementIntensity.NONE
    elif discrete_steps == 1:
        # Leggero: 1 step (0.25) o movimento relativo < 15%
        if relative_movement < 0.15:
            return MovementIntensity.LIGHT
        else:
            return MovementIntensity.MEDIUM
    elif discrete_steps == 2:
        # Medio: 2 step (0.5) o movimento relativo 15-35%
        if relative_movement < 0.35:
            return MovementIntensity.MEDIUM
        else:
            return MovementIntensity.STRONG
    elif discrete_steps >= 3:
        # Forte: 3+ step (0.75+) o movimento relativo > 35%
        return MovementIntensity.STRONG
    else:
        # Default per valori intermedi
        if relative_movement < 0.20:
            return MovementIntensity.LIGHT
        elif relative_movement < 0.40:
            return MovementIntensity.MEDIUM
        else:
            return MovementIntensity.STRONG
```

---

## ❌ Problema 2: Calcolo Step Impreciso

### Situazione Attuale:
```python
# Calcola step (multipli di 0.25)
steps = abs_movement / 0.25
```

### Problemi:
1. **Non arrotonda**: Restituisce valori decimali come 1.2, 2.7 invece di interi
2. **Non considera il movimento reale**: Se lo spread passa da -1.0 a -1.25, è 1 step, non 0.25/0.25 = 1.0
3. **Non distingue direzione**: Non tiene conto se il movimento attraversa linee chiave (es. -0.75 → -1.0)

### ✅ Miglioramento Proposto:
```python
def _calculate_discrete_steps(self, opening: float, closing: float, 
                              is_spread: bool) -> float:
    """Calcola step discreti effettivi tra valori di linea"""
    
    # Normalizza ai valori di linea (multipli di 0.25)
    def normalize_to_line(value: float) -> float:
        return round(value * 4) / 4
    
    norm_open = normalize_to_line(opening)
    norm_close = normalize_to_line(closing)
    
    if is_spread:
        # Per spread: usa valore assoluto
        abs_open = abs(norm_open)
        abs_close = abs(norm_close)
        diff = abs_close - abs_open
    else:
        # Per total: differenza diretta
        diff = norm_close - norm_open
    
    # Converti in step discreti (ogni 0.25 = 1 step)
    steps = diff / 0.25
    
    return round(steps, 2)  # Arrotonda a 2 decimali per precisione
```

---

## ❌ Problema 3: Stima Total HT Troppo Grezza

### Situazione Attuale:
```python
# Over/Under HT
ht_total_estimate = total.closing_value * 0.5
```

### Problemi:
1. **Assunzione troppo semplice**: Non tutti i gol vengono segnati nella stessa proporzione
2. **Non considera il contesto**: Una partita che parte lenta può avere più gol nel 2T, e viceversa
3. **Ignora lo spread**: Se lo spread si indurisce, il favorito può segnare prima, aumentando gol 1T

### ✅ Miglioramento Proposto:
```python
def _estimate_ht_total(self, total_value: float, spread_analysis: MovementAnalysis,
                      total_analysis: MovementAnalysis) -> float:
    """Stima più accurata del Total per primo tempo"""
    
    base_ht = total_value * 0.45  # Base: ~45% dei gol nel 1T (media storica)
    
    # Fattore spread: favorito forte → più gol 1T
    if spread_analysis.direction == MovementDirection.HARDEN:
        if spread_analysis.intensity == MovementIntensity.STRONG:
            spread_factor = 1.15  # +15% se favorito molto forte
        else:
            spread_factor = 1.08  # +8% se favorito medio
    elif spread_analysis.direction == MovementDirection.SOFTEN:
        spread_factor = 0.92  # -8% se favorito debole (partita più equilibrata)
    else:
        spread_factor = 1.0
    
    # Fattore total movement: se total sale → partita più viva → più gol 1T
    if total_analysis.direction == MovementDirection.HARDEN:
        total_factor = 1.10 if total_analysis.intensity == MovementIntensity.STRONG else 1.05
    elif total_analysis.direction == MovementDirection.SOFTEN:
        total_factor = 0.95  # Partita più tattica
    else:
        total_factor = 1.0
    
    # Valore base totale
    adjusted_ht = base_ht * spread_factor * total_factor
    
    # Normalizza ai valori di linea
    return round(adjusted_ht * 4) / 4
```

---

## ❌ Problema 4: Interpretazioni con Match Esatti

### Situazione Attuale:
```python
for start, (end, interpretation) in self.SPREAD_MOVEMENTS_DOWN.items():
    if opening >= start and closing <= end:
        return interpretation
```

### Problemi:
1. **Match esatti**: Se opening è -1.3 e closing -0.9, non matcha nulla
2. **Manca interpolazione**: Non considera movimenti parziali o intermedi
3. **Fallback generico**: Troppo vago quando non trova match

### ✅ Miglioramento Proposto:
```python
def _get_interpolated_interpretation(self, opening: float, closing: float,
                                    movements_dict: Dict, is_down: bool) -> str:
    """Ottiene interpretazione interpolata tra valori tabellari"""
    
    # Trova il range più vicino
    best_match = None
    min_distance = float('inf')
    
    for start, (end, interpretation) in movements_dict.items():
        if is_down:
            # Movimento verso il basso (ammorbidisce)
            if opening >= start and closing <= end:
                # Match esatto
                return interpretation
            
            # Calcola distanza dal range
            if opening >= start:
                distance = abs(closing - end) if closing > end else 0
            else:
                distance = abs(opening - start)
        else:
            # Movimento verso l'alto (indurisce)
            if opening <= start and closing >= end:
                return interpretation
            
            if opening <= start:
                distance = abs(closing - end) if closing < end else 0
            else:
                distance = abs(opening - start)
        
        if distance < min_distance:
            min_distance = distance
            best_match = (start, end, interpretation)
    
    # Se trovato un match vicino, modifica l'interpretazione
    if best_match and min_distance < 0.5:
        start, end, base_interpretation = best_match
        
        # Calcola quanto siamo lontani dal match perfetto
        if is_down:
            movement_completion = abs(closing - end) / abs(start - end) if abs(start - end) > 0 else 1.0
        else:
            movement_completion = abs(closing - end) / abs(start - end) if abs(start - end) > 0 else 1.0
        
        if movement_completion < 0.3:
            modifier = "parzialmente "
        elif movement_completion > 0.7:
            modifier = ""
        else:
            modifier = "progressivamente "
        
        return modifier + base_interpretation.lower()
    
    # Fallback più dettagliato
    if is_down:
        direction_text = "calo di fiducia" if abs(closing) < abs(opening) else "ammorbidimento"
    else:
        direction_text = "aumento di fiducia" if abs(closing) > abs(opening) else "indurimento"
    
    return f"Spread mostra {direction_text} da {opening} a {closing}"
```

---

## ❌ Problema 5: Soglia Stabilità Troppo Rigida

### Situazione Attuale:
```python
if abs(movement) < 0.01:
    direction = MovementDirection.STABLE
```

### Problemi:
1. **Troppo rigido**: 0.01 è troppo piccolo per spread/total che hanno granularità 0.25
2. **Non considera rumore**: Piccole variazioni di 0.05-0.10 possono essere rumore, non movimento reale
3. **Dovrebbe essere basato su step discreti**: < 0.125 (mezzo step) = stabile

### ✅ Miglioramento Proposto:
```python
def _is_stable_movement(self, movement: float, is_spread: bool) -> bool:
    """Determina se il movimento è stabile (rumore vs movimento reale)"""
    
    abs_movement = abs(movement)
    
    # Per valori discreti (0.25), consideriamo stabile se < 0.125 (mezzo step)
    discrete_threshold = 0.125
    
    # Movimento relativo: se < 2% del valore base, è probabilmente rumore
    # (assumendo che il valore base sia significativo)
    
    if abs_movement < discrete_threshold:
        return True
    
    # Se il movimento è molto piccolo rispetto alla granularità
    # (es. 0.05 su spread -1.5 = 3% = rumore)
    if abs_movement < 0.05:
        return True
    
    return False
```

---

## ❌ Problema 6: Confidenza Non Pesata per Intensità

### Situazione Attuale:
```python
def _calculate_confidence(self, spread: MovementAnalysis,
                         total: MovementAnalysis) -> ConfidenceLevel:
    # SEGNALI CONTRASTANTI
    if (spread.direction == MovementDirection.HARDEN and total.direction == MovementDirection.SOFTEN):
        return ConfidenceLevel.MEDIUM
    
    # Se entrambi concordi (stessa direzione) e forti → HIGH
    if (spread.direction == total.direction and
        spread.direction != MovementDirection.STABLE and
        spread.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG] and
        total.intensity in [MovementIntensity.MEDIUM, MovementIntensity.STRONG]):
        return ConfidenceLevel.HIGH
```

### Problemi:
1. **Non pesa l'intensità**: Due movimenti MEDIUM hanno stessa confidenza di due STRONG
2. **Non considera l'ampiezza**: Un movimento da -0.5 a -1.0 è diverso da -2.0 a -2.5
3. **Binario**: O HIGH o MEDIUM, manca granularità

### ✅ Miglioramento Proposto:
```python
def _calculate_confidence(self, spread: MovementAnalysis,
                         total: MovementAnalysis) -> ConfidenceLevel:
    """Calcola confidenza pesata per intensità e ampiezza"""
    
    # Pesi per intensità
    intensity_weights = {
        MovementIntensity.NONE: 0.0,
        MovementIntensity.LIGHT: 0.3,
        MovementIntensity.MEDIUM: 0.6,
        MovementIntensity.STRONG: 1.0
    }
    
    spread_weight = intensity_weights.get(spread.intensity, 0.5)
    total_weight = intensity_weights.get(total.intensity, 0.5)
    
    # Fattore ampiezza: movimento più ampio = più confidenza
    spread_amplitude = min(abs(spread.movement_steps) / 4.0, 1.0)  # Normalizza a max 4 step
    total_amplitude = min(abs(total.movement_steps) / 4.0, 1.0)
    
    # Peso combinato
    combined_weight = (spread_weight * 0.6 + total_weight * 0.4)  # Spread più importante
    amplitude_factor = (spread_amplitude * 0.6 + total_amplitude * 0.4)
    
    # Score finale (0-1)
    confidence_score = combined_weight * 0.7 + amplitude_factor * 0.3
    
    # Penalità per segnali contrastanti
    if (spread.direction == MovementDirection.HARDEN and total.direction == MovementDirection.SOFTEN) or \
       (spread.direction == MovementDirection.SOFTEN and total.direction == MovementDirection.HARDEN):
        confidence_score *= 0.65  # Riduce confidenza del 35%
    
    # Conversione a ConfidenceLevel
    if confidence_score >= 0.75:
        return ConfidenceLevel.HIGH
    elif confidence_score >= 0.45:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
```

---

## ❌ Problema 7: Calcolo Step per Spread Non Considera Cambio Segno

### Situazione Attuale:
```python
abs_opening = abs(opening)
abs_closing = abs(closing)
movement = abs_closing - abs_opening
```

### Problemi:
1. **Ignora cambio segno**: Se spread passa da -0.5 a +0.5, è un cambio di favorito, non solo "ammorbidimento"
2. **Non distingue casa/trasferta**: -1.0 e +1.0 sono diversi, non solo valori assoluti

### ✅ Miglioramento Proposto:
```python
def analyze(self, opening: float, closing: float) -> MovementAnalysis:
    """Analizza il movimento dello spread con gestione cambio segno"""
    
    # Gestione cambio segno (pick'em o cambio favorito)
    sign_changed = (opening > 0 and closing < 0) or (opening < 0 and closing > 0)
    
    if sign_changed:
        # Cambio di favorito: caso speciale
        total_movement = abs(opening) + abs(closing)
        direction = MovementDirection.SOFTEN  # Sempre ammorbidisce verso equilibrio
        interpretation = f"Cambio favorito: da {format_spread_display(opening)} a {format_spread_display(closing)}"
    else:
        # Movimento normale: usa valore assoluto
        abs_opening = abs(opening)
        abs_closing = abs(closing)
        movement = abs_closing - abs_opening
        total_movement = abs(movement)
        
        if total_movement < 0.125:
            direction = MovementDirection.STABLE
            interpretation = "Spread stabile"
        elif movement < 0:
            direction = MovementDirection.SOFTEN
            interpretation = self._get_soften_interpretation(opening, closing)
        else:
            direction = MovementDirection.HARDEN
            interpretation = self._get_harden_interpretation(opening, closing)
    
    # Resto della logica...
```

---

## 📊 Riepilogo Miglioramenti

| # | Problema | Impatto | Priorità |
|---|----------|---------|----------|
| 1 | Intensità troppo semplice | Alto | 🔴 Alta |
| 2 | Step imprecisi | Medio | 🟡 Media |
| 3 | Total HT grezzo | Medio | 🟡 Media |
| 4 | Interpretazioni rigide | Basso | 🟢 Bassa |
| 5 | Soglia stabilità | Medio | 🟡 Media |
| 6 | Confidenza non pesata | Alto | 🔴 Alta |
| 7 | Cambio segno spread | Medio | 🟡 Media |

---

## 🎯 Prossimi Passi

1. **Implementare miglioramenti priorità ALTA** (1, 6)
2. **Testare con dati reali** per validare le nuove formule
3. **Aggiungere logging** per tracciare i calcoli migliorati
4. **Calibrare i parametri** basandosi su performance storiche

---

*Documento creato: 2024*
*Versione: 1.0*

