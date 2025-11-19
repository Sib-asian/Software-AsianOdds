# Calibrazione Soglie Sistema Automazione

## Problema Risolto

Il sistema inviava troppi segnali nonostante soglie teoricamente alte (70% confidence, 8% EV).

## Cause Identificate

1. **Dati Live Inesistenti**: TheOddsAPI non fornisce dati live (minuto, score), quindi le analisi "live" erano basate solo sull'ora di inizio, causando raccomandazioni assurde come:
   - "Più gol nel primo tempo" quando già nel secondo tempo
   - "2 (trasferta)" quando la favorita sta vincendo

2. **Mancanza Market Selection**: Il sistema non sceglieva intelligentemente quale mercato (HOME/DRAW/AWAY) puntare

3. **Filtri Troppo Permissivi**: Le soglie nominali non erano sufficienti

## Soluzioni Implementate

### 1. Disabilitazione Analisi "Live"

```python
# Prima (SBAGLIATO):
is_live = commence_time < now  # Marcava come "live" solo per ora inizio

# Dopo (CORRETTO):
if commence_time < now:
    continue  # SKIP - nessun dato live disponibile
```

**Risultato**: Il sistema analizza SOLO partite PRE-MATCH (future entro 24h)

### 2. Selezione Intelligente Mercato

Nuova funzione `_determine_best_market()` che:
- Calcola EV per HOME/DRAW/AWAY
- Seleziona il mercato con EV più alto
- Valida che probabilità sia ragionevole (>25%)
- Confronta probabilità AI vs probabilità implicita bookmaker

```python
# Esempio:
# Quote: 1=2.10 X=3.40 2=3.20
# AI dice: HOME ha 55% probabilità
# Sistema calcola:
#   - EV_HOME = (0.55 * 2.10 - 1) * 100 = +15.5%
#   - EV_DRAW = (0.30 * 3.40 - 1) * 100 = +2.0%
#   - EV_AWAY = (0.15 * 3.20 - 1) * 100 = -52.0%
# Sceglie: HOME (EV più alto)
```

### 3. Filtri Rigorosi Multipli

Il sistema ora applica **8 filtri in cascata**:

| Filtro | Soglia | Scopo |
|--------|--------|-------|
| 1. EV Base | >= 10% | Base + 2% di margine |
| 2. Confidence | >= 70% | Soglia utente |
| 3. Edge | >= 7% | Prob vs Quota implicita |
| 4. Uncertainty | <= 20% | Coerenza modelli AI |
| 5. Quote Min | >= 1.30 | Evita favoriti eccessivi |
| 6. Quote Max | <= 5.0 | Evita trap bookmaker |
| 7. Prob Min | >= 30% | Evita sottostime |
| 8. Prob Max | <= 75% | Evita sovrastime |

**Esempio di filtro in azione**:

```
❌ Segnale DEBOLE rifiutato:
- EV: 9% < 10% (soglia effettiva) ❌
- Confidence: 72% >= 70% ✓
- Uncertainty: 22% > 20% ❌
- Quote: 2.50 (ok) ✓
RISULTATO: NON passa filtri

✅ Segnale FORTE accettato:
- EV: 15.5% >= 10% ✓
- Confidence: 75% >= 70% ✓
- Edge: 11% >= 7% ✓
- Uncertainty: 15% <= 20% ✓
- Quote: 2.10 (1.30-5.0) ✓
- Prob: 55% (30-75%) ✓
RISULTATO: Passa tutti i filtri
```

### 4. Messaggi in Italiano

Tutti i messaggi Telegram ora sono in italiano:

```
Prima (inglese):
- Market: 1X2_HOME
- Expected Value: +15.5%
- Win Probability: 55.0%
- Confidence: HIGH (75%)

Dopo (italiano):
- Mercato: 1 (Vittoria Casa)
- Valore Atteso (EV): +15.5%
- Probabilità Vittoria: 55.0%
- Confidenza: ALTA (75%)
```

### 5. Statistiche Dettagliate

Ogni notifica ora include statistiche per verifica:

```
📈 Statistiche Estratte
Quote: 1=2.10 X=3.40 2=3.20
Prob. Implicite: 1=44.0% X=27.2% 2=28.9%
Margine Bookmaker: 8.3%
Vantaggio: +11.0% (Nostra prob. vs Bookmaker)

🤖 Ensemble AI
Dixon-Coles: 52.3% (peso: 35%)
XGBoost: 58.1% (peso: 40%)
LSTM: 54.7% (peso: 25%)
Incertezza: 15.0%
```

## Come Regolare le Soglie

Se ricevi ancora troppi segnali, puoi:

### Opzione 1: Aumentare soglie base (consigliato)

Nel file `.env` o variabili ambiente Railway:

```bash
AUTOMATION_MIN_EV=12.0          # Da 8% a 12%
AUTOMATION_MIN_CONFIDENCE=75.0  # Da 70% a 75%
```

### Opzione 2: Modificare filtri interni

In `automation_24h.py`, funzione `_is_real_value_opportunity()`:

```python
# Più restrittivo
effective_min_ev = self.min_ev + 4.0  # Da +2% a +4%
min_edge=0.10  # Da 7% a 10%
uncertainty > 0.15  # Da 20% a 15%
odds < 1.50  # Da 1.30 a 1.50
probability < 0.35  # Da 30% a 35%
```

### Opzione 3: Limita partite analizzate

In `_fetch_real_matches()`:

```python
# Analizza solo prime 10 partite
if len(matches) >= 10:
    break
```

## Test Automatico

Esegui test per verificare i filtri:

```bash
python3 test_market_selection.py
```

Output atteso:
```
✓ Segnale FORTE: Passa tutti i filtri
✗ Segnale DEBOLE: Non passa (come previsto)
```

## Monitoraggio

Controlla i log per vedere quanti segnali vengono filtrati:

```bash
grep "No real value" /var/log/automation.log
grep "EV too low" /var/log/automation.log
grep "Confidence too low" /var/log/automation.log
grep "High model uncertainty" /var/log/automation.log
```

## Risultati Attesi

Con le nuove soglie:

| Prima | Dopo |
|-------|------|
| ~50-100 segnali/giorno | ~5-15 segnali/giorno |
| Molti falsi positivi | Solo opportunità solide |
| Messaggi in inglese | Messaggi in italiano |
| Dati live errati | Solo pre-match |
| Nessuna statistica | Statistiche complete |

## Supporto

Se continui ad avere problemi:
1. Controlla i log per pattern
2. Aumenta gradualmente le soglie
3. Usa `test_market_selection.py` per debugging
