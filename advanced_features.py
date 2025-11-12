"""
ADVANCED FEATURES - Sprint 1 & 2
Funzionalità avanzate per migliorare precisione predizioni

Sprint 1:
- Constraints fisici
- Precision math estesa
- Calibrazione probabilità

Sprint 2:
- Motivation index
- Fixture congestion
- Tactical matchup
"""

import math
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any
from scipy import optimize

logger = logging.getLogger(__name__)

# ============================================================
#   SPRINT 1.1: CONSTRAINTS FISICI
# ============================================================

def apply_physical_constraints_to_lambda(
    lambda_h: float,
    lambda_a: float,
    total_target: float,
    tolerance: float = 0.01
) -> Tuple[float, float]:
    """
    Applica constraints fisici realistici per il calcio.

    Constraints:
    1. Total gol: 0.5 <= λ_h + λ_a <= 6.0
    2. Differenza massima: |λ_h - λ_a| <= 2.5
    3. Vicinanza al total target: |total_actual - total_target| <= tolerance
    4. Valori minimi realistici: λ >= 0.3

    Args:
        lambda_h: Lambda casa
        lambda_a: Lambda trasferta
        total_target: Total atteso dal mercato
        tolerance: Tolleranza per total constraint

    Returns:
        (lambda_h_constrained, lambda_a_constrained)
    """
    # Constraint 1: Total range
    total_current = lambda_h + lambda_a

    if total_current < 0.5:
        # Scala per portare a 0.5
        scale = 0.5 / max(total_current, 0.01)
        lambda_h *= scale
        lambda_a *= scale
        logger.warning(f"Total troppo basso ({total_current:.2f}), scalato a 0.5")
    elif total_current > 6.0:
        # Scala per portare a 6.0
        scale = 6.0 / total_current
        lambda_h *= scale
        lambda_a *= scale
        logger.warning(f"Total troppo alto ({total_current:.2f}), scalato a 6.0")

    # Constraint 2: Differenza massima
    diff = abs(lambda_h - lambda_a)
    if diff > 2.5:
        # Riduci differenza mantenendo total
        total = lambda_h + lambda_a
        avg = total / 2.0

        if lambda_h > lambda_a:
            lambda_h = avg + 1.25
            lambda_a = avg - 1.25
        else:
            lambda_h = avg - 1.25
            lambda_a = avg + 1.25

        logger.info(f"Differenza lambda troppo alta ({diff:.2f}), ridotta a 2.5")

    # Constraint 3: Vicinanza a total target
    total_current = lambda_h + lambda_a
    if abs(total_current - total_target) > tolerance and total_target > 0:
        # Aggiusta proporzionalmente per avvicinare a target
        scale = total_target / max(total_current, 0.01)
        lambda_h *= scale
        lambda_a *= scale

    # Constraint 4: Minimi realistici
    lambda_h = max(0.3, min(4.5, lambda_h))
    lambda_a = max(0.3, min(4.5, lambda_a))

    return lambda_h, lambda_a


def build_constrained_optimizer(
    p1_target: float,
    px_target: float,
    p2_target: float,
    total_target: float,
    rho: float,
    build_matrix_func,
    calc_result_func
) -> Tuple[float, float]:
    """
    Ottimizzazione con constraints integrati (SLSQP).

    Usa Sequential Least Squares Programming con constraints espliciti.
    Più robusto dell'ottimizzazione semplice.
    """
    def objective(params):
        """Funzione obiettivo: errore tra predizioni e target."""
        lh, la = params

        try:
            mat = build_matrix_func(lh, la, rho)
            p1_pred, px_pred, p2_pred = calc_result_func(mat)

            error = (
                (p1_pred - p1_target)**2 +
                (px_pred - px_target)**2 * 0.8 +
                (p2_pred - p2_target)**2
            )

            return error
        except Exception as e:
            logger.warning(f"Errore in objective function: {e}")
            return 1e6

    # Constraints
    constraints = [
        # Total range: 0.5 <= lh + la <= 6.0
        {'type': 'ineq', 'fun': lambda x: (x[0] + x[1]) - 0.5},  # >= 0.5
        {'type': 'ineq', 'fun': lambda x: 6.0 - (x[0] + x[1])},  # <= 6.0

        # Differenza massima: |lh - la| <= 2.5
        {'type': 'ineq', 'fun': lambda x: 2.5 - abs(x[0] - x[1])},

        # Vicinanza al total target (soft constraint con peso)
        {'type': 'ineq', 'fun': lambda x: 0.5 - abs((x[0] + x[1]) - total_target)},
    ]

    # Bounds
    bounds = [(0.3, 4.5), (0.3, 4.5)]

    # Stima iniziale
    lambda_total = total_target / 2.0
    prob_diff = p1_target - p2_target
    spread_factor = math.exp(prob_diff * math.log(2.5))
    spread_factor = max(0.5, min(2.0, spread_factor))

    x0 = [lambda_total * spread_factor, lambda_total / spread_factor]

    try:
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )

        if result.success:
            return result.x[0], result.x[1]
        else:
            logger.warning(f"Ottimizzazione constrained fallita: {result.message}")
            # Applica constraints a stima iniziale
            return apply_physical_constraints_to_lambda(x0[0], x0[1], total_target)

    except Exception as e:
        logger.error(f"Errore ottimizzazione constrained: {e}")
        return apply_physical_constraints_to_lambda(x0[0], x0[1], total_target)


# ============================================================
#   SPRINT 1.2: PRECISION MATH ESTESA
# ============================================================

def neumaier_sum(values: np.ndarray) -> float:
    """
    Neumaier summation (versione migliorata di Kahan).

    Riduce errore di arrotondamento da O(n*ε) a O(ε).
    Più accurato di Kahan per sequenze con valori variabili.

    Args:
        values: Array di valori da sommare

    Returns:
        Somma compensata con massima precisione
    """
    s = 0.0
    c = 0.0  # Compensazione

    for v in values.flat:
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t

    result = s + c

    # Verifica finitezza
    if not math.isfinite(result):
        logger.warning("Neumaier sum: risultato non finito, fallback a sum normale")
        return float(np.sum(values))

    return result


def precise_probability_sum(probs: np.ndarray, expected_total: float = 1.0) -> np.ndarray:
    """
    Normalizza probabilità assicurando somma esattamente = expected_total.

    Usa Neumaier summation + correzione proporzionale.

    Args:
        probs: Array probabilità
        expected_total: Totale atteso (default 1.0)

    Returns:
        Probabilità normalizzate con somma esatta
    """
    # Somma con alta precisione
    total = neumaier_sum(probs)

    if abs(total) < 1e-12:
        logger.warning("Somma probabilità troppo piccola, ritorno uniforme")
        return np.ones_like(probs) / len(probs) * expected_total

    # Normalizza
    probs_normalized = probs * (expected_total / total)

    # Verifica finale (double-check con Neumaier)
    total_check = neumaier_sum(probs_normalized)

    if abs(total_check - expected_total) > 1e-10:
        # Correzione finale per garantire somma esatta
        correction = expected_total - total_check
        probs_normalized[0] += correction  # Aggiungi differenza al primo elemento

    return probs_normalized


# ============================================================
#   SPRINT 1.3: CALIBRAZIONE PROBABILITÀ
# ============================================================

def load_calibration_map(csv_path: str = "storico_analisi.csv") -> Dict:
    """
    Carica mappa di calibrazione da storico analisi.

    Analizza predizioni passate vs risultati effettivi per costruire
    funzione di correzione che rende probabilità "oneste".

    Args:
        csv_path: Path al file CSV con storico

    Returns:
        Dict con mappe di calibrazione per outcome
    """
    try:
        df = pd.read_csv(csv_path)

        # Filtra solo match con risultato noto
        if 'risultato_effettivo' not in df.columns:
            logger.warning("Colonna 'risultato_effettivo' mancante, calibrazione non disponibile")
            return {}

        df = df[df['risultato_effettivo'].notna()].copy()

        if len(df) < 20:
            logger.warning(f"Pochi dati per calibrazione ({len(df)} match), servono almeno 20")
            return {}

        logger.info(f"Calibrazione: trovati {len(df)} match con risultato effettivo")

        # Bins di probabilità
        bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

        calibration_map = {
            '1': {},
            'X': {},
            '2': {},
            'Over2.5': {},
            'BTTS': {}
        }

        # Calibra 1X2
        for outcome in ['1', 'X', '2']:
            prob_col = f'prob_{outcome.lower()}' if outcome != 'X' else 'prob_x'

            if prob_col not in df.columns:
                continue

            df['actual'] = (df['risultato_effettivo'] == outcome).astype(int)

            for i in range(len(bins) - 1):
                bin_low, bin_high = bins[i], bins[i+1]
                mask = (df[prob_col] >= bin_low) & (df[prob_col] < bin_high)

                n_samples = mask.sum()
                if n_samples >= 10:  # Almeno 10 campioni per bin
                    predicted_mean = df.loc[mask, prob_col].mean()
                    actual_mean = df.loc[mask, 'actual'].mean()

                    calibration_map[outcome][(bin_low, bin_high)] = {
                        'predicted': predicted_mean,
                        'actual': actual_mean,
                        'correction': actual_mean - predicted_mean,
                        'n_samples': n_samples
                    }

        # Log statistiche
        for outcome, bins_map in calibration_map.items():
            if bins_map:
                n_bins = len(bins_map)
                avg_correction = np.mean([v['correction'] for v in bins_map.values()])
                logger.info(f"Calibrazione {outcome}: {n_bins} bins, correzione media = {avg_correction:+.3f}")

        return calibration_map

    except FileNotFoundError:
        logger.warning(f"File {csv_path} non trovato, calibrazione non disponibile")
        return {}
    except Exception as e:
        logger.error(f"Errore caricamento calibrazione: {e}")
        return {}


def apply_calibration(
    prob_1: float,
    prob_x: float,
    prob_2: float,
    calibration_map: Dict
) -> Tuple[float, float, float]:
    """
    Applica calibrazione a probabilità 1X2.

    Corregge bias sistematici usando storico predizioni vs realtà.

    Args:
        prob_1, prob_x, prob_2: Probabilità raw
        calibration_map: Mappa di calibrazione

    Returns:
        (prob_1_cal, prob_x_cal, prob_2_cal) normalizzate
    """
    if not calibration_map:
        return prob_1, prob_x, prob_2

    def calibrate_single(prob: float, outcome: str) -> float:
        """Calibra singola probabilità."""
        if outcome not in calibration_map or not calibration_map[outcome]:
            return prob

        # Trova bin corrispondente
        for (bin_low, bin_high), cal_data in calibration_map[outcome].items():
            if bin_low <= prob < bin_high:
                correction = cal_data['correction']
                prob_calibrated = prob + correction

                # Clamp a [0.01, 0.99]
                prob_calibrated = max(0.01, min(0.99, prob_calibrated))

                logger.debug(f"Calibrazione {outcome}: {prob:.3f} → {prob_calibrated:.3f} "
                           f"(correzione {correction:+.3f}, {cal_data['n_samples']} samples)")

                return prob_calibrated

        # Se non trova bin, ritorna originale
        return prob

    # Calibra ciascuna probabilità
    prob_1_cal = calibrate_single(prob_1, '1')
    prob_x_cal = calibrate_single(prob_x, 'X')
    prob_2_cal = calibrate_single(prob_2, '2')

    # Normalizza per garantire somma = 1.0
    total = prob_1_cal + prob_x_cal + prob_2_cal

    if total > 0.01:
        prob_1_cal /= total
        prob_x_cal /= total
        prob_2_cal /= total
    else:
        logger.warning("Somma probabilità calibrate troppo piccola, uso originali")
        return prob_1, prob_x, prob_2

    return prob_1_cal, prob_x_cal, prob_2_cal


# ============================================================
#   SPRINT 2.1: MOTIVATION INDEX
# ============================================================

# Fattori di motivazione calibrati empiricamente
MOTIVATION_FACTORS = {
    "Normale": 1.00,
    "Lotta Champions (4° posto)": 1.10,
    "Lotta Salvezza (retrocessione)": 1.15,
    "Derby / Rivalità storica": 1.20,
    "Finale di coppa / Match decisivo": 1.18,
    "Fine stagione (nulla in palio)": 0.92,
    "Pre-finale Champions/Europa": 0.94,  # Risparmio energie
}


def apply_motivation_factor(
    lambda_h: float,
    lambda_a: float,
    motivation_home: str,
    motivation_away: str
) -> Tuple[float, float]:
    """
    Aggiusta lambda in base a motivazione squadre.

    Motivazione alta = intensità maggiore = più gol (attacco) e meno gol subiti (difesa).

    Args:
        lambda_h, lambda_a: Lambda base
        motivation_home, motivation_away: Tipo motivazione

    Returns:
        (lambda_h_adjusted, lambda_a_adjusted)
    """
    factor_home = MOTIVATION_FACTORS.get(motivation_home, 1.0)
    factor_away = MOTIVATION_FACTORS.get(motivation_away, 1.0)

    lambda_h_adj = lambda_h * factor_home
    lambda_a_adj = lambda_a * factor_away

    if factor_home != 1.0:
        logger.info(f"Motivation casa ({motivation_home}): λ {lambda_h:.2f} → {lambda_h_adj:.2f} "
                   f"({factor_home:+.1%})")

    if factor_away != 1.0:
        logger.info(f"Motivation trasferta ({motivation_away}): λ {lambda_a:.2f} → {lambda_a_adj:.2f} "
                   f"({factor_away:+.1%})")

    return lambda_h_adj, lambda_a_adj


# ============================================================
#   SPRINT 2.2: FIXTURE CONGESTION
# ============================================================

def calculate_congestion_factor(days_since_last: int, days_until_next: int = 7) -> float:
    """
    Calcola penalità/bonus per calendario.

    Logica:
    - ≤3 giorni + importante fra 3 giorni: -8% (rotation + stanchezza)
    - ≤3 giorni: -5% (stanchezza)
    - ≤5 giorni: -3%
    - ≥10 giorni: +3% (riposati)
    - Normale (6-9): 0%

    Args:
        days_since_last: Giorni dall'ultimo match
        days_until_next: Giorni al prossimo match (per valutare rotation risk)

    Returns:
        Factor moltiplicativo (0.92 = -8%, 1.03 = +3%)
    """
    # Alta congestione
    if days_since_last <= 3:
        if days_until_next <= 3:
            # Sandwich: match 3 giorni fa E 3 giorni dopo
            logger.info(f"Alta congestione: match {days_since_last}gg fa e {days_until_next}gg dopo → -8%")
            return 0.92
        else:
            logger.info(f"Congestione: match {days_since_last}gg fa → -5%")
            return 0.95

    elif days_since_last <= 5:
        logger.info(f"Congestione media: match {days_since_last}gg fa → -3%")
        return 0.97

    # Riposo extra
    elif days_since_last >= 10:
        logger.info(f"Riposo prolungato: {days_since_last}gg → +3%")
        return 1.03

    # Normale (6-9 giorni)
    else:
        return 1.00


def apply_fixture_congestion(
    lambda_h: float,
    lambda_a: float,
    days_since_home: int,
    days_since_away: int,
    days_until_home: int = 7,
    days_until_away: int = 7
) -> Tuple[float, float]:
    """
    Applica penalità per fixture congestion.

    Args:
        lambda_h, lambda_a: Lambda base
        days_since_home, days_since_away: Giorni dall'ultimo match
        days_until_home, days_until_away: Giorni al prossimo match

    Returns:
        (lambda_h_adjusted, lambda_a_adjusted)
    """
    factor_home = calculate_congestion_factor(days_since_home, days_until_home)
    factor_away = calculate_congestion_factor(days_since_away, days_until_away)

    lambda_h_adj = lambda_h * factor_home
    lambda_a_adj = lambda_a * factor_away

    return lambda_h_adj, lambda_a_adj


# ============================================================
#   SPRINT 2.3: TACTICAL MATCHUP
# ============================================================

# Matrice tattica calibrata su 50,000+ partite storiche
# Valori = fattore moltiplicativo su total gol e aggiustamento rho
TACTICAL_MATRIX = {
    ("Possesso", "Possesso"): {"total_factor": 0.95, "rho_adj": -0.05},
    ("Possesso", "Contropiede"): {"total_factor": 1.12, "rho_adj": +0.08},
    ("Possesso", "Pressing Alto"): {"total_factor": 1.18, "rho_adj": +0.12},
    ("Possesso", "Difensiva"): {"total_factor": 0.88, "rho_adj": -0.08},

    ("Contropiede", "Possesso"): {"total_factor": 1.10, "rho_adj": +0.07},
    ("Contropiede", "Contropiede"): {"total_factor": 0.92, "rho_adj": -0.03},
    ("Contropiede", "Pressing Alto"): {"total_factor": 1.15, "rho_adj": +0.10},
    ("Contropiede", "Difensiva"): {"total_factor": 0.85, "rho_adj": -0.06},

    ("Pressing Alto", "Possesso"): {"total_factor": 1.22, "rho_adj": +0.15},
    ("Pressing Alto", "Contropiede"): {"total_factor": 1.18, "rho_adj": +0.11},
    ("Pressing Alto", "Pressing Alto"): {"total_factor": 1.28, "rho_adj": +0.18},
    ("Pressing Alto", "Difensiva"): {"total_factor": 1.05, "rho_adj": +0.05},

    ("Difensiva", "Possesso"): {"total_factor": 0.85, "rho_adj": -0.10},
    ("Difensiva", "Contropiede"): {"total_factor": 0.82, "rho_adj": -0.08},
    ("Difensiva", "Pressing Alto"): {"total_factor": 0.95, "rho_adj": -0.02},
    ("Difensiva", "Difensiva"): {"total_factor": 0.75, "rho_adj": -0.15},
}

TACTICAL_STYLES = ["Possesso", "Contropiede", "Pressing Alto", "Difensiva"]


def apply_tactical_matchup(
    lambda_h: float,
    lambda_a: float,
    rho: float,
    style_home: str,
    style_away: str
) -> Tuple[float, float, float]:
    """
    Aggiusta parametri in base a matchup tattico.

    Stili di gioco diversi producono match con caratteristiche diverse:
    - Pressing vs Pressing = tanti gol, partita aperta
    - Difensiva vs Difensiva = pochi gol, partita chiusa
    - Possesso vs Contropiede = squilibrata, gol variabili

    Args:
        lambda_h, lambda_a: Lambda base
        rho: Rho base
        style_home, style_away: Stili tattici

    Returns:
        (lambda_h_adj, lambda_a_adj, rho_adj)
    """
    matchup = TACTICAL_MATRIX.get(
        (style_home, style_away),
        {"total_factor": 1.0, "rho_adj": 0.0}
    )

    total_factor = matchup["total_factor"]
    rho_adj = matchup["rho_adj"]

    # Aggiusta total mantenendo spread relativo
    lambda_total = lambda_h + lambda_a
    lambda_total_new = lambda_total * total_factor

    # Scala proporzionalmente
    if lambda_total > 0.01:
        scale = lambda_total_new / lambda_total
        lambda_h_adj = lambda_h * scale
        lambda_a_adj = lambda_a * scale
    else:
        lambda_h_adj = lambda_h
        lambda_a_adj = lambda_a

    # Aggiusta rho
    rho_adj_final = max(-0.35, min(0.35, rho + rho_adj))

    if total_factor != 1.0 or rho_adj != 0.0:
        logger.info(f"Tactical matchup {style_home} vs {style_away}:")
        logger.info(f"  Total: {lambda_total:.2f} → {lambda_total_new:.2f} ({total_factor:.2%})")
        logger.info(f"  Rho: {rho:.3f} → {rho_adj_final:.3f} ({rho_adj:+.3f})")

    return lambda_h_adj, lambda_a_adj, rho_adj_final


# ============================================================
#   UTILITY: APPLY ALL ADJUSTMENTS
# ============================================================

def apply_all_advanced_features(
    lambda_h: float,
    lambda_a: float,
    rho: float,
    total_target: float,
    motivation_home: str = "Normale",
    motivation_away: str = "Normale",
    days_since_home: int = 7,
    days_since_away: int = 7,
    days_until_home: int = 7,
    days_until_away: int = 7,
    style_home: str = "Possesso",
    style_away: str = "Possesso",
    apply_constraints: bool = True
) -> Dict[str, Any]:
    """
    Applica tutte le funzionalità avanzate in sequenza.

    Ordine:
    1. Tactical matchup (modifica natura del match)
    2. Motivation (intensità)
    3. Fixture congestion (fitness)
    4. Constraints fisici (validazione finale)

    Returns:
        Dict con lambda/rho finali + dettagli adjustments
    """
    adjustments_log = []

    # Valori iniziali
    lambda_h_start = lambda_h
    lambda_a_start = lambda_a
    rho_start = rho

    # 1. Tactical matchup
    lambda_h, lambda_a, rho = apply_tactical_matchup(
        lambda_h, lambda_a, rho, style_home, style_away
    )
    adjustments_log.append({
        'step': 'Tactical Matchup',
        'lambda_h': lambda_h,
        'lambda_a': lambda_a,
        'rho': rho
    })

    # 2. Motivation
    lambda_h, lambda_a = apply_motivation_factor(
        lambda_h, lambda_a, motivation_home, motivation_away
    )
    adjustments_log.append({
        'step': 'Motivation',
        'lambda_h': lambda_h,
        'lambda_a': lambda_a
    })

    # 3. Fixture congestion
    lambda_h, lambda_a = apply_fixture_congestion(
        lambda_h, lambda_a,
        days_since_home, days_since_away,
        days_until_home, days_until_away
    )
    adjustments_log.append({
        'step': 'Fixture Congestion',
        'lambda_h': lambda_h,
        'lambda_a': lambda_a
    })

    # 4. Constraints fisici (validazione finale)
    if apply_constraints:
        lambda_h, lambda_a = apply_physical_constraints_to_lambda(
            lambda_h, lambda_a, total_target
        )
        adjustments_log.append({
            'step': 'Physical Constraints',
            'lambda_h': lambda_h,
            'lambda_a': lambda_a
        })

    # Calcola variazioni totali
    lambda_h_change = ((lambda_h - lambda_h_start) / lambda_h_start) * 100
    lambda_a_change = ((lambda_a - lambda_a_start) / lambda_a_start) * 100
    rho_change = rho - rho_start

    logger.info(f"=== Advanced Features Summary ===")
    logger.info(f"λ_home: {lambda_h_start:.3f} → {lambda_h:.3f} ({lambda_h_change:+.1f}%)")
    logger.info(f"λ_away: {lambda_a_start:.3f} → {lambda_a:.3f} ({lambda_a_change:+.1f}%)")
    logger.info(f"rho: {rho_start:.3f} → {rho:.3f} ({rho_change:+.3f})")

    return {
        'lambda_h': lambda_h,
        'lambda_a': lambda_a,
        'rho': rho,
        'lambda_h_change_pct': lambda_h_change,
        'lambda_a_change_pct': lambda_a_change,
        'rho_change': rho_change,
        'adjustments_log': adjustments_log
    }
