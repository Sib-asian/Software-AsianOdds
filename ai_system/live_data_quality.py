"""
Sistema di Validazione e Quality Control per Dati Live
========================================================

Componenti principali:
1. LiveDataValidator - Validazione robusta dati live
2. AdvancedStatsCalculator - Calcolo statistiche avanzate
3. DynamicConfidenceCalculator - Confidence dinamico basato su dati reali
4. SignalQualityScorer - Scoring qualità segnale multi-fattoriale

Usage:
    validator = LiveDataValidator()
    is_valid, quality_score = validator.validate_and_score(live_data)

    if is_valid:
        stats_calc = AdvancedStatsCalculator()
        advanced_stats = stats_calc.calculate(live_data)

        conf_calc = DynamicConfidenceCalculator()
        confidence = conf_calc.calculate(opportunity, live_data, advanced_stats)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report qualità dati live"""
    is_valid: bool
    quality_score: float  # 0-100
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


@dataclass
class AdvancedStats:
    """Statistiche avanzate calcolate da dati live"""
    # Metriche di pericolosità
    attack_intensity_home: float  # 0-100
    attack_intensity_away: float
    shot_quality_home: float  # % tiri in porta su totali
    shot_quality_away: float

    # Metriche temporali (per minuto)
    shots_per_minute_home: float
    shots_per_minute_away: float
    goals_per_minute_home: float
    goals_per_minute_away: float
    xg_per_minute_home: float
    xg_per_minute_away: float

    # Metriche di dominanza
    possession_dominance: float  # -50 a +50 (+ = home domina)
    shot_dominance: float
    territorial_control: float  # Basato su attacks, corners, ecc.

    # Trend (ultimi 10-15 minuti)
    momentum_shift: float  # -1 a +1 (chi sta migliorando)
    pressure_trend: str  # 'increasing', 'stable', 'decreasing'

    # Qualità complessiva
    data_completeness: float  # % dati disponibili
    data_consistency: float  # % dati coerenti
    overall_reliability: float  # 0-100


class LiveDataValidator:
    """
    Validatore robusto per dati live.

    Controlla:
    - Coerenza (shots on target <= total shots, possession totale ~= 100%)
    - Range validi (minute: 0-90, possession: 0-100, ecc.)
    - Outlier (valori anomali)
    - Completezza dati
    """

    def __init__(self):
        """Inizializza validatore"""
        # Range validi per ogni campo
        self.valid_ranges = {
            'minute': (0, 120),  # Include extra time
            'score_home': (0, 15),  # Max ragionevole
            'score_away': (0, 15),
            'possession_home': (0, 100),
            'possession_away': (0, 100),
            'shots_home': (0, 50),
            'shots_away': (0, 50),
            'shots_on_target_home': (0, 30),
            'shots_on_target_away': (0, 30),
            'corners_home': (0, 25),
            'corners_away': (0, 25),
            'fouls_home': (0, 40),
            'fouls_away': (0, 40),
            'yellow_cards_home': (0, 7),
            'yellow_cards_away': (0, 7),
            'red_cards_home': (0, 2),
            'red_cards_away': (0, 2),
            'xg_home': (0.0, 10.0),
            'xg_away': (0.0, 10.0),
        }

        # Campi essenziali (devono esistere)
        self.essential_fields = ['minute', 'score_home', 'score_away']

        # Campi importanti (meglio averli)
        self.important_fields = [
            'possession_home', 'shots_home', 'shots_away',
            'shots_on_target_home', 'shots_on_target_away'
        ]

    def validate_and_score(self, live_data: Dict[str, Any]) -> DataQualityReport:
        """
        Valida dati live e calcola quality score.

        Returns:
            DataQualityReport con validazione e score
        """
        errors = []
        warnings = []
        metrics = {}

        if not live_data:
            return DataQualityReport(
                is_valid=False,
                quality_score=0.0,
                errors=["live_data è vuoto"],
                warnings=[],
                metrics={}
            )

        # 1. CHECK CAMPI ESSENZIALI
        for field in self.essential_fields:
            if field not in live_data or live_data[field] is None:
                errors.append(f"Campo essenziale mancante: {field}")

        if errors:
            return DataQualityReport(
                is_valid=False,
                quality_score=0.0,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

        # 2. VALIDAZIONE RANGE
        range_violations = self._check_ranges(live_data)
        errors.extend(range_violations)

        # 3. VALIDAZIONE COERENZA
        consistency_issues = self._check_consistency(live_data)
        warnings.extend(consistency_issues)

        # 4. OUTLIER DETECTION
        outliers = self._detect_outliers(live_data)
        if outliers:
            warnings.extend([f"Possibile outlier: {o}" for o in outliers])

        # 5. CALCOLO COMPLETEZZA
        completeness = self._calculate_completeness(live_data)
        metrics['completeness'] = completeness

        # 6. CALCOLO CONSISTENCY SCORE
        consistency_score = self._calculate_consistency_score(live_data)
        metrics['consistency'] = consistency_score

        # 7. QUALITY SCORE FINALE
        quality_score = self._calculate_quality_score(
            completeness, consistency_score, len(errors), len(warnings)
        )

        # Invalido se ci sono errori critici
        is_valid = len(errors) == 0 and quality_score >= 30.0

        return DataQualityReport(
            is_valid=is_valid,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def _check_ranges(self, live_data: Dict[str, Any]) -> List[str]:
        """Controlla che i valori siano nei range validi"""
        errors = []

        for field, (min_val, max_val) in self.valid_ranges.items():
            if field in live_data and live_data[field] is not None:
                value = live_data[field]
                try:
                    if isinstance(value, (int, float)):
                        if not (min_val <= value <= max_val):
                            errors.append(
                                f"{field}={value} fuori range ({min_val}-{max_val})"
                            )
                except (TypeError, ValueError):
                    errors.append(f"{field} ha valore non numerico: {value}")

        return errors

    def _check_consistency(self, live_data: Dict[str, Any]) -> List[str]:
        """Controlla coerenza tra campi correlati"""
        warnings = []

        # 1. Shots on target <= Total shots
        for team in ['home', 'away']:
            shots = live_data.get(f'shots_{team}', 0) or 0
            sot = live_data.get(f'shots_on_target_{team}', 0) or 0

            if sot > shots and shots > 0:
                warnings.append(
                    f"{team}: shots_on_target ({sot}) > total_shots ({shots})"
                )

        # 2. Possession totale ~= 100% (±5% tolleranza)
        poss_home = live_data.get('possession_home')
        poss_away = live_data.get('possession_away')

        if poss_home is not None and poss_away is not None:
            total_poss = poss_home + poss_away
            if not (95 <= total_poss <= 105):
                warnings.append(
                    f"Possesso totale = {total_poss}% (dovrebbe essere ~100%)"
                )

        # 3. Score ragionevole rispetto a minute
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        total_goals = score_home + score_away

        if minute > 0:
            goals_per_min = total_goals / minute
            # Più di 0.3 gol/minuto è anomalo (es: 3-3 al 10')
            if goals_per_min > 0.3:
                warnings.append(
                    f"Troppi gol per minuto: {total_goals} gol in {minute}' "
                    f"({goals_per_min:.2f} gol/min)"
                )

        # 4. xG ragionevole rispetto a shots on target
        for team in ['home', 'away']:
            xg_key = f'xg_{team}' if team == 'home' else f'xg_{team}'
            xg = live_data.get(xg_key, 0) or 0
            sot = live_data.get(f'shots_on_target_{team}', 0) or 0

            # xG > shots on target * 1.5 è sospetto
            if xg > 0 and sot > 0 and xg > sot * 1.5:
                warnings.append(
                    f"{team}: xG ({xg:.2f}) molto alto rispetto a SOT ({sot})"
                )

        # 5. Red cards <= Yellow cards (approssimativo)
        for team in ['home', 'away']:
            yellow = live_data.get(f'yellow_cards_{team}', 0) or 0
            red = live_data.get(f'red_cards_{team}', 0) or 0

            # Se hai 2 rossi ma 0 gialli, è strano
            if red > 0 and yellow == 0 and red > 1:
                warnings.append(
                    f"{team}: {red} cartellini rossi ma 0 gialli (insolito)"
                )

        return warnings

    def _detect_outliers(self, live_data: Dict[str, Any]) -> List[str]:
        """Rileva valori anomali/outlier"""
        outliers = []
        minute = live_data.get('minute', 0)

        if minute == 0:
            return outliers  # Non possiamo rilevare outlier senza minutaggio

        # 1. Troppi tiri per il minutaggio
        for team in ['home', 'away']:
            shots = live_data.get(f'shots_{team}', 0) or 0
            shots_per_min = shots / minute if minute > 0 else 0

            # > 0.8 tiri/minuto è tantissimo (es: 36 tiri in 45 minuti)
            if shots_per_min > 0.8:
                outliers.append(
                    f"{team}: {shots} tiri in {minute}' ({shots_per_min:.2f}/min)"
                )

        # 2. Troppi corner per il minutaggio
        for team in ['home', 'away']:
            corners = live_data.get(f'corners_{team}', 0) or 0
            corners_per_min = corners / minute if minute > 0 else 0

            # > 0.3 corner/minuto è tanto
            if corners_per_min > 0.3:
                outliers.append(
                    f"{team}: {corners} corner in {minute}' ({corners_per_min:.2f}/min)"
                )

        # 3. Possesso estremo nei primi minuti
        poss_home = live_data.get('possession_home')
        if poss_home is not None and minute < 20:
            # Possesso > 80% o < 20% nei primi 20' è insolito
            if poss_home > 80 or poss_home < 20:
                outliers.append(
                    f"Possesso estremo nei primi minuti: {poss_home:.0f}% al {minute}'"
                )

        return outliers

    def _calculate_completeness(self, live_data: Dict[str, Any]) -> float:
        """Calcola % di dati disponibili"""
        total_important = len(self.important_fields)
        available = sum(
            1 for field in self.important_fields
            if field in live_data and live_data[field] is not None
        )

        return (available / total_important) * 100 if total_important > 0 else 0

    def _calculate_consistency_score(self, live_data: Dict[str, Any]) -> float:
        """Calcola score di coerenza (0-100)"""
        issues = self._check_consistency(live_data)

        # Ogni issue riduce score di 15 punti
        penalty = min(len(issues) * 15, 70)
        return max(30, 100 - penalty)

    def _calculate_quality_score(
        self,
        completeness: float,
        consistency: float,
        num_errors: int,
        num_warnings: int
    ) -> float:
        """
        Calcola quality score finale (0-100).

        Formula:
        - Base: media(completeness, consistency)
        - Penalty: -30 per errore, -10 per warning
        """
        base_score = (completeness + consistency) / 2

        penalty = (num_errors * 30) + (num_warnings * 10)

        final_score = max(0, base_score - penalty)

        return min(100, final_score)


class AdvancedStatsCalculator:
    """
    Calcolatore statistiche avanzate da dati live.

    Calcola metriche derivate come:
    - Attack intensity (pericolosità attacchi)
    - Shot quality (% tiri in porta)
    - Metrics per minuto (shots/min, xG/min, ecc.)
    - Dominance metrics
    """

    def calculate(self, live_data: Dict[str, Any]) -> Optional[AdvancedStats]:
        """
        Calcola statistiche avanzate.

        Returns:
            AdvancedStats o None se dati insufficienti
        """
        if not live_data:
            return None

        minute = live_data.get('minute', 0)
        if minute == 0:
            return None  # Non possiamo calcolare per-minute stats

        # Estrai dati base
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        shots_home = live_data.get('shots_home', 0) or 0
        shots_away = live_data.get('shots_away', 0) or 0
        sot_home = live_data.get('shots_on_target_home', 0) or 0
        sot_away = live_data.get('shots_on_target_away', 0) or 0
        poss_home = live_data.get('possession_home', 50) or 50
        poss_away = live_data.get('possession_away', 50) or 50
        xg_home = live_data.get('xg_home', 0.0) or 0.0
        xg_away = live_data.get('xg_away', 0.0) or 0.0
        corners_home = live_data.get('corners_home', 0) or 0
        corners_away = live_data.get('corners_away', 0) or 0
        dangerous_home = live_data.get('dangerous_attacks_home', 0) or 0
        dangerous_away = live_data.get('dangerous_attacks_away', 0) or 0

        # === CALCOLA METRICHE PERICOLOSITÀ ===

        # Attack Intensity (0-100): combinazione di shots, xG, dangerous attacks
        attack_intensity_home = self._calculate_attack_intensity(
            shots_home, sot_home, xg_home, dangerous_home, minute
        )
        attack_intensity_away = self._calculate_attack_intensity(
            shots_away, sot_away, xg_away, dangerous_away, minute
        )

        # Shot Quality (% tiri in porta su totali)
        shot_quality_home = (sot_home / shots_home * 100) if shots_home > 0 else 0
        shot_quality_away = (sot_away / shots_away * 100) if shots_away > 0 else 0

        # === METRICHE PER MINUTO ===

        shots_per_min_home = shots_home / minute
        shots_per_min_away = shots_away / minute
        goals_per_min_home = score_home / minute
        goals_per_min_away = score_away / minute
        xg_per_min_home = xg_home / minute
        xg_per_min_away = xg_away / minute

        # === DOMINANCE METRICS ===

        # Possession dominance: -50 a +50
        poss_dominance = poss_home - poss_away

        # Shot dominance: normalizzato -100 a +100
        total_shots = shots_home + shots_away
        if total_shots > 0:
            shot_dominance = ((shots_home - shots_away) / total_shots) * 100
        else:
            shot_dominance = 0

        # Territorial control: basato su corners, attacks, possession
        territorial_control = self._calculate_territorial_control(
            poss_home, corners_home, corners_away, dangerous_home, dangerous_away
        )

        # === TREND (semplificato - richiede dati storici) ===

        # Per ora: momentum basato su differenza xG vs gol
        xg_diff_home = xg_home - score_home
        xg_diff_away = xg_away - score_away

        # Se xG > gol, squadra sta creando ma non finalizzando (momentum futuro)
        momentum_shift = 0.0
        if xg_diff_home > 1:
            momentum_shift += 0.3  # Home sta spingendo
        if xg_diff_away > 1:
            momentum_shift -= 0.3  # Away sta spingendo

        # Pressure trend: basato su tiri recenti (semplificato)
        total_shots_per_min = (shots_home + shots_away) / minute
        if total_shots_per_min > 0.5:
            pressure_trend = 'increasing'
        elif total_shots_per_min > 0.25:
            pressure_trend = 'stable'
        else:
            pressure_trend = 'decreasing'

        # === DATA QUALITY ===

        # Completeness: quanti campi important abbiamo?
        important_fields = [
            'possession_home', 'shots_home', 'shots_away',
            'shots_on_target_home', 'shots_on_target_away', 'xg_home', 'xg_away'
        ]
        available = sum(
            1 for field in important_fields
            if live_data.get(field) is not None and live_data.get(field) != 0
        )
        data_completeness = (available / len(important_fields)) * 100

        # Consistency: controllo base (SOT <= Shots)
        consistency_score = 100.0
        if sot_home > shots_home and shots_home > 0:
            consistency_score -= 30
        if sot_away > shots_away and shots_away > 0:
            consistency_score -= 30
        data_consistency = max(0, consistency_score)

        # Overall reliability: media pesata
        overall_reliability = (data_completeness * 0.6 + data_consistency * 0.4)

        return AdvancedStats(
            attack_intensity_home=attack_intensity_home,
            attack_intensity_away=attack_intensity_away,
            shot_quality_home=shot_quality_home,
            shot_quality_away=shot_quality_away,
            shots_per_minute_home=shots_per_min_home,
            shots_per_minute_away=shots_per_min_away,
            goals_per_minute_home=goals_per_min_home,
            goals_per_minute_away=goals_per_min_away,
            xg_per_minute_home=xg_per_min_home,
            xg_per_minute_away=xg_per_min_away,
            possession_dominance=poss_dominance,
            shot_dominance=shot_dominance,
            territorial_control=territorial_control,
            momentum_shift=momentum_shift,
            pressure_trend=pressure_trend,
            data_completeness=data_completeness,
            data_consistency=data_consistency,
            overall_reliability=overall_reliability
        )

    def _calculate_attack_intensity(
        self,
        shots: int,
        sot: int,
        xg: float,
        dangerous: int,
        minute: int
    ) -> float:
        """
        Calcola intensità attacco (0-100).

        Formula pesata:
        - 30% shots per minute
        - 30% shots on target per minute
        - 25% xG per minute
        - 15% dangerous attacks per minute
        """
        if minute == 0:
            return 0.0

        # Normalizza su scale 0-1
        shots_per_min = shots / minute
        sot_per_min = sot / minute
        xg_per_min = xg / minute
        dangerous_per_min = dangerous / minute

        # Normalizza (valori "ottimi" per una squadra molto pericolosa)
        # Shots: 0.5/min = 100%
        shots_score = min(1.0, shots_per_min / 0.5) * 30

        # SOT: 0.25/min = 100%
        sot_score = min(1.0, sot_per_min / 0.25) * 30

        # xG: 0.05/min (4.5 xG in 90') = 100%
        xg_score = min(1.0, xg_per_min / 0.05) * 25

        # Dangerous: 1.0/min = 100%
        dangerous_score = min(1.0, dangerous_per_min / 1.0) * 15

        total = shots_score + sot_score + xg_score + dangerous_score

        return min(100, total)

    def _calculate_territorial_control(
        self,
        poss_home: float,
        corners_home: int,
        corners_away: int,
        dangerous_home: int,
        dangerous_away: int
    ) -> float:
        """
        Calcola controllo territoriale home (0-100).

        Combinazione di:
        - Possession
        - Corner difference
        - Dangerous attacks difference
        """
        # Base: possession
        base = poss_home

        # Bonus corner (max ±10)
        total_corners = corners_home + corners_away
        if total_corners > 0:
            corner_diff = (corners_home - corners_away) / total_corners
            corner_bonus = corner_diff * 10
        else:
            corner_bonus = 0

        # Bonus dangerous (max ±10)
        total_dangerous = dangerous_home + dangerous_away
        if total_dangerous > 0:
            dangerous_diff = (dangerous_home - dangerous_away) / total_dangerous
            dangerous_bonus = dangerous_diff * 10
        else:
            dangerous_bonus = 0

        total = base + corner_bonus + dangerous_bonus

        return max(0, min(100, total))


class DynamicConfidenceCalculator:
    """
    Calcola confidence dinamicamente basandosi su:
    - Qualità dei dati
    - Statistiche avanzate
    - Situazione partita
    - Tipo di mercato

    NON usa valori fissi hardcoded!
    """

    def calculate(
        self,
        market_type: str,
        situation: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats],
        data_quality: Optional[DataQualityReport]
    ) -> float:
        """
        ✅ RISCRITTURA COMPLETA: Calcola confidence dinamico con logica corretta.

        APPROCCIO CORRETTO:
        1. Calcola probabilità BASE dalla SITUAZIONE di gioco (50-80%)
        2. Adatta questa probabilità con un FATTORE di qualità dati (0.7x - 1.15x)
        3. Risultato: confidence realistico allineato alle quote

        NON più somma percentuali (60% + 20% + 10% = 90% anche se situazione è pessima!)

        Args:
            market_type: Tipo mercato (over_2.5, 1x2, ecc.)
            situation: Tipo situazione (ribaltone, under_opportunity, ecc.)
            live_data: Dati live raw
            advanced_stats: Statistiche avanzate
            data_quality: Report qualità dati

        Returns:
            Confidence score 0-100
        """
        # Validazione dati
        if data_quality is None or not data_quality.is_valid:
            return 0.0

        # 1️⃣ PRIMA: Calcola probabilità BASE dalla SITUAZIONE reale
        base_probability = self._calculate_base_probability(
            market_type, situation, live_data, advanced_stats
        )

        # 2️⃣ POI: Calcola fattore di fiducia dai dati (0.7 - 1.15)
        quality_factor = self._calculate_quality_trust_factor(
            data_quality, advanced_stats
        )

        # 3️⃣ ADATTA: Moltiplica probabilità per fiducia
        # Se dati ottimi (quality_factor=1.15): 60% → 69%
        # Se dati scarsi (quality_factor=0.7): 60% → 42%
        adjusted_confidence = base_probability * quality_factor

        # Clamp 0-100
        return max(0.0, min(100.0, adjusted_confidence))

    def _calculate_quality_trust_factor(
        self,
        data_quality: DataQualityReport,
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """
        Calcola fattore di fiducia dai dati (0.7 - 1.15).

        Se dati ottimi: possiamo fidarci di più → factor > 1.0 (boost)
        Se dati scarsi: dobbiamo essere cauti → factor < 1.0 (penalty)

        Returns:
            Factor moltiplicativo: 0.7 (dati scarsi) - 1.15 (dati ottimi)
        """
        # Base: qualità dati (contributo principale)
        # Quality 100 → 1.0, Quality 50 → 0.85, Quality 0 → 0.7
        base_factor = 0.7 + (data_quality.quality_score / 100) * 0.3

        # Bonus da reliability e completeness (piccolo contributo)
        bonus = 0.0
        if advanced_stats:
            # Reliability alta → +0.08 max
            if advanced_stats.overall_reliability > 80:
                bonus += 0.08
            elif advanced_stats.overall_reliability > 60:
                bonus += 0.04

            # Completeness alta → +0.07 max
            if advanced_stats.data_completeness > 80:
                bonus += 0.07
            elif advanced_stats.data_completeness > 60:
                bonus += 0.03

        final_factor = base_factor + bonus

        # Clamp 0.7 - 1.15
        return max(0.7, min(1.15, final_factor))

    def _calculate_base_probability(
        self,
        market_type: str,
        situation: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """
        ✅ CUORE DEL SISTEMA: Calcola probabilità BASE dalla situazione reale.

        Questo metodo risponde alla domanda:
        "Data la situazione attuale di gioco, quanto è probabile questo outcome?"

        NON considera qualità dati (quello viene dopo).
        Si basa solo su: score, minuto, statistiche, tipo mercato.

        Returns:
            Probabilità base 0-100 (tipicamente 40-80%)
        """
        market = market_type.lower()

        # Routing verso metodi specifici per mercato
        if 'over' in market and 'under' not in market:
            return self._calc_over_probability(market, live_data, advanced_stats)

        elif 'under' in market:
            return self._calc_under_probability(market, live_data, advanced_stats)

        elif any(x in market for x in ['1x2', 'home_win', 'away_win']) or 'ribaltone' in situation:
            return self._calc_win_probability(market, situation, live_data, advanced_stats)

        elif 'next_goal' in market:
            return self._calc_next_goal_probability(market, live_data, advanced_stats)

        elif 'btts' in market:
            return self._calc_btts_probability(market, live_data, advanced_stats)

        elif 'clean_sheet' in market:
            return self._calc_clean_sheet_probability(market, live_data, advanced_stats)

        else:
            # Default: situazione neutra
            return 55.0

    def _calc_over_probability(
        self,
        market: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità Over basata su situazione reale"""
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        total_goals = score_home + score_away

        # Estrai threshold (over_1.5 → 1.5)
        threshold = self._extract_threshold(market)
        if threshold is None:
            return 55.0

        goals_needed = max(0, int(threshold) + 1 - total_goals)
        time_remaining = max(0, 90 - minute)

        # Già superato? 100%
        if total_goals > threshold:
            return 100.0

        # Tempo insufficiente per gol necessari? Bassa probabilità
        if time_remaining < 5 and goals_needed >= 1:
            return 15.0
        if time_remaining < 15 and goals_needed >= 2:
            return 25.0

        # Calcolo basato su attack intensity
        if advanced_stats:
            avg_intensity = (
                advanced_stats.attack_intensity_home +
                advanced_stats.attack_intensity_away
            ) / 2

            # Expected goals per minute
            # Intensity 70 = ~0.03 gol/min (2.7 gol in 90 min)
            # Intensity 50 = ~0.02 gol/min (1.8 gol in 90 min)
            goals_per_min = (avg_intensity / 100) * 0.035
            expected_goals = goals_per_min * time_remaining

            # Probabilità basata su expected goals
            if goals_needed == 1:
                # Serve 1 gol: alta prob se expected > 1
                prob = min(80, 35 + expected_goals * 35)
            elif goals_needed == 2:
                # Servono 2 gol
                prob = min(70, 30 + expected_goals * 20)
            else:
                # Servono 3+ gol
                prob = min(60, 25 + expected_goals * 12)

            return prob
        else:
            # Fallback senza advanced stats
            if goals_needed == 0:
                return 95.0
            elif goals_needed == 1 and time_remaining > 35:
                return 55.0
            elif goals_needed == 1:
                return 40.0
            elif goals_needed == 2 and time_remaining > 50:
                return 45.0
            else:
                return 30.0

    def _calc_under_probability(
        self,
        market: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità Under"""
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        total_goals = score_home + score_away

        threshold = self._extract_threshold(market)
        if threshold is None:
            return 55.0

        goals_allowed = int(threshold) - total_goals
        time_remaining = max(0, 90 - minute)

        # Già superato? 0%
        if total_goals > threshold:
            return 0.0

        # Pochi minuti e serve 0 gol? Alta probabilità
        if time_remaining < 10 and goals_allowed == 0:
            return 85.0

        # Calcolo basato su attack intensity (BASSA intensity = under più probabile)
        if advanced_stats:
            avg_intensity = (
                advanced_stats.attack_intensity_home +
                advanced_stats.attack_intensity_away
            ) / 2

            goals_per_min = (avg_intensity / 100) * 0.035
            expected_goals = goals_per_min * time_remaining

            # Se expected goals < goals_allowed → under probabile
            if expected_goals < goals_allowed * 0.5:
                # Molto pochi gol attesi
                prob = 75.0
            elif expected_goals < goals_allowed * 0.8:
                prob = 65.0
            elif expected_goals < goals_allowed:
                prob = 55.0
            else:
                # Troppi gol attesi → under improbabile
                prob = 40.0

            return prob
        else:
            # Fallback
            if time_remaining < 20 and goals_allowed >= 1:
                return 70.0
            elif goals_allowed >= 2:
                return 60.0
            else:
                return 50.0

    def _calc_win_probability(
        self,
        market: str,
        situation: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità vittoria/ribaltone"""
        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        time_remaining = max(0, 90 - minute)

        # Ribaltone: favorita in svantaggio
        if 'ribaltone' in situation:
            goal_diff = abs(score_home - score_away)

            # Differenza troppo alta? Improbabile
            if goal_diff >= 3:
                return 20.0
            elif goal_diff == 2:
                base_prob = 35.0
            else:
                base_prob = 55.0

            # Poco tempo? Riduce probabilità
            if time_remaining < 20:
                base_prob *= 0.7
            elif time_remaining < 40:
                base_prob *= 0.85

            # Se favorita domina, aumenta
            if advanced_stats and abs(advanced_stats.possession_dominance) > 15:
                base_prob += 10

            return min(75.0, base_prob)
        else:
            # Normale win: dipende da score attuale
            if 'home' in market:
                if score_home > score_away:
                    return 75.0
                elif score_home == score_away:
                    return 50.0
                else:
                    return 35.0
            elif 'away' in market:
                if score_away > score_home:
                    return 75.0
                elif score_home == score_away:
                    return 50.0
                else:
                    return 35.0
            else:
                return 50.0

    def _calc_next_goal_probability(
        self,
        market: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità prossimo gol"""
        if not advanced_stats:
            return 52.0

        # Basato su attack intensity
        intensity_home = advanced_stats.attack_intensity_home
        intensity_away = advanced_stats.attack_intensity_away

        if 'home' in market:
            # Prossimo gol casa
            if intensity_home > intensity_away * 1.5:
                return 70.0
            elif intensity_home > intensity_away * 1.2:
                return 62.0
            elif intensity_home > intensity_away:
                return 55.0
            else:
                return 45.0
        elif 'away' in market:
            # Prossimo gol trasferta
            if intensity_away > intensity_home * 1.5:
                return 70.0
            elif intensity_away > intensity_home * 1.2:
                return 62.0
            elif intensity_away > intensity_home:
                return 55.0
            else:
                return 45.0
        else:
            return 50.0

    def _calc_btts_probability(
        self,
        market: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità BTTS (Both Teams To Score)"""
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        minute = live_data.get('minute', 0)
        time_remaining = max(0, 90 - minute)

        # BTTS Yes
        if 'yes' in market or 'btts' == market:
            # Entrambi hanno già segnato? 100%
            if score_home > 0 and score_away > 0:
                return 100.0

            # Uno ha segnato, manca l'altro
            if score_home > 0 or score_away > 0:
                # Tempo sufficiente?
                if time_remaining > 30:
                    return 60.0
                elif time_remaining > 15:
                    return 45.0
                else:
                    return 30.0
            else:
                # Nessuno ha segnato
                if time_remaining > 45:
                    return 50.0
                else:
                    return 35.0
        else:
            # BTTS No
            if score_home > 0 and score_away > 0:
                return 0.0
            elif time_remaining < 10:
                return 80.0
            elif time_remaining < 25:
                return 65.0
            else:
                return 50.0

    def _calc_clean_sheet_probability(
        self,
        market: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """Calcola probabilità porta inviolata"""
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        minute = live_data.get('minute', 0)
        time_remaining = max(0, 90 - minute)

        if 'home' in market:
            # Porta inviolata casa (away non segna)
            if score_away > 0:
                return 0.0
            elif time_remaining < 10:
                return 80.0
            elif time_remaining < 25:
                return 65.0
            else:
                return 50.0
        elif 'away' in market:
            # Porta inviolata trasferta (home non segna)
            if score_home > 0:
                return 0.0
            elif time_remaining < 10:
                return 80.0
            elif time_remaining < 25:
                return 65.0
            else:
                return 50.0
        else:
            return 50.0

    def _extract_threshold(self, market: str) -> Optional[float]:
        """Estrae soglia numerica da mercato (over_1.5 → 1.5)"""
        import re
        match = re.search(r'(\d+\.?\d*)', market)
        if match:
            return float(match.group(1))
        return None


class SignalQualityScorer:
    """
    Sistema di scoring multi-fattoriale per qualità segnale.

    Assegna punteggio 0-100 considerando:
    - Data quality
    - Statistical significance
    - Timing appropriateness
    - Uniqueness (non banale)
    - Market-situation fit
    """

    def __init__(self):
        """Inizializza scorer"""
        self.data_validator = LiveDataValidator()
        self.stats_calculator = AdvancedStatsCalculator()
        self.conf_calculator = DynamicConfidenceCalculator()

    def score_signal(
        self,
        market_type: str,
        situation: str,
        live_data: Dict[str, Any],
        confidence: float,
        ev: float
    ) -> Dict[str, Any]:
        """
        Calcola quality score completo per un segnale.

        Returns:
            Dict con score totale e breakdown
        """
        # 1. Data Quality (30 punti)
        data_quality = self.data_validator.validate_and_score(live_data)
        data_score = (data_quality.quality_score / 100) * 30

        # 2. Statistical Significance (25 punti)
        advanced_stats = self.stats_calculator.calculate(live_data)
        if advanced_stats:
            stat_score = (advanced_stats.overall_reliability / 100) * 25
        else:
            stat_score = 0

        # 3. Confidence Level (20 punti)
        conf_score = (confidence / 100) * 20

        # 4. Expected Value (15 punti)
        ev_score = min(15, (ev / 20) * 15)  # EV 20% = full score

        # 5. Uniqueness / Non-Banality (10 punti)
        uniqueness_score = self._calculate_uniqueness(
            market_type, situation, live_data, advanced_stats
        )

        # Total score
        total_score = (
            data_score + stat_score + conf_score + ev_score + uniqueness_score
        )

        return {
            'total_score': round(total_score, 1),
            'breakdown': {
                'data_quality': round(data_score, 1),
                'statistical_significance': round(stat_score, 1),
                'confidence': round(conf_score, 1),
                'expected_value': round(ev_score, 1),
                'uniqueness': round(uniqueness_score, 1)
            },
            'pass_threshold': total_score >= 60.0,  # Soglia minima 60/100
            'grade': self._get_grade(total_score)
        }

    def _calculate_uniqueness(
        self,
        market_type: str,
        situation: str,
        live_data: Dict[str, Any],
        advanced_stats: Optional[AdvancedStats]
    ) -> float:
        """
        Calcola quanto il segnale è "unico" e non banale.

        Penalizza situazioni ovvie tipo:
        - "Over 0.5 con score 2-1"
        - "Favorita vince quando già vince 2-0"
        """
        score = 10.0  # Base

        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0) or 0
        score_away = live_data.get('score_away', 0) or 0
        total_goals = score_home + score_away

        market = market_type.lower()

        # PENALITÀ SEGNALI BANALI

        # Over 0.5 con già gol segnati = banale
        if 'over_0.5' in market and total_goals >= 1:
            score -= 8

        # Over 1.5 con già 2+ gol = banale
        if 'over_1.5' in market and total_goals >= 2:
            score -= 7

        # Under 2.5 con 0-0 al 85' = troppo ovvio
        if 'under' in market and total_goals == 0 and minute > 85:
            score -= 6

        # "Prossimo gol favorita" quando favorita sta già vincendo 2-0 = banale
        if 'next_goal' in market and abs(score_home - score_away) >= 2:
            score -= 5

        # BONUS SEGNALI INTERESSANTI

        # Situazione equilibrata late game = interessante
        if minute > 70 and abs(score_home - score_away) <= 1:
            score += 3

        # Advanced stats mostrano dominanza nascosta = interessante
        if advanced_stats:
            # Squadra domina ma è in svantaggio
            if advanced_stats.possession_dominance > 20 and score_home < score_away:
                score += 4
            elif advanced_stats.possession_dominance < -20 and score_away < score_home:
                score += 4

        return max(0, min(10, score))

    def _get_grade(self, score: float) -> str:
        """Assegna grade letterale"""
        if score >= 85:
            return 'A+ (Excellent)'
        elif score >= 75:
            return 'A (Very Good)'
        elif score >= 65:
            return 'B (Good)'
        elif score >= 55:
            return 'C (Fair)'
        else:
            return 'D (Poor)'
