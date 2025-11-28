"""
Advanced Signal Metrics Module
Fornisce metriche granulari per valutare la qualità dei segnali live
"""

import logging
from typing import Dict, Optional, List, Tuple
import re

logger = logging.getLogger(__name__)


class AdvancedSignalMetrics:
    """
    Calcola metriche avanzate per valutare la coerenza e qualità dei segnali live
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_offensive_pressure(self, team_stats: Dict) -> float:
        """
        Calcola la pressione offensiva di una squadra combinando multiple metriche

        Args:
            team_stats: Dict con shots, shots_on_target, xg, dangerous_attacks, corners

        Returns:
            Score normalizzato 0.0-1.0
        """
        shots = team_stats.get('shots', 0)
        shots_on_target = team_stats.get('shots_on_target', 0)
        xg = team_stats.get('xg', 0.0)
        dangerous_attacks = team_stats.get('dangerous_attacks', 0)
        corners = team_stats.get('corners', 0)

        # Weights per ogni componente
        pressure_score = (
            shots * 0.25 +
            shots_on_target * 0.30 +
            xg * 10 * 0.25 +  # xG è già 0-3+, moltiplichiamo per dare peso
            dangerous_attacks * 0.10 +
            corners * 0.10
        )

        # Normalizza a 0-1 (assumendo max realistico = 100)
        normalized = min(pressure_score / 100.0, 1.0)

        return normalized

    def calculate_xg_trend(self, current_xg: float, previous_xg: float = None) -> Tuple[str, float]:
        """
        Analizza il trend di xG (se disponibile storico)

        Args:
            current_xg: xG attuale
            previous_xg: xG precedente (opzionale, per calcolo trend)

        Returns:
            Tuple (trend_label, trend_score)
            - trend_label: 'accelerating', 'growing', 'stable', 'declining'
            - trend_score: 0.8-1.2 (moltiplicatore)
        """
        if previous_xg is None or previous_xg == 0:
            # Nessun dato storico, valutiamo solo valore assoluto
            if current_xg > 2.0:
                return ('high', 1.15)
            elif current_xg > 1.0:
                return ('medium', 1.0)
            else:
                return ('low', 0.90)

        delta = current_xg - previous_xg

        if delta > 0.5:
            return ('accelerating', 1.20)
        elif delta > 0.2:
            return ('growing', 1.10)
        elif delta < -0.2:
            return ('declining', 0.85)
        else:
            return ('stable', 1.0)

    def calculate_momentum_score(self, recent_events: List[Dict] = None) -> float:
        """
        Calcola il momentum basato su eventi recenti (ultimi 5-10 minuti)

        Args:
            recent_events: Lista di eventi recenti (opzionale per ora)

        Returns:
            Momentum score 0.0-1.0
        """
        # Per ora ritorniamo 0.5 (neutrale) dato che gli eventi recenti
        # non sono sempre disponibili in live_data
        # TODO: Integrare quando abbiamo storico eventi
        if not recent_events:
            return 0.5

        momentum = 0
        for event in recent_events:
            event_type = event.get('type', '')

            if event_type == 'corner':
                momentum += 5
            elif event_type == 'shot_on_target':
                momentum += 10
            elif event_type == 'yellow_card':
                momentum -= 3
            elif event_type == 'goal':
                momentum += 20
            elif event_type == 'dangerous_attack':
                momentum += 3

        # Normalizza -50 to +50 → 0.0 to 1.0
        normalized = (momentum + 50) / 100.0
        return max(0.0, min(1.0, normalized))

    def calculate_possession_dominance(self, possession_home: float, possession_away: float) -> Tuple[str, float]:
        """
        Valuta il livello di dominanza del possesso

        Returns:
            Tuple (dominance_team, dominance_score)
            - dominance_team: 'home', 'away', 'balanced'
            - dominance_score: 0.0-1.0 (quanto è dominante)
        """
        diff = abs(possession_home - possession_away)

        if diff >= 20:
            team = 'home' if possession_home > possession_away else 'away'
            return (team, 0.8)
        elif diff >= 10:
            team = 'home' if possession_home > possession_away else 'away'
            return (team, 0.6)
        else:
            return ('balanced', 0.3)

    def calculate_shots_efficiency(self, shots: int, shots_on_target: int, xg: float) -> float:
        """
        Calcola l'efficienza dei tiri (qualità vs quantità)

        Returns:
            Efficiency score 0.0-1.0
        """
        if shots == 0:
            return 0.0

        # On-target rate
        on_target_rate = shots_on_target / shots if shots > 0 else 0

        # xG per shot (qualità media)
        xg_per_shot = xg / shots if shots > 0 else 0

        # Combina: 60% on-target rate, 40% xG quality
        efficiency = (on_target_rate * 0.60) + (min(xg_per_shot, 0.5) * 2 * 0.40)

        return min(efficiency, 1.0)

    def evaluate_market_coherence(self, market: str, live_data: Dict, opportunity_data: Dict) -> Tuple[float, List[str]]:
        """
        Valuta la coerenza tra il mercato predetto e le statistiche live

        Args:
            market: Nome del mercato (es. "over_2.5", "home", "btts_yes")
            live_data: Dati live della partita
            opportunity_data: Dati dell'opportunità (include confidence, ev, etc.)

        Returns:
            Tuple (coherence_score, warnings)
            - coherence_score: 0.0-1.0 (1.0 = perfetta coerenza)
            - warnings: Lista di warning strings
        """
        coherence = 1.0
        warnings = []

        minute = live_data.get('minute', 0)
        score_home = live_data.get('score_home', 0)
        score_away = live_data.get('score_away', 0)
        current_goals = score_home + score_away

        xg_home = live_data.get('xg_home', 0.0)
        xg_away = live_data.get('xg_away', 0.0)
        total_xg = xg_home + xg_away

        shots_home = live_data.get('shots_home', 0)
        shots_away = live_data.get('shots_away', 0)
        total_shots = shots_home + shots_away

        possession_home = live_data.get('possession_home', 50.0)
        possession_away = live_data.get('possession_away', 50.0)

        market_lower = market.lower()

        # === OVER/UNDER MARKETS ===
        if 'over' in market_lower or 'under' in market_lower:
            threshold_match = re.search(r'(\d+\.?\d*)', market_lower)
            if threshold_match:
                threshold = float(threshold_match.group(1))

                # Calcola goal rimanenti necessari
                remaining_goals_needed = threshold - current_goals

                if 'over' in market_lower:
                    # Over: xG dovrebbe essere alto
                    expected_remaining_xg = (total_xg / max(minute, 1)) * (90 - minute)

                    if total_xg < threshold * 0.5 and minute > 30:
                        coherence -= 0.20
                        warnings.append(f"xG ({total_xg:.1f}) basso per over {threshold}")

                    if total_shots < 8 and minute > 30:
                        coherence -= 0.10
                        warnings.append(f"Pochi tiri ({total_shots}) per over {threshold}")

                    if remaining_goals_needed > 2 and minute > 70:
                        coherence -= 0.15
                        warnings.append(f"Servono {remaining_goals_needed} goal in {90-minute} min")

                elif 'under' in market_lower:
                    # Under: xG dovrebbe essere basso
                    if total_xg > threshold * 0.8 and minute > 30:
                        coherence -= 0.20
                        warnings.append(f"xG ({total_xg:.1f}) alto per under {threshold}")

                    if total_shots > 20 and minute > 30:
                        coherence -= 0.10
                        warnings.append(f"Troppi tiri ({total_shots}) per under {threshold}")

        # === BTTS (Both Teams To Score) ===
        elif 'btts' in market_lower:
            if 'yes' in market_lower:
                # Entrambe devono avere potenziale offensivo
                if xg_home < 0.3 or xg_away < 0.3:
                    coherence -= 0.15
                    warnings.append(f"xG basso per BTTS (H:{xg_home:.1f}, A:{xg_away:.1f})")

                if shots_home < 3 or shots_away < 3:
                    coherence -= 0.10
                    warnings.append(f"Pochi tiri per BTTS (H:{shots_home}, A:{shots_away})")

            elif 'no' in market_lower:
                # Una squadra dovrebbe essere molto debole offensivamente
                if xg_home > 1.0 and xg_away > 1.0:
                    coherence -= 0.20
                    warnings.append(f"Entrambe xG alte per BTTS No (H:{xg_home:.1f}, A:{xg_away:.1f})")

        # === HOME/AWAY WIN ===
        elif market_lower in ['home', '1x2_home', 'home_win']:
            # Home dovrebbe dominare
            if possession_home < 45:
                coherence -= 0.15
                warnings.append(f"Possesso home basso ({possession_home:.0f}%) per home win")

            if xg_home < xg_away:
                coherence -= 0.20
                warnings.append(f"xG home < away ({xg_home:.1f} vs {xg_away:.1f})")

            if shots_home < shots_away * 0.8:
                coherence -= 0.10
                warnings.append(f"Tiri home < away ({shots_home} vs {shots_away})")

        elif market_lower in ['away', '1x2_away', 'away_win']:
            # Away dovrebbe dominare
            if possession_away < 45:
                coherence -= 0.15
                warnings.append(f"Possesso away basso ({possession_away:.0f}%) per away win")

            if xg_away < xg_home:
                coherence -= 0.20
                warnings.append(f"xG away < home ({xg_away:.1f} vs {xg_home:.1f})")

            if shots_away < shots_home * 0.8:
                coherence -= 0.10
                warnings.append(f"Tiri away < home ({shots_away} vs {shots_home})")

        # === CLEAN SHEET ===
        elif 'clean' in market_lower:
            team = 'home' if 'home' in market_lower else 'away'
            opponent_xg = xg_away if team == 'home' else xg_home
            opponent_shots = shots_away if team == 'home' else shots_home

            if opponent_xg > 1.0:
                coherence -= 0.25
                warnings.append(f"xG avversario alto ({opponent_xg:.1f}) per clean sheet")

            if opponent_shots > 8:
                coherence -= 0.15
                warnings.append(f"Tiri avversario alti ({opponent_shots}) per clean sheet")

        return (max(coherence, 0.0), warnings)

    def get_comprehensive_metrics(self, live_data: Dict, market: str = None,
                                  opportunity_data: Dict = None) -> Dict:
        """
        Calcola tutte le metriche avanzate in un colpo solo

        Returns:
            Dict con tutte le metriche calcolate
        """
        metrics = {}

        # Stats individuali squadre
        home_stats = {
            'shots': live_data.get('shots_home', 0),
            'shots_on_target': live_data.get('shots_on_target_home', 0),
            'xg': live_data.get('xg_home', 0.0),
            'dangerous_attacks': live_data.get('dangerous_attacks_home', 0),
            'corners': live_data.get('corners_home', 0)
        }

        away_stats = {
            'shots': live_data.get('shots_away', 0),
            'shots_on_target': live_data.get('shots_on_target_away', 0),
            'xg': live_data.get('xg_away', 0.0),
            'dangerous_attacks': live_data.get('dangerous_attacks_away', 0),
            'corners': live_data.get('corners_away', 0)
        }

        # Pressione offensiva
        metrics['offensive_pressure_home'] = self.calculate_offensive_pressure(home_stats)
        metrics['offensive_pressure_away'] = self.calculate_offensive_pressure(away_stats)

        # Trend xG
        xg_trend_home, xg_score_home = self.calculate_xg_trend(home_stats['xg'])
        xg_trend_away, xg_score_away = self.calculate_xg_trend(away_stats['xg'])
        metrics['xg_trend_home'] = xg_trend_home
        metrics['xg_score_home'] = xg_score_home
        metrics['xg_trend_away'] = xg_trend_away
        metrics['xg_score_away'] = xg_score_away

        # Possesso
        possession_home = live_data.get('possession_home', 50.0)
        possession_away = live_data.get('possession_away', 50.0)
        dominance_team, dominance_score = self.calculate_possession_dominance(possession_home, possession_away)
        metrics['possession_dominance_team'] = dominance_team
        metrics['possession_dominance_score'] = dominance_score

        # Efficienza tiri
        metrics['shots_efficiency_home'] = self.calculate_shots_efficiency(
            home_stats['shots'], home_stats['shots_on_target'], home_stats['xg']
        )
        metrics['shots_efficiency_away'] = self.calculate_shots_efficiency(
            away_stats['shots'], away_stats['shots_on_target'], away_stats['xg']
        )

        # Coerenza mercato (se fornito)
        if market and opportunity_data:
            coherence_score, coherence_warnings = self.evaluate_market_coherence(
                market, live_data, opportunity_data
            )
            metrics['market_coherence'] = coherence_score
            metrics['coherence_warnings'] = coherence_warnings

        return metrics
