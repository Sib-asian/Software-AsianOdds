"""
Sistema Auto-Ottimizzazione Parametri
======================================

Analizza performance storiche e trova parametri ottimali automaticamente.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Ottimizza parametri automaticamente basandosi su performance storiche.
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_parameters(
        self,
        betting_results: List[Dict[str, Any]],
        current_params: Dict[str, Any],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Trova parametri ottimali basandosi su performance.
        
        Args:
            betting_results: Lista risultati scommesse
            current_params: Parametri attuali (min_ev, min_confidence, etc)
            days: Giorni da analizzare
            
        Returns:
            Dict con parametri ottimali suggeriti e analisi
        """
        try:
            if not betting_results:
                return {
                    'suggested_params': current_params,
                    'improvement_estimate': 0.0,
                    'analysis': 'Nessun dato disponibile',
                    'confidence': 'LOW'
                }
            
            # Filtra per periodo
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_results = [
                r for r in betting_results
                if isinstance(r.get('timestamp'), datetime) and r['timestamp'] >= cutoff_date
            ]
            
            if len(recent_results) < 10:
                return {
                    'suggested_params': current_params,
                    'improvement_estimate': 0.0,
                    'analysis': f'Dati insufficienti (solo {len(recent_results)} risultati)',
                    'confidence': 'LOW'
                }
            
            # Analizza performance attuale
            current_perf = self._calculate_performance(recent_results)
            
            # Testa diverse combinazioni di parametri
            test_configs = self._generate_test_configs(current_params)
            
            best_config = None
            best_perf = current_perf
            improvements = []
            
            for config in test_configs:
                # Simula performance con questi parametri
                simulated_perf = self._simulate_performance(recent_results, config)
                
                if simulated_perf['roi'] > best_perf['roi']:
                    improvement = simulated_perf['roi'] - best_perf['roi']
                    improvements.append({
                        'config': config,
                        'performance': simulated_perf,
                        'improvement': improvement
                    })
                    
                    if simulated_perf['roi'] > best_perf['roi']:
                        best_config = config
                        best_perf = simulated_perf
            
            # Se trovato miglioramento significativo
            if best_config and best_perf['roi'] > current_perf['roi'] + 2.0:
                improvement_pct = ((best_perf['roi'] - current_perf['roi']) / abs(current_perf['roi']) * 100) if current_perf['roi'] != 0 else 0
                
                return {
                    'suggested_params': best_config,
                    'current_params': current_params,
                    'current_performance': current_perf,
                    'projected_performance': best_perf,
                    'improvement_estimate': improvement_pct,
                    'analysis': self._generate_analysis(current_perf, best_perf, best_config),
                    'confidence': 'MEDIUM' if len(recent_results) >= 20 else 'LOW',
                    'recommendation': 'UPDATE' if improvement_pct > 5 else 'MONITOR'
                }
            else:
                return {
                    'suggested_params': current_params,
                    'current_performance': current_perf,
                    'improvement_estimate': 0.0,
                    'analysis': 'Parametri attuali sono ottimali o dati insufficienti',
                    'confidence': 'MEDIUM',
                    'recommendation': 'KEEP'
                }
                
        except Exception as e:
            logger.error(f"❌ Error optimizing parameters: {e}", exc_info=True)
            return {
                'suggested_params': current_params,
                'error': str(e),
                'confidence': 'LOW'
            }
    
    def _calculate_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Calcola performance attuale."""
        total_bets = len(results)
        wins = sum(1 for r in results if r.get('outcome') == 'W')
        losses = sum(1 for r in results if r.get('outcome') == 'L')
        
        total_stake = sum(r.get('stake', 0) for r in results)
        total_profit = sum(
            (r.get('stake', 0) * (r.get('odds', 1) - 1)) if r.get('outcome') == 'W'
            else -r.get('stake', 0)
            for r in results
        )
        
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_profit': total_profit,
            'roi': roi
        }
    
    def _generate_test_configs(self, current: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera configurazioni di test."""
        configs = []
        
        current_min_ev = current.get('min_ev', 8.0)
        current_min_conf = current.get('min_confidence', 70.0)
        
        # Test diverse combinazioni
        ev_values = [
            max(3.0, current_min_ev - 2),
            current_min_ev - 1,
            current_min_ev,
            current_min_ev + 1,
            min(15.0, current_min_ev + 2)
        ]
        
        conf_values = [
            max(40.0, current_min_conf - 10),
            current_min_conf - 5,
            current_min_conf,
            current_min_conf + 5,
            min(90.0, current_min_conf + 10)
        ]
        
        # Genera combinazioni (max 9 per non esagerare)
        for ev in ev_values:
            for conf in conf_values:
                if abs(ev - current_min_ev) <= 2 and abs(conf - current_min_conf) <= 10:
                    configs.append({
                        'min_ev': ev,
                        'min_confidence': conf
                    })
        
        return configs[:9]  # Limita a 9 configurazioni
    
    def _simulate_performance(
        self,
        results: List[Dict],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simula performance con una configurazione."""
        filtered_results = []
        
        for result in results:
            # Filtra basandosi su parametri
            ev = result.get('ev', 0) or result.get('expected_value', 0)
            if isinstance(ev, float) and ev < 1.0:
                ev = ev * 100
            
            confidence = result.get('confidence', 0) or result.get('confidence_level', 0)
            
            # Applica filtri
            if ev >= config.get('min_ev', 8.0) and confidence >= config.get('min_confidence', 70.0):
                filtered_results.append(result)
        
        # Calcola performance filtrata
        return self._calculate_performance(filtered_results)
    
    def _generate_analysis(
        self,
        current: Dict[str, Any],
        projected: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """Genera analisi del miglioramento."""
        roi_diff = projected['roi'] - current['roi']
        bets_diff = projected['total_bets'] - current['total_bets']
        
        analysis = f"""
Analisi Ottimizzazione Parametri:

Performance Attuale:
- ROI: {current['roi']:.2f}%
- Win Rate: {current['win_rate']:.1f}%
- Totale Scommesse: {current['total_bets']}

Performance Proiettata (con nuovi parametri):
- ROI: {projected['roi']:.2f}% ({roi_diff:+.2f}%)
- Win Rate: {projected['win_rate']:.1f}%
- Totale Scommesse: {projected['total_bets']} ({bets_diff:+d})

Parametri Suggeriti:
- Min EV: {config.get('min_ev', 8.0):.1f}%
- Min Confidence: {config.get('min_confidence', 70.0):.1f}%

Miglioramento Stimato: {roi_diff:+.2f}% ROI
"""
        return analysis.strip()

